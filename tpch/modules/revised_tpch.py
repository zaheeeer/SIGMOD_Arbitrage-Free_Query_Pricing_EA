"""Revision 3, Part 4: the revised protocol on TPC-H (supervisor comments 1 to 6, decisions
D1, D2, D3, D6, D7, D8 applied to the second workload).

The 2.x TPC-H run (``run_tpch.py --protocol paper``, the default) is unchanged. This module
is used only by ``run_tpch.py --protocol revised``. It shares the verified pricing code of
the SQLShare pipeline (``sqlshare_rerun.pricing``), so both workloads use one
implementation of every mechanism, fit and buyer model.

What changes against the 2.x TPC-H run:

  Qirana support (D8)   neighbours are Qirana row updates and swap updates on a relation
                        chosen uniformly among the relations the catalog reads (here only
                        lineitem, as for SQLShare, where it is the market's tables); keys of
                        the TPC-H specification are never changed. 2.x: row deletions.
  Conflict sets          computed exactly: every asset is an additive aggregate, so its
                        answer on a neighbour is the true answer plus the integer change
                        of the changed rows (decimals are held in cents). The same change
                        gives the answer fingerprint used for the entropy partition.
  Qirana prices (D2)     weighted coverage, true uniform entropy gain log|C|/log|S|, and
                        Shannon entropy over the answer partition (2.x: |C|/|S| and the
                        singleton partition).
  Fit (D1)               exact breakpoint fit, no cap (2.x: grid capped at 60).
  Chawla et al. (D6)     LPIP with one LP per hyperedge (2.x: at most 60 LPs).
  Buyers (D7)            demand arms (same, shift10, shift25, shift50, redrawn) and
                        training sizes; the main arm is "same" at 800 training buyers.
  Attacker (D3)          the operational buyer: plans from query texts and prices only,
                        executes on purchased answers only. On TPC-H the query texts
                        guarantee month unions, roll-ups of additive measures, and
                        restriction on a visible shipmonth key. 2.x: exact month covers
                        without restriction, at most 8 plans re-executed.
  Answer cells (D1)      rows x output columns (2.x: rows only).
  Compute-metered (D1)   median of three timings (2.x: one timing).
  Costs (S8.c)           support build, scores, fit and planning seconds are recorded.
  Plans (S4)             every operational plan is logged with its sources.

Runs are resumable: the catalog and support are built once (``build.done``), and every
(arm, family, seed) task writes its own result file, so a stopped run continues where it
stopped when the same command is given again.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import sys
import time
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
SQLSHARE = os.path.normpath(os.path.join(HERE, "..", "..", "sqlshare"))
for _p in (HERE, SQLSHARE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_tpch_lattice as L                        # noqa: E402
import tpc_substrate as SUB                           # noqa: E402
from sqlshare_rerun import pricing as P               # noqa: E402
from sqlshare_rerun.operational import PurchasedAnswers  # noqa: E402

FAMILIES = ("answer_cells", "support_rows", "rows_x_log_support", "flat")
DOMAIN_SCHEMA = "__qirana_domain"
# Primary keys of the TPC-H specification (clause 1.4.2). DuckDB's dbgen declares none,
# so they are listed here. Only lineitem is read by the catalog.
TPCH_KEYS = {"lineitem": ("l_orderkey", "l_linenumber"), "orders": ("o_orderkey",),
             "customer": ("c_custkey",), "part": ("p_partkey",), "supplier": ("s_suppkey",),
             "partsupp": ("ps_partkey", "ps_suppkey"), "nation": ("n_nationkey",),
             "region": ("r_regionkey",)}
CATALOG_RELATIONS = ("lineitem",)                     # relations the 368 assets read
READ_COLS = ("l_shipdate", "l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice",
             "l_discount")
DIM_OF = {"returnflag": 1, "linestatus": 2, "shipmonth": 0}   # position in a Row tuple


@dataclasses.dataclass
class TpchSettings:
    n_assets: int = 200                 # build_assets_n target; 368 with the month views
    support_n: int = 400
    support_seed: int = 0
    neighbours: str = "qirana"          # "qirana" (D8) or "delete" (2.x neighbours)
    swap_ratio: float = 0.5
    fit: str = "breakpoints"
    max_lps: int = 0                    # 0 = one LP per hyperedge (D6)
    cip_eps: float = 0.5
    n_buyers: int = 800                 # test buyers, and the main training size
    seeds: tuple = (0, 1, 2, 3, 4)
    demand_arms: tuple = ("same", "shift10", "shift25", "shift50", "redrawn")
    main_arm: str = "same"
    train_sizes: tuple = (100, 200, 400, 800, 1600)
    metered_repeats: int = 3
    ilp_time_limit: float = 10.0
    expect_catalog: tuple = (368, 1046)  # checked when sf = 1; () to skip
    qirana_donors: str = "distinct"     # Part 7c, study D: "frequency" copies a random row's value

    def fingerprint(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("expect_catalog")
        if d["qirana_donors"] == "distinct":            # the P3 fingerprints stay valid
            d.pop("qirana_donors")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------
# exact row values and template rules
# ---------------------------------------------------------------------------------------
def _cents(x) -> int:
    d = Decimal(str(x)) * 100
    i = int(d)
    if d != i:
        raise ValueError(f"value {x!r} is not a whole number of cents")
    return i


def _month(d) -> dt.date:
    if isinstance(d, dt.datetime):
        d = d.date()
    return dt.date(d.year, d.month, 1)


def make_row(vals: dict) -> tuple:
    """Row = (month, returnflag, linestatus, qty_cents, extprice_cents, disc_cents)."""
    return (_month(vals["l_shipdate"]), str(vals["l_returnflag"]), str(vals["l_linestatus"]),
            _cents(vals["l_quantity"]), _cents(vals["l_extendedprice"]), _cents(vals["l_discount"]))


def passes(template: str, row) -> bool:
    """Template predicate (TEMPLATES[...]['extra_where']) on exact values."""
    if template == "q1":
        return True
    if template == "q6":                # l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24
        return 5 <= row[5] <= 7 and row[3] < 2400
    raise KeyError(template)


def contribution(template: str, row) -> tuple:
    """(count, measures...) of one row, as integers (units of 0.01 or 0.0001)."""
    _, _, _, q, e, d = row
    if template == "q1":                # sum_qty, sum_base_price, sum_disc_price, count_order
        return (1, q, e, e * (100 - d), 1)
    return (1, e * d)                   # q6: revenue = sum(l_extendedprice * l_discount)


def has_count_measure(template: str) -> bool:
    return template == "q1"


def group_key(asset, row) -> tuple:
    dims = L.TEMPLATES[asset.template]["group_dims"]
    return tuple(row[DIM_OF[g]] for g in dims if g in asset.group)


# ---------------------------------------------------------------------------------------
# conflict sets and answer fingerprints of one neighbour (exact)
# ---------------------------------------------------------------------------------------
class DeltaEngine:
    """Answer change of every asset on a neighbour that replaces some rows.

    ``base_cnt[template][(month, rf, ls)]`` is the number of rows of D that pass the
    template predicate in that cell. An asset's answer on D' differs from its answer on D
    exactly when, for some group, the group appears or disappears or a measure changes;
    the fingerprint lists every such group with its new state, so two neighbours share a
    fingerprint exactly when the asset returns the same answer on both.
    """

    def __init__(self, assets, base_cnt):
        self.assets = list(assets)
        self.base_cnt = base_cnt
        self._cnt_cache = {}

    def _base_count(self, ai, key) -> int:
        ck = (ai, key)
        if ck not in self._cnt_cache:
            a = self.assets[ai]
            tot = 0
            for (m, rf, ls), c in self.base_cnt[a.template].items():
                if m in a.months and group_key(a, (m, rf, ls)) == key:
                    tot += c
            self._cnt_cache[ck] = tot
        return self._cnt_cache[ck]

    def asset_change(self, ai, changes):
        """``changes`` = [(old_row or None, new_row or None)]. Returns the fingerprint
        (a sorted tuple) or None when the answer is unchanged."""
        a = self.assets[ai]
        delta = {}
        for old, new in changes:
            for row, sign in ((old, -1), (new, 1)):
                if row is None or row[0] not in a.months or not passes(a.template, row):
                    continue
                k = group_key(a, row)
                c = contribution(a.template, row)
                cur = delta.get(k, (0,) * len(c))
                delta[k] = tuple(x + sign * y for x, y in zip(cur, c))
        entries = []
        grouped = bool(a.group)
        for k, dv in delta.items():
            if not any(dv):
                continue
            c0 = self._base_count(ai, k)
            c1 = c0 + dv[0]
            meas = dv[1:]
            if grouped:
                if c0 > 0 and c1 == 0:
                    entries.append((k, ("gone",)))
                elif c1 > 0 and (c0 == 0 or any(meas)):
                    entries.append((k, ("val",) + tuple(meas)))
            else:                       # no GROUP BY: one row, NULL sums when empty
                if c1 == 0 and c0 > 0:
                    entries.append((k, ("null",)))
                elif c1 > 0 and (c0 == 0 or any(meas)):
                    entries.append((k, ("val",) + tuple(meas)))
        if not entries:
            return None
        return tuple(sorted(entries, key=repr))

    def neighbour(self, changes):
        """{asset index: fingerprint} for every asset whose answer changes."""
        out = {}
        for ai in range(len(self.assets)):
            fp = self.asset_change(ai, changes)
            if fp is not None:
                out[ai] = hashlib.sha1(repr(fp).encode()).hexdigest()[:16]
        return out


def base_counts(con) -> dict:
    rows = con.execute(
        "SELECT CAST(date_trunc('month', l_shipdate) AS DATE), l_returnflag, l_linestatus, count(*), "
        "count(*) FILTER (WHERE l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24) "
        "FROM lineitem GROUP BY ALL").fetchall()
    out = {"q1": {}, "q6": {}}
    for m, rf, ls, c1, c6 in rows:
        out["q1"][(m, str(rf), str(ls))] = int(c1)
        if c6:
            out["q6"][(m, str(rf), str(ls))] = int(c6)
    return out


# ---------------------------------------------------------------------------------------
# Qirana neighbours on lineitem (Deep and Koutris, SIGMOD 2017, Section 4)
# ---------------------------------------------------------------------------------------
class LineitemNeighbours:
    """Row and swap updates of one relation, drawn as in ``sqlshare_rerun.support``:
    a swap with probability ``swap_ratio``; each non-key attribute chosen with probability
    0.5, at least one; a row update takes the new values uniformly from the attribute's
    active domain; a draw that leaves the database unchanged is rejected."""

    def __init__(self, con, relation="lineitem", swap_ratio=0.5, donors="distinct"):
        if donors not in ("distinct", "frequency"):
            raise ValueError(f"unknown donors {donors!r}")
        self.con, self.rel, self.swap_ratio, self.donors = con, relation, swap_ratio, donors
        cols = [d[0] for d in con.execute(f"SELECT * FROM {relation} LIMIT 0").description]
        keys = set(TPCH_KEYS.get(relation, ()))
        self.cols = cols
        self.eligible = [c for c in cols if c not in keys]
        self.count = int(con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0])
        mx = con.execute(f"SELECT max(rowid) FROM {relation}").fetchone()[0]
        if self.count and int(mx) != self.count - 1:
            raise RuntimeError(f"{relation}: row ids are not contiguous")
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {DOMAIN_SCHEMA}")
        self.dom = []
        for c in self.eligible:
            name = f"{DOMAIN_SCHEMA}.d_{relation}_{c}"
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS "
                        f"SELECT DISTINCT {c} AS v FROM {relation} ORDER BY v NULLS LAST")
            self.dom.append((name, int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])))
        self.domain_sizes = {c: n for c, (_, n) in zip(self.eligible, self.dom)}

    def draw(self, rng):
        swap = self.count >= 2 and rng.random() < self.swap_ratio
        while True:
            mask = rng.random(len(self.eligible)) < 0.5
            if mask.any():
                break
        chosen = tuple(int(j) for j in np.flatnonzero(mask))
        if swap:
            r1 = int(rng.integers(self.count))
            r2 = int(rng.integers(self.count - 1))
            r2 = r2 + 1 if r2 >= r1 else r2
            return ("swap", r1, r2, chosen)
        r = int(rng.integers(self.count))
        if self.donors == "frequency":           # study D: a random row's value (frequency weighted)
            donors = tuple(int(rng.integers(self.count)) for _ in chosen)
        else:
            donors = tuple(int(rng.integers(self.dom[j][1])) for j in chosen)
        return ("row", r, chosen, donors)

    def _row(self, rid):
        got = self.con.execute(f"SELECT * FROM {self.rel} WHERE rowid = {int(rid)}").fetchone()
        return dict(zip(self.cols, got))

    def changes(self, spec):
        """[(old values, new values)] of the changed rows, or None if D' = D."""
        if spec[0] == "row":
            _, r, chosen, donors = spec
            old = self._row(r)
            new = dict(old)
            for j, d in zip(chosen, donors):
                if self.donors == "frequency":
                    new[self.eligible[j]] = self.con.execute(
                        f"SELECT {self.eligible[j]} FROM {self.rel} WHERE rowid = {int(d)}").fetchone()[0]
                else:
                    new[self.eligible[j]] = self.con.execute(
                        f"SELECT v FROM {self.dom[j][0]} WHERE rowid = {int(d)}").fetchone()[0]
            if all(old[self.eligible[j]] == new[self.eligible[j]] for j in chosen):
                return None
            return [(old, new)]
        _, r1, r2, chosen = spec
        v1, v2 = self._row(r1), self._row(r2)
        names = [self.eligible[j] for j in chosen]
        if all(v1[c] == v2[c] for c in names):
            return None
        n1, n2 = dict(v1), dict(v2)
        for c in names:
            n1[c], n2[c] = v2[c], v1[c]
        return [(v1, n1), (v2, n2)]


def build_support(con, assets, s: TpchSettings):
    """Support set S, conflict sets C(Q) and answer fingerprints for every asset."""
    t0 = time.perf_counter()
    eng = DeltaEngine(assets, base_counts(con))
    rng = np.random.default_rng(s.support_seed)
    accepted, changed, kinds, rejected = [], [], {"row": 0, "swap": 0, "delete": 0}, 0
    info = {}
    if s.neighbours == "qirana":
        # Relation chosen uniformly among the relations the catalog reads (one here).
        gen = LineitemNeighbours(con, CATALOG_RELATIONS[0], s.swap_ratio, s.qirana_donors)
        info["domain_sizes"] = gen.domain_sizes
        info["eligible_columns"] = gen.eligible
        guard = 0
        while len(accepted) < s.support_n:
            guard += 1
            if guard > 100 * s.support_n:
                raise RuntimeError("too many rejected neighbours")
            rel = CATALOG_RELATIONS[int(rng.integers(len(CATALOG_RELATIONS)))]
            spec = gen.draw(rng)
            ch = gen.changes(spec)
            if ch is None:
                rejected += 1
                continue
            rows = [(make_row(o), make_row(n)) for o, n in ch]
            accepted.append([rel] + list(spec))
            kinds[spec[0]] += 1
            changed.append(eng.neighbour(rows))
    elif s.neighbours == "delete":
        cols = [d[0] for d in con.execute("SELECT * FROM lineitem LIMIT 0").description]
        cnt = int(con.execute("SELECT count(*) FROM lineitem").fetchone()[0])
        for r in sorted(rng.choice(cnt, size=min(s.support_n, cnt), replace=False).tolist()):
            old = dict(zip(cols, con.execute(f"SELECT * FROM lineitem WHERE rowid = {int(r)}").fetchone()))
            accepted.append(["lineitem", "delete", int(r)])
            kinds["delete"] += 1
            changed.append(eng.neighbour([(make_row(old), None)]))
    else:
        raise ValueError(s.neighbours)
    n = len(accepted)
    edges = [frozenset(k for k in range(n) if ai in changed[k]) for ai in range(len(assets))]
    fps = [{k: changed[k][ai] for k in edges[ai]} for ai in range(len(assets))]
    info.update({"neighbours": s.neighbours, "n": n, "kinds": kinds, "rejected_draws": rejected,
                 "accepted": accepted, "seconds": time.perf_counter() - t0,
                 "assets_with_empty_conflict_set": int(sum(1 for e in edges if not e)),
                 "neighbours_changing_no_asset": int(sum(1 for c in changed if not c)),
                 "mean_conflict_set_size": float(np.mean([len(e) for e in edges])) if edges else 0.0})
    return edges, fps, info


# ---------------------------------------------------------------------------------------
# operational buyer on TPC-H (D3): query texts and prices only
# ---------------------------------------------------------------------------------------
def op_structures(assets):
    """For every target, the sources the query texts allow and how. A source s can
    contribute to target t when both use the same template and t's grouping keys are a
    subset of s's (measures are sums and counts, so they roll up). If s exposes shipmonth,
    the buyer can keep only the months it needs (restriction); otherwise s must be used
    whole, so its months must lie inside t's. Built from the asset definitions alone."""
    out = {}
    for ti, t in enumerate(assets):
        cands = []
        for si, a in enumerate(assets):
            if si == ti or a.template != t.template or not t.group <= a.group:
                continue
            restr = "shipmonth" in a.group
            if restr:
                ms = a.months & t.months
                if ms:
                    cands.append((si, True, frozenset(ms)))
            elif a.months <= t.months:
                cands.append((si, False, frozenset(a.months)))
        out[ti] = cands
    return out


def plan_target(ti, t_months, cands, prices, time_limit=10.0):
    """Cheapest guaranteed plan for one target, or None. Each month of the target must
    come from exactly one source: whole sources may not overlap, restrictable sources drop
    the months others supply. Returns (cost, [(source, restricted, months used)])."""
    pt = prices[ti]
    cheap = [(si, r, ms) for si, r, ms in cands if prices[si] < pt - 1e-9]
    if not cheap:
        return None
    months = sorted(t_months)
    if not set(months) <= set().union(*[ms for _, _, ms in cheap]):
        return None
    mi = {m: k for k, m in enumerate(months)}
    nrow = len(months)
    A_whole = np.zeros((nrow, len(cheap)))
    A_all = np.zeros((nrow, len(cheap)))
    for c, (si, r, ms) in enumerate(cheap):
        for m in ms:
            A_all[mi[m], c] = 1.0
            if not r:
                A_whole[mi[m], c] = 1.0
    cons = [LinearConstraint(A_all, lb=np.ones(nrow), ub=np.inf)]
    if A_whole.any():
        cons.append(LinearConstraint(A_whole, lb=-np.inf, ub=np.ones(nrow)))
    cost = np.array([prices[si] for si, _, _ in cheap], float)
    res = milp(c=cost, constraints=cons, integrality=np.ones(len(cheap)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return None
    chosen = [cheap[k] for k in range(len(cheap)) if res.x[k] > 0.5]
    total = float(sum(prices[si] for si, _, _ in chosen))
    if total >= pt - 1e-9:
        return None
    left = set(months)
    plan = []
    for si, r, ms in chosen:
        if not r:
            plan.append((si, False, frozenset(ms)))
            left -= ms
    for si, r, ms in sorted((x for x in chosen if x[1]), key=lambda x: x[0]):
        use = frozenset(ms & left)
        left -= use
        plan.append((si, True, use))
    if left:
        return None
    return total, plan


def execute_plan(target, plan, assets, held: PurchasedAnswers):
    """Rebuild the target answer from purchased answers only."""
    gcols = [g for g in L.TEMPLATES[target.template]["group_dims"] if g in target.group]
    mcols = list(target.measures)
    parts = []
    for si, restricted, months in plan:
        df = held[si]
        if restricted:
            want = pd.to_datetime(sorted(months))
            df = df[pd.to_datetime(df["shipmonth"]).isin(want).to_numpy()]
        parts.append(df.loc[:, gcols + mcols])
    allp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=gcols + mcols)
    allp = allp.dropna(subset=mcols, how="all")
    if gcols:
        out = allp.groupby(gcols, dropna=False, as_index=False)[mcols].sum()
        return out.sort_values(gcols).reset_index(drop=True).loc[:, gcols + mcols]
    if len(allp) == 0:                  # SQL: sums over no rows are NULL, count is 0
        row = {c: (0 if c == "count_order" else np.nan) for c in mcols}
        return pd.DataFrame([row]).loc[:, mcols]
    return pd.DataFrame([allp[mcols].sum()]).loc[:, mcols]


def answers_equal(got, want) -> bool:
    if got.shape != want.shape or list(got.columns) != list(want.columns):
        return False
    for c in got.columns:
        a, b = got[c], want[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True):
                return False
        elif c == "shipmonth":
            if not pd.to_datetime(a).reset_index(drop=True).equals(pd.to_datetime(b).reset_index(drop=True)):
                return False
        elif not a.astype(str).reset_index(drop=True).equals(b.astype(str).reset_index(drop=True)):
            return False
    return True


# ---------------------------------------------------------------------------------------
# build stage (cached)
# ---------------------------------------------------------------------------------------
def build_market(con, s: TpchSettings, sf_is_one: bool, log=print):
    """Catalog, answers, pairs, timings, support, scores and valuation families."""
    costs = {}
    t0 = time.perf_counter()
    months = L.month_universe(con)
    A, _ = L.build_assets_n(months, s.n_assets, seed=0)
    views = [L.Asset(t, frozenset(spec["group_dims"]), frozenset({m}))
             for t, spec in L.TEMPLATES.items() for m in months]
    A2 = list(A) + [v for v in views if v not in set(A)]
    ans = {i: L.evaluate(con, a) for i, a in enumerate(A2)}
    sup = SUB.support_tables(con, L.TEMPLATES)
    ssz = np.array([SUB.support_size(a, sup) for a in A2], float)
    rowsz = np.array([len(ans[i]) for i in range(len(A2))], float)
    ncols = np.array([len(ans[i].columns) for i in range(len(A2))], float)
    pairs = [(i, j) for i in range(len(A2)) for j in range(len(A2)) if i != j and L.determines(A2[i], A2[j])]
    if sf_is_one and s.expect_catalog:
        assert (len(A2), len(pairs)) == tuple(s.expect_catalog), (len(A2), len(pairs))
    costs["catalog_seconds"] = time.perf_counter() - t0
    log(f"catalog: {len(A2)} assets, {len(pairs)} pairs")

    t0 = time.perf_counter()
    et = np.zeros(len(A2))
    for i, a in enumerate(A2):
        sql = L.asset_sql(a)
        ts = []
        for _ in range(max(1, s.metered_repeats)):
            t1 = time.perf_counter()
            con.execute(sql).fetchall()
            ts.append(time.perf_counter() - t1)
        et[i] = float(np.median(ts))
    costs["timing_seconds"] = time.perf_counter() - t0

    edges, fps, sup_info = build_support(con, A2, s)
    costs["support_seconds"] = sup_info["seconds"]
    log(f"support: |S| = {sup_info['n']} ({s.neighbours}), assets with an empty conflict set "
        f"{sup_info['assets_with_empty_conflict_set']}")

    t0 = time.perf_counter()
    q = P.qirana_prices(edges, sup_info["n"], changed=fps)
    view_idx = [i for i, a in enumerate(A2)
                if len(a.months) == 1 and a.group == frozenset(L.TEMPLATES[a.template]["group_dims"])]
    import ports as PORTS
    qm = PORTS.querymarket_prices(A2, view_idx, {i: float(ssz[i]) for i in view_idx}, SUB.min_cost_cover)
    unsellable = int((~np.isfinite(qm)).sum())
    scores = {"uniform": np.ones(len(A2)), "size_proportional": rowsz.copy(),
              "compute_metered": et.copy(),
              "qirana_weighted_coverage": q["qirana_weighted_coverage"],
              "qirana_uniform_entropy_gain": q["qirana_uniform_entropy_gain"],
              "qirana_shannon": np.maximum(q["qirana_shannon"], 0.0),
              "querymarket_viewcover": qm}
    costs["scores_seconds"] = time.perf_counter() - t0
    info = P.info_families(rowsz, ssz, ncols)
    t0 = time.perf_counter()
    structures = op_structures(A2)
    costs["op_structure_seconds"] = time.perf_counter() - t0
    market = {"assets": A2, "answers": ans, "pairs": pairs, "scores": scores, "edges": edges,
              "support_n": sup_info["n"], "info": info, "structures": structures,
              "labels": [a.label() for a in A2]}
    meta = {"n_assets": len(A2), "n_pairs": len(pairs), "n_months": len(months),
            "querymarket_unsellable": unsellable, "costs": costs,
            "single_source_op_pairs": int(sum(1 for t, c in structures.items() for si, r, ms in c
                                               if ms == A2[t].months))}
    return market, sup_info, meta


# ---------------------------------------------------------------------------------------
# one (arm, family, seed) task
# ---------------------------------------------------------------------------------------
_MARKET, _VERIFIED = {}, {}


def _load(path):
    if path not in _MARKET:
        _MARKET.clear()
        _VERIFIED.clear()
        with open(path, "rb") as fh:
            _MARKET[path] = pickle.load(fh)
    return _MARKET[path]


def task_name(arm, family, seed):
    return f"{arm}__{family}__s{seed}"


def run_task(args):
    """Fit, freeze and evaluate every mechanism for one (arm, family, seed, n_train,
    planned) and write the records to the task folder. Returns the task name."""
    path, out_dir, arm, family, seed, n_train, planned = args
    name = task_name(arm, family, seed)
    final = os.path.join(out_dir, name + ".csv")
    if os.path.exists(final):
        return name
    d = _load(path)
    cfg = d["cfg"]
    n = len(d["assets"])
    info = d["info"][family]
    demand_arm = arm.split("_n")[0]
    (tr_t, tr_v), (te_t, te_v), stats = P.buyer_arms(info, 1000 * seed + 7, demand_arm, n_train, cfg["n_buyers"])
    prices, fit_s = {}, {}
    for mech, sc in d["scores"].items():
        t0 = time.perf_counter()
        prices[mech], _ = P.scaled_prices(cfg["fit"], tr_t, tr_v, sc)
        fit_s[mech] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mono = P.monopoly_prices(tr_t, tr_v, n)
    prices["monopoly_per_asset"] = mono
    fit_s["monopoly_per_asset"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prices["monotone_constrained"] = P.enforce_monotone(mono, d["pairs"])
    fit_s["monotone_constrained"] = time.perf_counter() - t0
    ch_s = {}
    prices.update(P.chawla_all(d["edges"], tr_t, tr_v, d["support_n"], max_lps=cfg["max_lps"],
                               eps=cfg["cip_eps"], seconds=ch_s))
    fit_s.update(ch_s)
    ref_train = P.score(tr_t, tr_v, prices["monopoly_per_asset"])["revenue"]
    ratio = cfg["n_buyers"] / n_train
    ref = ref_train * ratio
    ref_test_opt = P.score(te_t, te_v, P.monopoly_prices(te_t, te_v, n))["revenue"]   # Part 5
    rows, plan_rows, sweep_m, sweep_c = [], [], [], []
    for mech, p in prices.items():
        p = np.asarray(p, float)
        eff = None
        nai = P.score(te_t, te_v, p)
        ins = P.score(tr_t, tr_v, p)["revenue"]
        ma = P.audit_monotone(p, d["pairs"])
        row = {"arm": arm, "family": family, "seed": seed, "mechanism": mech,
               "n_train": n_train, "n_test": cfg["n_buyers"], "planned": planned,
               "revenue_train_norm": ins / max(ref_train, 1e-12),
               "revenue_naive_norm": nai["revenue"] / max(ref, 1e-12),
               "surplus_naive_norm": nai["surplus"] / max(ref, 1e-12),
               "welfare_naive_norm": nai["welfare"] / max(ref, 1e-12),
               "served_naive": nai["served"],
               "generalization_gap": 1 - nai["revenue"] / max(ins * ratio, 1e-12),
               "mono_violation_rate": ma["violation_rate"],
               "revenue_train": ins, "revenue_naive": nai["revenue"], "ref_monopoly_train": ref_train,
               "ref_monopoly_test": ref_test_opt, "fit_seconds": fit_s.get(mech, np.nan), **stats}
        if planned:
            t0 = time.perf_counter()
            cost = np.full(n, np.inf)
            ok_n = bad_n = 0
            for ti in range(n):
                got = plan_target(ti, d["assets"][ti].months, d["structures"][ti], p, cfg["ilp_time_limit"])
                if got is None:
                    continue
                c_, plan = got
                vkey = (ti, tuple(plan))
                if vkey not in _VERIFIED:
                    try:
                        held = PurchasedAnswers(d["answers"], [si for si, _, _ in plan])
                        rebuilt = execute_plan(d["assets"][ti], plan, d["assets"], held)
                        _VERIFIED[vkey] = answers_equal(rebuilt, d["answers"][ti])
                    except Exception:
                        _VERIFIED[vkey] = False
                ok = _VERIFIED[vkey]
                if ok:
                    cost[ti] = c_
                    ok_n += 1
                else:
                    bad_n += 1
                pr = float(p[ti])
                plan_rows.append({"arm": arm, "family": family, "seed": seed, "mechanism": mech,
                                  "attacker": "operational", "target": ti, "target_label": d["labels"][ti],
                                  "price": pr, "cost": float(c_),
                                  "gain": (pr - c_) / pr if pr > 0 else 0.0, "n_sources": len(plan),
                                  "sources": ";".join(str(si) for si, _, _ in plan),
                                  "restricted": ";".join(str(si) for si, r, _ in plan if r),
                                  "verified": bool(ok)})
            eff = np.minimum(p, cost)
            stra = P.score(te_t, te_v, eff)
            with np.errstate(invalid="ignore", divide="ignore"):
                gain = np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0)
            row.update({
                "revenue_strategic_op_norm": stra["revenue"] / max(ref, 1e-12),
                "surplus_strategic_op_norm": stra["surplus"] / max(ref, 1e-12),
                "welfare_strategic_op_norm": stra["welfare"] / max(ref, 1e-12),
                "served_strategic_op": stra["served"], "revenue_strategic_op": stra["revenue"],
                "comb_violation_rate_op": float(np.mean(gain > 1e-6)),
                "comb_mean_gain_op": float(np.mean(gain)), "comb_max_gain_op": float(gain.max()),
                "plans_verified_op": ok_n, "plans_failed_op": bad_n,
                "plan_seconds_operational": time.perf_counter() - t0})
        rows.append(row)
        if cfg.get("sweep"):                     # Part 7c, study C
            from sqlshare_rerun.study_core import sweep
            views = {"naive": p, **({"operational": eff} if eff is not None else {})}
            m_rows, c_rows = sweep(views, te_t, te_v, ref, {"arm": arm, "family": family, "seed": seed,
                                                            "mechanism": mech})
            sweep_m += m_rows
            sweep_c += c_rows
    if cfg.get("sweep"):
        _write_csv_atomic(pd.DataFrame(sweep_m), os.path.join(out_dir, name + ".matched.csv"))
        _write_csv_atomic(pd.DataFrame(sweep_c), os.path.join(out_dir, name + ".curve.csv"))
    _write_csv_atomic(pd.DataFrame(plan_rows), os.path.join(out_dir, name + ".plans.csv"))
    _write_csv_atomic(pd.DataFrame(rows), final)        # written last: marks the task done
    return name


def _write_csv_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def task_list(s: TpchSettings):
    """Main arm at the main size (planned); other arms and sizes (revenue out of sample)."""
    tasks = [(s.main_arm, f, sd, s.n_buyers, True) for f in FAMILIES for sd in s.seeds]
    tasks += [(a, f, sd, s.n_buyers, False) for a in s.demand_arms if a != s.main_arm
              for f in FAMILIES for sd in s.seeds]
    tasks += [(f"{s.main_arm}_n{n}", f, sd, n, False) for n in s.train_sizes if n != s.n_buyers
              for f in FAMILIES for sd in s.seeds]
    return tasks


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def run_revised(con, out_dir, s: TpchSettings, workers=1, sf_is_one=True, log=print):
    """Build (once), run every task (resumable), and write the summaries."""
    os.makedirs(out_dir, exist_ok=True)
    fp = s.fingerprint()
    marker = os.path.join(out_dir, "build.done")
    mpath = os.path.join(out_dir, "market.pkl")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            done = json.load(fh)
        if done.get("fingerprint") != fp:
            raise RuntimeError(f"{out_dir} holds a run with other settings ({done.get('fingerprint')}); "
                               f"use another --out folder")
        log(f"build: reusing {mpath}")
    else:
        t0 = time.perf_counter()
        market, sup_info, meta = build_market(con, s, sf_is_one, log)
        market["cfg"] = {"fit": s.fit, "max_lps": s.max_lps, "cip_eps": s.cip_eps,
                         "n_buyers": s.n_buyers, "ilp_time_limit": s.ilp_time_limit}
        tmp = mpath + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(market, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, mpath)
        with open(os.path.join(out_dir, "pricing_support.json"), "w", encoding="utf-8") as fh:
            json.dump(sup_info, fh, indent=1, default=str)
        meta["settings"] = dataclasses.asdict(s)
        meta["fingerprint"] = fp
        meta["build_seconds"] = time.perf_counter() - t0
        with open(os.path.join(out_dir, "build.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, default=str)
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp}, fh)
    tdir = os.path.join(out_dir, "tasks")
    os.makedirs(tdir, exist_ok=True)
    tasks = [(mpath, tdir) + t for t in task_list(s)]
    todo = [t for t in tasks if not os.path.exists(os.path.join(tdir, task_name(*t[2:5]) + ".csv"))]
    log(f"tasks: {len(tasks)} in total, {len(tasks) - len(todo)} already done, {len(todo)} to run")
    t0 = time.perf_counter()
    if todo:
        from sqlshare_rerun.parallel import run_pool
        run_pool(run_task, todo, workers, "TPC-H fit, freeze, and evaluate runs")
    log(f"tasks finished in {time.perf_counter() - t0:.0f}s")
    return collect(out_dir, s)


def collect(out_dir, s: TpchSettings):
    tdir = os.path.join(out_dir, "tasks")
    names = [task_name(*t[:3]) for t in task_list(s)]
    runs = pd.concat([pd.read_csv(os.path.join(tdir, nm + ".csv")) for nm in names], ignore_index=True)
    plans = [pd.read_csv(os.path.join(tdir, nm + ".plans.csv")) for nm in names
             if os.path.getsize(os.path.join(tdir, nm + ".plans.csv")) > 2]
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    if plans:
        pd.concat(plans, ignore_index=True).to_csv(os.path.join(out_dir, "plans_log.csv"), index=False)
    cols = [c for c in runs.columns if c not in ("arm", "family", "seed", "mechanism", "planned", "n_train")
            and runs[c].dtype.kind in "fcbiu"]
    g = runs.groupby(["arm", "n_train", "mechanism"])[cols].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.reset_index().to_csv(os.path.join(out_dir, "summary_by_arm_size_mechanism.csv"), index=False)
    runs.groupby(["arm", "n_train", "family", "mechanism"])[cols].mean().reset_index().to_csv(
        os.path.join(out_dir, "summary_by_arm_size_family_mechanism.csv"), index=False)
    return runs


# ---------------------------------------------------------------------------------------
# small synthetic lineitem (tests and smoke checks; never used for reported results)
# ---------------------------------------------------------------------------------------
def synthetic_lineitem(con, n_rows=3000, n_months=18, seed=0, start=dt.date(1994, 1, 1)):
    """A lineitem table with the TPC-H column names and types and random values."""
    rng = np.random.default_rng(seed)
    days = (dt.date(start.year + (start.month - 1 + n_months) // 12,
                    (start.month - 1 + n_months) % 12 + 1, 1) - start).days
    ship = [start + dt.timedelta(days=int(x)) for x in rng.integers(0, days, n_rows)]
    qty = rng.integers(1, 51, n_rows)
    price = rng.integers(90000, 210000, n_rows)              # cents per unit
    df = pd.DataFrame({
        "l_orderkey": np.repeat(np.arange(1, n_rows // 4 + 2), 4)[:n_rows],
        "l_partkey": rng.integers(1, 200, n_rows), "l_suppkey": rng.integers(1, 10, n_rows),
        "l_linenumber": np.tile(np.arange(1, 5), n_rows // 4 + 1)[:n_rows],
        "l_quantity": qty.astype(float), "l_extendedprice": (qty * price) / 100.0,
        "l_discount": rng.integers(0, 11, n_rows) / 100.0, "l_tax": rng.integers(0, 9, n_rows) / 100.0,
        "l_returnflag": rng.choice(["R", "A", "N"], n_rows), "l_linestatus": rng.choice(["O", "F"], n_rows),
        "l_shipdate": ship, "l_commitdate": ship, "l_receiptdate": ship,
        "l_shipinstruct": rng.choice(["DELIVER IN PERSON", "NONE", "TAKE BACK RETURN"], n_rows),
        "l_shipmode": rng.choice(["AIR", "MAIL", "SHIP", "TRUCK"], n_rows),
        "l_comment": [f"c{int(x)}" for x in rng.integers(0, n_rows, n_rows)],
    })
    con.register("_li", df)
    con.execute("""CREATE OR REPLACE TABLE lineitem AS SELECT
        CAST(l_orderkey AS BIGINT) l_orderkey, CAST(l_partkey AS BIGINT) l_partkey,
        CAST(l_suppkey AS BIGINT) l_suppkey, CAST(l_linenumber AS BIGINT) l_linenumber,
        CAST(l_quantity AS DECIMAL(15,2)) l_quantity, CAST(l_extendedprice AS DECIMAL(15,2)) l_extendedprice,
        CAST(l_discount AS DECIMAL(15,2)) l_discount, CAST(l_tax AS DECIMAL(15,2)) l_tax,
        CAST(l_returnflag AS VARCHAR) l_returnflag, CAST(l_linestatus AS VARCHAR) l_linestatus,
        CAST(l_shipdate AS DATE) l_shipdate, CAST(l_commitdate AS DATE) l_commitdate,
        CAST(l_receiptdate AS DATE) l_receiptdate, l_shipinstruct, l_shipmode, l_comment FROM _li""")
    con.unregister("_li")

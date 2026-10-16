"""Revision 3, Part 7c: workload-independent pieces of the four approved studies.

Shared by the SQLShare driver (``studies.py``) and the TPC-H driver
(``tpch/modules/studies_tpch.py``), so both workloads use one implementation.

Study A (comment 5): the six Chawla et al. constructions under a factorial of
  protocol      in_sample (fitted and scored on the same buyers, as in Chawla et al.)
                or held_out (scored on independent buyers from the same demand)
  valuations    ours (lognormal times information^0.7, per valuation family) or the
                sampled bundle valuations of Chawla et al.: Uniform[1, k], k in {10, 100},
                and Zipf with exponent a in {1.5, 2.5}
  buyers        per_query (one buyer per query, as their one hyperedge per query) or
                popularity (800 buyers who pick a query by popularity, our protocol)
  LP cap        none (one LP per buyer) or 60 LPs
  support       the workload's pricing support (Qirana updates) or row deletions
Study C (comment 8e): every price vector is multiplied by m on a fine grid; welfare is
  compared at matched revenue (the smallest m that reaches the revenue level, which gives
  the highest welfare among the multipliers that reach it, since welfare does not rise
  with m) and at matched coverage (the largest m that still serves the share of buyers).
  A match at the edge of the grid (0.1 or 3.0) is flagged (at_grid_edge) and left out of
  the summary means, because the level may be met only outside the grid.
  Strategic views use the effective prices of the attacker at m = 1 multiplied by m:
  scaling every price by m scales every plan cost by m, so the same plans undercut.
"""
from __future__ import annotations

import hashlib
import itertools
import time

import numpy as np

from . import pricing as P

VALUATIONS = ("ours", "uniform10", "uniform100", "zipf1.5", "zipf2.5")
BUYERS = ("per_query", "popularity")
LP_CAPS = (0, 60)
PROTOCOLS_A = ("in_sample", "held_out")
SWEEP_GRID = np.round(np.arange(10, 301) / 100.0, 2)            # 0.10, 0.11, ..., 3.00
CURVE_GRID = np.round(np.arange(1, 31) / 10.0, 1)               # 0.1, 0.2, ..., 3.0
REVENUE_LEVELS = (0.30, 0.40, 0.50)
COVERAGE_LEVELS = (0.25, 0.50, 0.75)


def _code(x) -> int:
    return int(hashlib.sha1(str(x).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------------------
def chawla_values(kind: str, n: int, rng) -> np.ndarray:
    """Sampled bundle valuations of Chawla et al. (Section 6): Uniform[1, k] (real valued)
    or Zipf with exponent a (numpy's Zipf: integers >= 1)."""
    if kind.startswith("uniform"):
        return rng.uniform(1.0, float(kind[len("uniform"):]), size=n)
    if kind.startswith("zipf"):
        return rng.zipf(float(kind[len("zipf"):]), size=n).astype(float)
    raise ValueError(f"unknown valuation {kind!r}")


def study_a_buyers(info, seed: int, valuation: str, buyers: str, n_buyers: int = 800):
    """Training and evaluation buyers of one study-A cell: ((t, v), (t, v)).

    popularity + ours is exactly the main protocol's "same" arm (same seed rule), so that
    cell reproduces the main runs. Chawla valuations keep the popularity targets and redraw
    the values. per_query has one buyer per query in both samples; the evaluation sample
    is an independent second draw of the values.
    """
    m = len(info)
    ss = np.random.SeedSequence([int(seed), _code(valuation), _code(buyers)])
    r_tr, r_te = [np.random.default_rng(x) for x in ss.spawn(2)]
    if buyers == "popularity":
        (tr_t, tr_v), (te_t, te_v), _ = P.buyer_arms(info, 1000 * int(seed) + 7, "same", n_buyers, n_buyers)
        if valuation != "ours":
            tr_v = chawla_values(valuation, len(tr_t), r_tr)
            te_v = chawla_values(valuation, len(te_t), r_te)
    elif buyers == "per_query":
        tr_t = np.arange(m)
        te_t = np.arange(m)
        if valuation == "ours":
            inf = np.asarray(info, float) ** 0.7
            tr_v = 100.0 * r_tr.lognormal(0.0, 0.6, size=m) * inf
            te_v = 100.0 * r_te.lognormal(0.0, 0.6, size=m) * inf
        else:
            tr_v = chawla_values(valuation, m, r_tr)
            te_v = chawla_values(valuation, m, r_te)
    else:
        raise ValueError(f"unknown buyers {buyers!r}")
    return (np.asarray(tr_t), np.asarray(tr_v, float)), (np.asarray(te_t), np.asarray(te_v, float))


def chawla_cell(edges, support_n: int, info, seed: int, valuation: str, buyers: str, lp_cap: int,
                eps: float = 0.5, n_buyers: int = 800, labels: dict | None = None) -> list:
    """One study-A cell: fit the six constructions once, score in sample and held out."""
    (tr_t, tr_v), (te_t, te_v) = study_a_buyers(info, seed, valuation, buyers, n_buyers)
    sec = {}
    t0 = time.perf_counter()
    prices = P.chawla_all(edges, tr_t, tr_v, support_n, max_lps=lp_cap, eps=eps, seconds=sec)
    total_fit = time.perf_counter() - t0
    mono = P.monopoly_prices(tr_t, tr_v, len(edges))
    n_lps = len(tr_t) if (not lp_cap or len(tr_t) <= lp_cap) else int(lp_cap)
    rows = []
    for proto, (t, v) in (("in_sample", (tr_t, tr_v)), ("held_out", (te_t, te_v))):
        ref = P.score(t, v, mono)["revenue"]
        tot = float(np.sum(v))
        for mech, p in prices.items():
            sc = P.score(t, v, p)
            rows.append({**(labels or {}), "valuation": valuation, "buyers": buyers, "lp_cap": int(lp_cap),
                         "seed": int(seed), "protocol": proto, "mechanism": mech,
                         "n_buyers_fit": int(len(tr_t)), "n_lps": n_lps,
                         "revenue": sc["revenue"], "sum_values": tot,
                         "revenue_over_values": sc["revenue"] / max(tot, 1e-12),
                         "ref_monopoly": ref, "revenue_over_monopoly": sc["revenue"] / max(ref, 1e-12),
                         "served": sc["served"], "welfare_over_values": sc["welfare"] / max(tot, 1e-12),
                         "fit_seconds": sec.get(mech, np.nan), "cell_fit_seconds": total_fit})
    return rows


def add_main_norm(df):
    """Study A: revenue divided by the main protocol's reference, the per-asset monopoly
    revenue on the training buyers of the same cell (the in_sample row's ref_monopoly;
    every cell scores as many buyers as it trains on, so no size factor is needed). The
    held-out ``revenue_over_monopoly`` divides by the monopoly revenue on the evaluation
    buyers instead, which can be close to 0 with heavy-tailed (Zipf) values; the paper
    uses ``revenue_norm_main`` and the Chawla et al. ``revenue_over_values``."""
    import pandas as pd
    keys = ["workload", "support", "valuation", "family", "buyers", "lp_cap", "seed"]
    keys = [k for k in keys if k in df.columns]
    ref = (df[df.protocol == "in_sample"].groupby(keys).ref_monopoly.first().rename("ref_monopoly_train"))
    out = df.drop(columns=[c for c in ("ref_monopoly_train", "revenue_norm_main") if c in df]).merge(
        ref.reset_index(), on=keys, how="left")
    out["revenue_norm_main"] = out.revenue / out.ref_monopoly_train.where(out.ref_monopoly_train > 0)
    return out


def study_a_cells(families=("answer_cells", "support_rows", "rows_x_log_support", "flat"),
                  seeds=(0, 1, 2, 3, 4)):
    """(valuation, family, buyers, lp_cap, seed) of every cell. Our valuations are drawn
    per valuation family; Chawla valuations do not depend on the information measure, so
    they use one family ("none", the flat information only fixes the popularity draw)."""
    out = []
    for val in VALUATIONS:
        fams = families if val == "ours" else ("none",)
        for fam, b, cap, sd in itertools.product(fams, BUYERS, LP_CAPS, seeds):
            out.append((val, fam, b, cap, sd))
    return out


def edge_structure(edges, support_n: int, max_pairs: int = 20000, seed: int = 0) -> dict:
    """Structure of a hypergraph (the workload-structure factor of comment 5c)."""
    sizes = np.array([len(e) for e in edges], float)
    nz = [frozenset(e) for e in edges if len(e)]
    items = set().union(*nz) if nz else set()
    idx = list(itertools.combinations(range(len(nz)), 2))
    if len(idx) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = [idx[i] for i in rng.choice(len(idx), size=max_pairs, replace=False)]
    jac, nested, disjoint, equal = [], 0, 0, 0
    for i, j in idx:
        a, b = nz[i], nz[j]
        inter = len(a & b)
        jac.append(inter / len(a | b))
        nested += int(a <= b or b <= a)
        equal += int(a == b)
        disjoint += int(inter == 0)
    npair = max(len(idx), 1)
    return {"queries": len(edges), "support_n": int(support_n), "items_used": len(items),
            "empty_edges": int((sizes == 0).sum()), "empty_edge_share": float((sizes == 0).mean()) if len(sizes) else 0.0,
            "edge_size_mean": float(sizes.mean()) if len(sizes) else 0.0,
            "edge_size_median": float(np.median(sizes)) if len(sizes) else 0.0,
            "edge_size_max": float(sizes.max()) if len(sizes) else 0.0,
            "distinct_edges": len(set(nz)), "pairs_measured": len(idx),
            "mean_jaccard": float(np.mean(jac)) if jac else 0.0,
            "nested_share": nested / npair, "equal_share": equal / npair, "disjoint_share": disjoint / npair}


# ---------------------------------------------------------------------------------------
# Study C
# ---------------------------------------------------------------------------------------
def sweep(views: dict, te_t, te_v, ref: float, labels: dict):
    """Matched-revenue and matched-coverage rows, plus a coarse curve, for each view.

    ``views`` maps a view name (naive, oracle, operational) to the effective price vector at
    multiplier 1. Returns (matched rows, curve rows)."""
    v = np.asarray(te_v, float)
    matched, curve = [], []
    ref = max(float(ref), 1e-12)
    pos = {float(m): i for i, m in enumerate(SWEEP_GRID)}
    for view, p in views.items():
        c = np.asarray(p, float)[np.asarray(te_t)]
        M = SWEEP_GRID[:, None] * c[None, :]
        with np.errstate(invalid="ignore"):
            buys = M <= v[None, :]
        rev = np.where(buys, M, 0.0).sum(axis=1) / ref
        wel = np.where(buys, v[None, :], 0.0).sum(axis=1) / ref
        served = buys.mean(axis=1)
        base = {**labels, "view": view}
        for lvl in REVENUE_LEVELS:
            hit = np.flatnonzero(rev >= lvl - 1e-12)
            j = int(hit[0]) if len(hit) else None
            # at the grid's lower edge a smaller multiplier might also reach the level
            matched.append({**base, "match": "revenue", "level": lvl, "reached": j is not None,
                            "at_grid_edge": j == 0,
                            "multiplier": SWEEP_GRID[j] if j is not None else np.nan,
                            "revenue_norm": rev[j] if j is not None else np.nan,
                            "welfare_norm": wel[j] if j is not None else np.nan,
                            "served": served[j] if j is not None else np.nan,
                            "max_revenue_norm_on_grid": float(rev.max())})
        for lvl in COVERAGE_LEVELS:
            hit = np.flatnonzero(served >= lvl - 1e-12)
            j = int(hit[-1]) if len(hit) else None
            # at the grid's upper edge the coverage may still be above the level
            matched.append({**base, "match": "coverage", "level": lvl, "reached": j is not None,
                            "at_grid_edge": j == len(SWEEP_GRID) - 1,
                            "multiplier": SWEEP_GRID[j] if j is not None else np.nan,
                            "revenue_norm": rev[j] if j is not None else np.nan,
                            "welfare_norm": wel[j] if j is not None else np.nan,
                            "served": served[j] if j is not None else np.nan,
                            "max_revenue_norm_on_grid": float(rev.max())})
        for m in CURVE_GRID:
            j = pos[float(m)]
            curve.append({**base, "multiplier": float(m), "revenue_norm": rev[j], "welfare_norm": wel[j],
                          "served": served[j]})
    return matched, curve


# ---------------------------------------------------------------------------------------
# Study B
# ---------------------------------------------------------------------------------------
def trace_costs(trace):
    """Cumulative (seconds, queries) after each evaluated candidate, and the accepted
    count after it, from a sampler cost trace [(accepted index or -1, secs, queries)]."""
    secs = np.cumsum([t[1] for t in trace]) if trace else np.zeros(0)
    nq = np.cumsum([t[2] for t in trace]) if trace else np.zeros(0)
    acc = np.cumsum([1 if t[0] >= 0 else 0 for t in trace]) if trace else np.zeros(0, int)
    return secs, nq, acc


def cost_of_first(trace, n: int):
    """(seconds, queries, candidates) spent until the n-th neighbour was accepted."""
    secs, nq, acc = trace_costs(trace)
    hit = np.flatnonzero(acc >= n)
    if not len(hit):
        return None
    j = int(hit[0])
    return float(secs[j]), int(nq[j]), j + 1


def accepted_within(trace, budget: float, currency: str) -> int:
    """How many neighbours a sampler accepts before its spending exceeds ``budget``
    (currency "seconds" or "queries"). A candidate counts only if it fits in the budget."""
    secs, nq, acc = trace_costs(trace)
    spent = secs if currency == "seconds" else nq
    ok = np.flatnonzero(spent <= budget + 1e-12)
    return int(acc[ok[-1]]) if len(ok) else 0


def truncate_support(res, n: int):
    """Edges and answer fingerprints of the first n accepted neighbours of a SampleResult."""
    inc = {k: [i for i in v if i < n] for k, v in res.incidence.items()}
    inc = {k: v for k, v in inc.items() if v}
    fp = {k: {i: f for i, f in d.items() if i < n} for k, d in res.changed_fp.items()}
    fp = {k: d for k, d in fp.items() if d}
    return inc, fp


def support_stats(keys, incidence: dict, n: int, pairs_named) -> dict:
    """Support quality: priceable assets, price levels, and refuted lattice pairs."""
    sizes = np.array([len(incidence.get(k, ())) for k in keys], float)
    refuted = 0
    for a, b in pairs_named:
        if set(incidence.get(b, ())) - set(incidence.get(a, ())):
            refuted += 1
    return {"support_n": int(n), "assets": len(keys), "assets_priceable": int((sizes > 0).sum()),
            "zero_priced": int((sizes == 0).sum()), "distinct_price_levels": len(set(sizes.tolist())),
            "median_conflict_set": float(np.median(sizes)) if len(sizes) else 0.0,
            "lattice_pairs": len(pairs_named), "lattice_pairs_refuted": refuted}

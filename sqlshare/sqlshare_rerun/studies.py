"""Revision 3, Part 7c: the four approved studies on SQLShare.

    python -m sqlshare_rerun study --protocol revised [--which A,B,C,D]

Run after the revised protocol ("all --protocol revised", P3b). Every market is rebuilt
from the finished revised run: answers, lattice and pricing supports are read from the
caches (no pricing support of the main results is drawn again). Outputs go to
results/revised/studies/<A|B|C|D>/. Each unit of work writes its own file, so a stopped
run continues where it stopped when the same command is given again.

  A  Chawla et al. factorial (comment 5c), 15 markets; main market also on deletions.
  B  support samplers at equal size and at equal budget (comment 8c), markets 474, 4930,
     7635, 8126, 3 sampler seeds; budgets are the cost, in evaluation seconds and in query
     executions, of 1,000 accepted neighbours of each sampler.
  C  welfare at matched revenue and matched coverage (comment 8e), 15 markets, main arm,
     4 valuation families x 5 seeds; also checks that the multiplier-1 results equal the
     main results.
  D  Qirana supports whose new values are copied from a random row (frequency-weighted
     donors) instead of drawn uniformly from the distinct values (main market).

Compute-metered timings: the other markets reuse the stored single timings of the revised
run (identical to the main results). The main market's median-of-three timings were not
stored by the revised run; the first study run measures them once and stores them in
work/studies/metered_<cid>.pkl, and every study reuses that file.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import study_core as C
from .config import Settings
from .parallel import run_pool
from .pricing import info_families
from .util import LOG, load_pickle, read_json, save_pickle, write_csv, write_json

FAMILIES = ("answer_cells", "support_rows", "rows_x_log_support", "flat")
STUDY_B_MARKETS = (474, 4930, 7635, 8126)
STUDY_B_SEEDS = (0, 1, 2)
STUDY_B_N = 1000
STUDY_B_MAX_N = 20000
SAMPLERS_B = ("size_deletions", "readership_deletions", "qirana_updates")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------
# market payloads (the input of planner.run_task), rebuilt from the revised run
# ---------------------------------------------------------------------------------------
def _study_dir(s: Settings) -> Path:
    return s.results_dir / "studies"


def _main_seconds(s: Settings, S, con, assets, cid: int, keys, meta) -> dict:
    """Median-of-three timings of the main market, measured once for all studies."""
    path = s.work_dir / "studies" / f"metered_{cid}.pkl"
    if path.exists():
        return load_pickle(path)
    sub = assets[assets.k.isin(keys)]
    if s.limit_tiebreak:
        sub = sub.assign(duck=sub.k.map(lambda k: meta[k][1]))
    from . import database as D
    _, seconds = D.materialize(con, sub, s.query_timeout_s, repeats=s.metered_repeats)
    save_pickle(seconds, path)
    LOG.info("studies: main-market timings measured (median of %d) and stored in %s", s.metered_repeats, path)
    return seconds


def _payload_from_inputs(s: Settings, m: dict, cfg_extra: dict | None = None, edges=None, scores=None,
                         changed_fp=None, support_n=None) -> dict:
    keys, answers = m["keys"], m["answers"]
    contrib, nrows, op = m["contrib"], m["nrows"], m["op"]
    need = set(contrib)
    for c_ in contrib.values():
        need |= set(c_)
    if op is not None:
        for t, st in op["structures"].items():
            if st and st[0]:
                need |= {t} | set(st[0])
    price = m["supports"]["pricing"][1]
    ncols = np.array([answers[k].shape[1] for k in keys], float)
    info = info_families(m["rowsz"], m["prov"], ncols if s.answer_cells == "rows_x_cols" else np.ones(len(keys)))
    return {
        "keys": keys, "ki": {k: i for i, k in enumerate(keys)},
        "answers": {k: answers[k] for k in need}, "contrib": contrib, "nrows": nrows,
        "pairs": m["clean"], "scale_scores": m["scores"] if scores is None else scores,
        "edges": price.edges(keys) if edges is None else edges,
        "support_n": s.table6_n if support_n is None else support_n, "info": info,
        "cfg": {"demand": s.demand, "fit": s.fit, "n_buyers": s.n_buyers, "max_lps": s.max_lps,
                "cip_eps": s.cip_eps, "ilp_time_limit": s.ilp_time_limit,
                "exact_formulation": s.exact_formulation(), "log_plans": False,
                "record_costs": bool(s.record_costs), **(cfg_extra or {})},
        "op": None if (op is None or s.attackers != "both") else {k: op[k] for k in ("forms", "structures", "sem_pairs")},
    }


def _support_scores(s: Settings, S, m: dict, edges, changed_fp: dict, support_n: int) -> dict:
    """The market's scores with the support-based (Qirana) ones recomputed on another
    support, by the formulas of stages._scores; every other score is unchanged."""
    from .pricing import exact_entropy, qirana_from_edges, qirana_prices
    keys = m["keys"]
    q = qirana_from_edges(edges, support_n)
    out = dict(m["scores"])
    out["qirana_weighted_coverage"] = q["qirana_weighted_coverage"]
    sh = q["qirana_shannon"]
    if s.entropy == "exact":
        sh = np.array([exact_entropy(changed_fp.get(k, {}), support_n) for k in keys])
    out["qirana_shannon"] = np.maximum(sh, 0)
    if s.qirana_gain == "entropy_gain":
        out["qirana_uniform_entropy_gain"] = qirana_prices(edges, support_n)["qirana_uniform_entropy_gain"]
    else:
        out["qirana_uniform_gain"] = q["qirana_uniform_gain"]
    return out


class _Markets:
    """Lazy access to the rebuilt markets of the revised run."""

    def __init__(self, s: Settings):
        from . import stages as S
        from .stages import _Closure
        self.s, self.S = s, S
        self.corpus, self.sel, self.assets, self.cat = S._load_prepared(s)
        self.closure = _Closure(self.corpus)
        self.ranks = S._market_ranks(self.assets, self.sel.selected)
        self.con = S._connect(s)
        self._m = {}

    def close(self):
        self.con.close()

    def full_markets(self) -> list:
        """(rank, cid) of every market with the full protocol, in rank order."""
        out = []
        for rank, cid, _ in self.ranks:
            data = self.S._answers(self.s, self.con, self.assets, cid)
            if len(data["answers"]) < self.s.market_min_assets:
                continue
            if rank > 1 and not (self.s.full_protocol_markets or cid in set(self.s.full_markets)):
                continue
            out.append((rank, cid))
        return out

    def inputs(self, rank: int, cid: int) -> dict | None:
        if cid not in self._m:
            from .extract import _market_inputs
            s, S = self.s, self.S
            secs = None
            if rank == 1:
                data = S._answers(s, self.con, self.assets, cid)
                keys = sorted(data["answers"])
                secs = _main_seconds(s, S, self.con, self.assets, cid, keys, data["meta"]) \
                    if s.metered_repeats > 1 else None
            s.extra["sample_cache_only"] = True          # the main results' supports, never redrawn
            try:
                m = _market_inputs(s, S, self.con, self.cat, self.closure, self.assets, rank, cid,
                                   main=(rank == 1), seconds=secs)
            finally:
                s.extra.pop("sample_cache_only", None)
            if m is not None:
                m["rank"], m["cid"] = rank, cid
            self._m.clear()                               # keep one market in memory
            self._m[cid] = m
        return self._m[cid]


# ---------------------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------------------
def _task_a(args):
    path, cell = args
    d = _payload_cached(path)
    val, fam, buyers, cap, seed = cell
    info = d["info"]["flat" if fam == "none" else fam]
    labels = {"workload": d["workload"], "support": d["support"], "family": fam}
    return C.chawla_cell(d["edges"], d["support_n"], info, seed, val, buyers, cap, d["cip_eps"],
                         d["n_buyers"], labels)


_PAYLOAD = {}


def _payload_cached(path):
    if path not in _PAYLOAD:
        _PAYLOAD.clear()
        _PAYLOAD[path] = load_pickle(Path(path))
    return _PAYLOAD[path]


def run_a_workload(work_dir: Path, out: Path, name: str, support: str, edges, support_n: int, info: dict,
                   workers: int, seeds, cip_eps: float, n_buyers: int, stall_s: float = 1800.0):
    """Every study-A cell of one workload and support (also used for TPC-H). Resumable per
    workload and support: the result file is written last."""
    final = out / "tasks" / f"{name}__{support}.csv"
    if final.exists():
        return
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"A_{name}_{support}.pkl"
    save_pickle({"workload": name, "support": support, "edges": edges, "support_n": support_n, "info": info,
                 "cip_eps": cip_eps, "n_buyers": n_buyers}, path)
    cells = C.study_a_cells(FAMILIES, seeds)
    t0 = time.perf_counter()
    res = run_pool(_task_a, [(str(path), c) for c in cells], workers, f"study A {name} ({support})", stall_s)
    _PAYLOAD.clear()
    _write_csv_atomic(pd.DataFrame([r for rows in res for r in rows]), final)
    st = C.edge_structure(edges, support_n)
    _write_csv_atomic(pd.DataFrame([{"workload": name, "support": support, **st,
                                     "seconds": round(time.perf_counter() - t0, 1)}]),
                      out / "tasks" / f"{name}__{support}.structure.csv")
    path.unlink(missing_ok=True)
    LOG.info("study A: %s (%s) done in %.0f s", name, support, time.perf_counter() - t0)


def _run_a_workload(s: Settings, out: Path, name: str, support: str, edges, support_n: int, info: dict,
                    workers: int):
    run_a_workload(s.work_dir / "studies", out, name, support, edges, support_n, info, workers, s.run_seeds,
                   s.cip_eps, s.n_buyers, s.pool_stall_s)


def study_a(s: Settings, mk: _Markets, out: Path):
    for rank, cid in mk.full_markets():
        name = f"sqlshare_{cid}"
        need_upd = not (out / "tasks" / f"{name}__updates.csv").exists()
        need_del = rank == 1 and not (out / "tasks" / f"{name}__deletions.csv").exists()
        if not (need_upd or need_del):
            continue
        m = mk.inputs(rank, cid)
        if m is None:
            continue
        p = _payload_from_inputs(s, m)
        if need_upd:
            _run_a_workload(s, out, name, "updates", p["edges"], p["support_n"], p["info"], s.n_workers())
        if need_del:
            dres = m["supports"]["readership"][1]
            _run_a_workload(s, out, name, "deletions", dres.edges(m["keys"]), dres.n, p["info"], s.n_workers())
    collect_a(out)


def collect_a(out: Path):
    files = sorted((out / "tasks").glob("*__*.csv"))
    runs = [pd.read_csv(f) for f in files if not f.name.endswith(".structure.csv")]
    if not runs:
        return
    df = pd.concat(runs, ignore_index=True)
    write_csv(df, out / "study_A_runs.csv")
    structs = [pd.read_csv(f) for f in files if f.name.endswith(".structure.csv")]
    if structs:
        write_csv(pd.concat(structs, ignore_index=True), out / "study_A_workloads.csv")
    summarize_a(df).to_csv(out / "study_A_summary.csv", index=False)


def summarize_a(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds (and families for our valuations) per cell and mechanism, with the
    rank of each mechanism in its cell (1 = most revenue)."""
    df = C.add_main_norm(df)
    keys = ["workload", "support", "valuation", "buyers", "lp_cap", "protocol", "mechanism"]
    g = df.groupby(keys)[["revenue_over_values", "revenue_norm_main", "served"]].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    cell = ["workload", "support", "valuation", "buyers", "lp_cap", "protocol"]
    g["rank_in_cell"] = g.groupby(cell)["revenue_norm_main_mean"].rank(ascending=False, method="min")
    return g


# ---------------------------------------------------------------------------------------
# Study C
# ---------------------------------------------------------------------------------------
def _task_c(args):
    from .planner import run_task
    return run_task(args)


def study_c(s: Settings, mk: _Markets, out: Path):
    from .planner import _MARKET, _VERIFIED
    tdir = out / "tasks"
    for rank, cid in mk.full_markets():
        final = tdir / f"sqlshare_{cid}.matched.csv"
        if final.exists():
            continue
        m = mk.inputs(rank, cid)
        if m is None:
            continue
        p = _payload_from_inputs(s, m, {"sweep": True})
        path = s.work_dir / "studies" / f"C_{cid}.pkl"
        save_pickle(p, path)
        tasks = [(str(path), s.main_arm, f, sd, s.n_buyers, True) for f in FAMILIES for sd in s.run_seeds]
        t0 = time.perf_counter()
        res = run_pool(_task_c, tasks, s.n_workers(), f"study C market {cid}", s.pool_stall_s)
        _MARKET.clear()
        _VERIFIED.clear()
        runs = pd.DataFrame([r for rows in res for r in rows if "_sweep" not in r and "_plans" not in r])
        matched = pd.DataFrame([x for rows in res for r in rows if "_sweep" in r for x in r["_sweep"]])
        curve = pd.DataFrame([x for rows in res for r in rows if "_curve" in r for x in r["_curve"]])
        for df in (runs, matched, curve):
            df.insert(0, "workload", f"sqlshare_{cid}")
        _write_csv_atomic(runs, tdir / f"sqlshare_{cid}.runs.csv")
        _write_csv_atomic(curve, tdir / f"sqlshare_{cid}.curve.csv")
        _write_csv_atomic(matched, final)                 # written last: marks the market done
        path.unlink(missing_ok=True)
        LOG.info("study C: market %d done in %.0f s", cid, time.perf_counter() - t0)
    _collect_c(s, out)


def _collect_c(s: Settings, out: Path):
    tdir = out / "tasks"
    parts = {k: sorted(tdir.glob(f"*.{k}.csv")) for k in ("runs", "matched", "curve")}
    if not parts["matched"]:
        return
    runs = pd.concat([pd.read_csv(f) for f in parts["runs"]], ignore_index=True)
    matched = pd.concat([pd.read_csv(f) for f in parts["matched"]], ignore_index=True)
    curve = pd.concat([pd.read_csv(f) for f in parts["curve"]], ignore_index=True)
    write_csv(runs, out / "study_C_runs.csv")
    write_csv(matched, out / "study_C_matched.csv")
    write_csv(curve, out / "study_C_curves.csv")
    matched["matched"] = matched.reached.astype(bool) & ~matched.at_grid_edge.astype(bool)
    ok = matched[matched.matched]
    g = matched.groupby(["workload", "view", "match", "level", "mechanism"]).agg(
        runs=("matched", "size"), matched_share=("matched", "mean")).reset_index()
    v = ok.groupby(["workload", "view", "match", "level", "mechanism"]).agg(
        welfare_norm=("welfare_norm", "mean"), revenue_norm=("revenue_norm", "mean"),
        served=("served", "mean"), multiplier=("multiplier", "mean")).reset_index()
    g = g.merge(v, how="left", on=["workload", "view", "match", "level", "mechanism"])
    write_csv(g, out / "study_C_summary.csv")
    write_csv(consistency_c(s, runs), out / "study_C_consistency.csv")


def consistency_c(s: Settings, runs: pd.DataFrame) -> pd.DataFrame:
    """Study C refits every main-arm run; its multiplier-1 numbers must equal the main
    results (compute_metered on the main market may differ: its timings are re-measured)."""
    ref = []
    mm = s.results_dir / "main_market" / "runs.csv"
    fp = s.results_dir / "markets" / "full_protocol_runs.csv"
    if fp.exists():
        r = pd.read_csv(fp)
        r = r[r.arm == s.main_arm]
        r["workload"] = "sqlshare_" + r.cid.astype(str)
        ref.append(r)
    elif mm.exists():
        r = pd.read_csv(mm)
        ref.append(r[r.arm == s.main_arm].assign(workload="main"))
    if not ref:
        return pd.DataFrame()
    ref = pd.concat(ref, ignore_index=True)
    cols = [c for c in ("revenue_naive_norm", "revenue_strategic_norm", "revenue_strategic_op_norm",
                        "welfare_strategic_op_norm", "comb_violation_rate_op") if c in runs and c in ref]
    j = runs.merge(ref, on=["workload", "family", "seed", "mechanism"], suffixes=("", "_main"))
    rows = []
    for (w, mech), g in j.groupby(["workload", "mechanism"]):
        row = {"workload": w, "mechanism": mech, "runs": len(g)}
        for c in cols:
            row[f"max_abs_diff_{c}"] = float(np.nanmax(np.abs(g[c].astype(float) - g[c + "_main"].astype(float))))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Study B
# ---------------------------------------------------------------------------------------
def _b_samplers(s: Settings, S, mk: _Markets, m: dict):
    from .support import SupportSampler
    sup, con, cat = m["sup"], mk.con, mk.cat
    common = dict(isolate_errors=s.isolate_errors, error_as_change=s.error_as_change, trace_costs=True)
    dele = SupportSampler(con, cat["schemas"], m["asset_sql"], sup["readers"], m["base_fp"], sup["targets"],
                          sup["counts"], s.row_sampling, s.n_threads(), s.pool_stall_s, **common)
    qir = SupportSampler(con, cat["schemas"], m["asset_sql"], sup["readers"], m["base_fp"], sup["targets"],
                         sup["counts"], s.row_sampling, s.n_threads(), s.pool_stall_s, neighbours="qirana",
                         swap_ratio=s.swap_ratio, donor_mode=s.qirana_donors, **common)
    return {"size_deletions": (dele, sup["size_w"]), "readership_deletions": (dele, sup["read_w"]),
            "qirana_updates": (qir, S._qirana_weights(s, sup))}


def _b_draw(s: Settings, S, m, samplers, name: str, sd: int, n: int):
    smp, w = samplers[name]
    j = SAMPLERS_B.index(name)
    tseed = 7000 + 100 * sd + j
    return S._sample(s, smp, m["prep_key"], f"studyB_{m['cid']}_{name}_s{sd}_n{n}", m["sup"]["readers"],
                     m["sup"]["tables"], w, n, tseed, S._row_seed(tseed), record_fp=(s.entropy == "exact"),
                     label=f"study B {m['cid']} {name} seed {sd} ({n})")


def study_b(s: Settings, mk: _Markets, out: Path):
    from .planner import _MARKET, _VERIFIED
    S = mk.S
    tdir = out / "tasks"
    avail = {cid: rank for rank, cid in mk.full_markets()}
    # tests on the synthetic release set other markets and smaller sizes through s.extra
    markets = tuple(s.extra.get("study_b_markets", STUDY_B_MARKETS))
    seeds = tuple(s.extra.get("study_b_seeds", STUDY_B_SEEDS))
    n_b = int(s.extra.get("study_b_n", STUDY_B_N))
    for cid in markets:
        if cid not in avail:
            LOG.warning("study B: market %d has no full protocol in this run; skipped", cid)
            continue
        for sd in seeds:
            final = tdir / f"sqlshare_{cid}_s{sd}.runs.csv"
            if final.exists():
                continue
            m = mk.inputs(avail[cid], cid)
            samplers = _b_samplers(s, S, mk, m)
            t0 = time.perf_counter()
            pilot = {nm: _b_draw(s, S, m, samplers, nm, sd, n_b) for nm in SAMPLERS_B}
            budget = {}
            for nm, res in pilot.items():
                c = C.cost_of_first(res.cost_trace, n_b)
                budget[nm] = {"seconds": c[0], "queries": c[1], "candidates": c[2]} if c else None
            # each sampler at every other sampler's budget: draw enough for the largest one
            big = {}
            for nm in SAMPLERS_B:
                own = budget[nm]
                if own is None:
                    big[nm] = pilot[nm]
                    continue
                need = 1.0
                for other in SAMPLERS_B:
                    if other == nm or budget[other] is None:
                        continue
                    for cur in ("seconds", "queries"):
                        need = max(need, budget[other][cur] / max(own[cur], 1e-9))
                n_big = int(min(STUDY_B_MAX_N, math.ceil(n_b * need * 1.15)))
                big[nm] = pilot[nm] if n_big <= n_b else _b_draw(s, S, m, samplers, nm, sd, n_big)
            pairs_named = list(m["fam"].keys())
            supports, srows = [], []
            for nm in SAMPLERS_B:
                cond = [("equal_size", "-", min(n_b, pilot[nm].n), pilot[nm])]
                for other in SAMPLERS_B:
                    if other == nm or budget[other] is None:
                        continue
                    for cur in ("seconds", "queries"):
                        n_at = C.accepted_within(big[nm].cost_trace, budget[other][cur], cur)
                        cond.append((f"budget_of_{other}", cur, n_at, big[nm]))
                for condition, cur, n_at, res in cond:
                    inc, fp = C.truncate_support(res, n_at)
                    st = C.support_stats(m["keys"], inc, n_at, pairs_named)
                    c = C.cost_of_first(res.cost_trace, n_at) if n_at else (0.0, 0, 0)
                    srow = {"workload": f"sqlshare_{cid}", "cid": cid, "sampler_seed": sd, "sampler": nm,
                            "condition": condition, "currency": cur,
                            "budget_seconds": budget[condition[len("budget_of_"):]]["seconds"]
                            if condition.startswith("budget_of_") else (budget[nm] or {}).get("seconds"),
                            "budget_queries": budget[condition[len("budget_of_"):]]["queries"]
                            if condition.startswith("budget_of_") else (budget[nm] or {}).get("queries"),
                            "spent_seconds": c[0] if c else np.nan, "spent_queries": c[1] if c else np.nan,
                            "candidates": c[2] if c else np.nan,
                            "draw_capped": bool(res.n >= STUDY_B_MAX_N and n_at >= res.n), **st}
                    srows.append(srow)
                    supports.append((srow, inc, fp, n_at))
            # downstream: refit every mechanism on each support (main arm, 4 families)
            tasks, payloads = [], []
            for i, (srow, inc, fp, n_at) in enumerate(supports):
                edges = [frozenset(inc.get(k, ())) for k in m["keys"]]
                scores = _support_scores(s, S, m, edges, fp, max(n_at, 1))
                p = _payload_from_inputs(s, m, {"compact_lp": True}, edges=edges, scores=scores,
                                         support_n=max(n_at, 1))
                path = s.work_dir / "studies" / f"B_{cid}_s{sd}_{i}.pkl"
                save_pickle(p, path)
                payloads.append(path)
                tasks += [(str(path), s.main_arm, f, sd, s.n_buyers, True) for f in FAMILIES]
            res = run_pool(_task_c, tasks, s.n_workers(), f"study B market {cid} seed {sd}", s.pool_stall_s)
            _MARKET.clear()
            _VERIFIED.clear()
            rows = []
            for t, rr in zip(tasks, res):
                i = payloads.index(Path(t[0]))
                srow = supports[i][0]
                for r in rr:
                    if "_plans" in r or "_sweep" in r:
                        continue
                    rows.append({**{k: srow[k] for k in ("workload", "cid", "sampler_seed", "sampler",
                                                          "condition", "currency", "support_n")}, **r})
            for pth in payloads:
                pth.unlink(missing_ok=True)
            _write_csv_atomic(pd.DataFrame(srows), tdir / f"sqlshare_{cid}_s{sd}.supports.csv")
            _write_csv_atomic(pd.DataFrame(rows), final)
            LOG.info("study B: market %d seed %d done in %.0f s", cid, sd, time.perf_counter() - t0)
    _collect_b(out)


def _collect_b(out: Path):
    tdir = out / "tasks"
    sp = sorted(tdir.glob("*.supports.csv"))
    rp = sorted(tdir.glob("*.runs.csv"))
    if not sp:
        return
    sup = pd.concat([pd.read_csv(f) for f in sp], ignore_index=True)
    runs = pd.concat([pd.read_csv(f) for f in rp], ignore_index=True)
    write_csv(sup, out / "study_B_supports.csv")
    write_csv(runs, out / "study_B_runs.csv")
    keys = ["workload", "sampler", "condition", "currency", "mechanism"]
    cols = [c for c in ("support_n", "revenue_naive_norm", "revenue_strategic_op_norm", "welfare_strategic_op_norm",
                        "comb_violation_rate_op", "mono_violation_rate") if c in runs]
    write_csv(runs.groupby(keys)[cols].mean().reset_index(), out / "study_B_summary.csv")


# ---------------------------------------------------------------------------------------
# Study D
# ---------------------------------------------------------------------------------------
def study_d(s: Settings, mk: _Markets, out: Path):
    from .planner import _MARKET, _VERIFIED
    S = mk.S
    tdir = out / "tasks"
    final = tdir / "sqlshare_main.runs.csv"
    if not final.exists():
        rank, cid = mk.full_markets()[0]
        m = mk.inputs(rank, cid)
        from .support import SupportSampler
        sup = m["sup"]
        qf = SupportSampler(mk.con, mk.cat["schemas"], m["asset_sql"], sup["readers"], m["base_fp"], sup["targets"],
                            sup["counts"], s.row_sampling, s.n_threads(), s.pool_stall_s, neighbours="qirana",
                            swap_ratio=s.swap_ratio, donor_mode="frequency", isolate_errors=s.isolate_errors,
                            error_as_change=s.error_as_change)
        t0 = time.perf_counter()
        # same tables, weights and seeds as the main pricing support; only the donors differ
        fres = S._sample(s, qf, m["prep_key"], f"qirana_{s.qirana_table_weights}", sup["readers"], sup["tables"],
                         S._qirana_weights(s, sup), s.table6_n, s.table6_seed, S._row_seed(s.table6_seed + 2),
                         record_fp=(s.entropy == "exact"), label="study D frequency donors")
        base = m["supports"]["pricing"][1]
        pairs_named = list(m["fam"].keys())
        srows, tasks, paths = [], [], []
        for donors, res in (("distinct", base), ("frequency", fres)):
            edges = res.edges(m["keys"])
            inc = {k: v for k, v in res.incidence.items()}
            st = C.support_stats(m["keys"], inc, res.n, pairs_named)
            srows.append({"workload": f"sqlshare_{cid}", "donors": donors, "draws": res.draws,
                          "rejected": res.skipped, "build_seconds": res.seconds, **st})
            scores = m["scores"] if donors == "distinct" else _support_scores(s, S, m, edges, res.changed_fp,
                                                                                s.table6_n)
            p = _payload_from_inputs(s, m, edges=edges, scores=scores)
            path = s.work_dir / "studies" / f"D_{cid}_{donors}.pkl"
            save_pickle(p, path)
            paths.append((donors, path))
            tasks += [(str(path), s.main_arm, f, sd, s.n_buyers, True) for f in FAMILIES for sd in s.run_seeds]
        res = run_pool(_task_c, tasks, s.n_workers(), "study D", s.pool_stall_s)
        _MARKET.clear()
        _VERIFIED.clear()
        rows = []
        for t, rr in zip(tasks, res):
            donors = next(dn for dn, pth in paths if str(pth) == t[0])
            rows += [{"workload": f"sqlshare_{cid}", "donors": donors, **r} for r in rr
                     if "_plans" not in r and "_sweep" not in r]
        for _, pth in paths:
            pth.unlink(missing_ok=True)
        _write_csv_atomic(pd.DataFrame(srows), tdir / "sqlshare_main.supports.csv")
        _write_csv_atomic(pd.DataFrame(rows), final)
        LOG.info("study D: done in %.0f s", time.perf_counter() - t0)
    sup = pd.read_csv(tdir / "sqlshare_main.supports.csv")
    runs = pd.read_csv(final)
    write_csv(sup, out / "study_D_supports.csv")
    write_csv(runs, out / "study_D_runs.csv")
    cols = [c for c in ("revenue_naive_norm", "revenue_strategic_op_norm", "welfare_strategic_op_norm",
                        "comb_violation_rate_op", "mono_violation_rate", "unsellable_assets") if c in runs]
    write_csv(runs.groupby(["workload", "donors", "mechanism"])[cols].mean().reset_index(), out / "study_D_summary.csv")


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def stage_studies(s: Settings, which=("A", "B", "C", "D")) -> dict:
    if s.protocol != "revised":
        raise ValueError("the studies run on the revised protocol only (--protocol revised)")
    for f in ("main_market/main.done", "markets/markets.done"):
        if not (s.results_dir / f).exists():
            raise RuntimeError(f"the revised run is not finished ({f} missing); run 'all --protocol revised' first")
    (s.work_dir / "studies").mkdir(parents=True, exist_ok=True)
    root = _study_dir(s)
    t0 = time.perf_counter()
    mk = _Markets(s)
    times = {}
    try:
        for w in which:
            t1 = time.perf_counter()
            out = root / w
            (out / "tasks").mkdir(parents=True, exist_ok=True)
            {"A": study_a, "B": study_b, "C": study_c, "D": study_d}[w](s, mk, out)
            times[w] = round(time.perf_counter() - t1, 1)
            LOG.info("study %s finished (%.0f s)", w, times[w])
    finally:
        mk.close()
    summ = {"which": list(which), "seconds": times, "fingerprint": s.fingerprint(),
            "total_seconds": round(time.perf_counter() - t0, 1)}
    prev = root / "studies_summary.json"
    if prev.exists():
        old = read_json(prev)
        old.get("seconds", {}).update(times)
        summ["seconds"] = old.get("seconds", times)
    write_json(summ, prev)
    return summ

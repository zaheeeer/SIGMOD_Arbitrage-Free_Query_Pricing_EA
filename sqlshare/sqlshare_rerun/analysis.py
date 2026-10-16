"""Revision 3, phase P6: every table and number of the revised paper, computed from the
finished result files (approved by Jack on 2026-10-15).

    python -m sqlshare_rerun analyze --protocol revised --root <sqlshare_root>
        [--tpch <TPC-H out folder with revised, revised_delete>]
        [--tpch-studies <TPC-H studies folder>]

Reads only finished results (no support set is drawn, no price is fitted) and writes
results/revised/analysis/:

  funnel.csv, lattice_families.csv     comment 7c/7d: 975 -> 806 -> 812 with exclusions
  main_results.csv                     per workload and mechanism, main arm (mean, sd)
  comparisons.csv                      comment 8a/8b: paired bootstrap 95% intervals
                                       (10,000 resamples) and, for each comparison, in how
                                       many settings (workload x family x seed) it holds
  per_family.csv, arms_sizes.csv       comment 6
  costs.csv                            comment 8d: support build, price fitting, planning
  causes.csv, causes_plans.csv         comment 4b: plans with gain 1.00 by cause
  relations_summary.csv                comment 3d: the three relations per market
  study_A_contrasts.csv, study_B_table.csv, study_C_comparisons.csv, study_D_comparisons.csv
  paper_numbers.json                   headline numbers quoted in the text

Workloads: sqlshare_main (cid 474), tpch, sqlshare_15 (the 15 markets with the full
protocol; units are markets) and sqlshare_14 (the same without cid 2610, whose pricing
support changes no answer).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .util import LOG, read_json, write_csv, write_json

N_BOOT = 10_000
BOOT_SEED = 20261015
LEVEL = 0.95
SINGLE_METRICS = ("revenue_strategic_op_norm", "welfare_strategic_op_norm", "comb_violation_rate_op")
ORACLE_PAIRS = (("revenue_strategic_norm", "revenue_strategic_op_norm"),
                ("comb_violation_rate", "comb_violation_rate_op"))
BASELINE = "uniform"
EXCLUDE_2610 = 2610
# constructions published as arbitrage-free (bundle arbitrage); uniform entropy gain is only
# information-arbitrage-free (D2), the rest are baselines without a guarantee
ARBITRAGE_FREE = ("qirana_weighted_coverage", "qirana_shannon", "querymarket_viewcover",
                  "querymarket_viewcover_instance", "chawla_ubp", "chawla_uip", "chawla_lpip",
                  "chawla_cip", "chawla_layering", "chawla_xos")
INFO_ONLY = ("qirana_uniform_entropy_gain",)


# ---------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------
def boot_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = LEVEL):
    """Mean and percentile interval of the mean, resampling the units with replacement."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(0, n_boot, 1000):
        k = min(1000, n_boot - i)
        means[i:i + k] = x[rng.integers(0, len(x), size=(k, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def _signs(d, tol=1e-12):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return int((d > tol).sum()), int((d < -tol).sum()), int((np.abs(d) <= tol).sum())


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_runs(s: Settings, tpch: Path | None) -> dict:
    """Main-arm runs (main training size) per workload."""
    out = {}
    res = s.results_dir
    mm = _read(res / "main_market" / "runs.csv")
    if mm is not None:
        out["sqlshare_main"] = mm[mm.arm == s.main_arm].copy()
    fp = _read(res / "markets" / "full_protocol_runs.csv")
    if fp is not None:
        fp = fp[fp.arm == s.main_arm].copy()
        fp["unit_market"] = fp.cid
        out["sqlshare_15"] = fp
        out["sqlshare_14"] = fp[fp.cid != EXCLUDE_2610].copy()
    if tpch is not None and (tpch / "revised" / "runs.csv").exists():
        t = pd.read_csv(tpch / "revised" / "runs.csv")
        out["tpch"] = t[(t.arm == "same") & (t.n_train == t.n_test)].copy()
    return out


def _unit_frame(df: pd.DataFrame, metric: str, workload: str) -> pd.DataFrame:
    """mechanism x unit table of a metric. Units: (family, seed) on one market; markets
    (mean over family and seed) across markets."""
    if workload in ("sqlshare_15", "sqlshare_14"):
        return df.groupby(["mechanism", "cid"])[metric].mean().unstack("cid")
    return df.set_index(["mechanism", "family", "seed"])[metric].unstack(["family", "seed"])


def _setting_frame(df: pd.DataFrame, metric: str, workload: str) -> pd.DataFrame:
    """mechanism x setting (market, family, seed) for the "every setting" counts."""
    keys = ["family", "seed"] + (["cid"] if "cid" in df and workload.startswith("sqlshare_1") else [])
    return df.set_index(["mechanism"] + keys)[metric].unstack(keys)


# ---------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------
def main_results(runs: dict) -> pd.DataFrame:
    rows = []
    cols = ("revenue_naive_norm", "revenue_strategic_op_norm", "welfare_strategic_op_norm", "served_strategic_op",
            "comb_violation_rate_op", "mono_violation_rate", "revenue_strategic_norm", "comb_violation_rate",
            "generalization_gap", "unsellable_assets")
    for w, df in runs.items():
        for mech, g in df.groupby("mechanism"):
            row = {"workload": w, "mechanism": mech, "runs": len(g)}
            for c in cols:
                if c in g:
                    if w in ("sqlshare_15", "sqlshare_14"):
                        per = g.groupby("cid")[c].mean()
                        row[c + "_mean"] = float(per.mean())
                        row[c + "_sd_across_markets"] = float(per.std(ddof=1)) if len(per) > 1 else np.nan
                    else:
                        row[c + "_mean"] = float(g[c].mean())
                        row[c + "_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons(runs: dict) -> pd.DataFrame:
    rows = []

    def add(w, kind, a, b, metric, ua, ub, sa, sb, metric_b=None):
        d = (ua - ub).to_numpy(float)
        mean, lo, hi = boot_ci(d)
        ds = (sa - sb).to_numpy(float)
        pos, neg, tie = _signs(ds)
        rows.append({"workload": w, "comparison": kind, "a": a, "b": b, "metric": metric,
                     **({"metric_b": metric_b} if metric_b else {}),
                     "units": int(np.isfinite(d).sum()), "mean_diff": mean, "ci_low": lo, "ci_high": hi,
                     "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                     "settings": pos + neg + tie, "settings_a_higher": pos, "settings_b_higher": neg,
                     "settings_equal": tie})

    for w, df in runs.items():
        mechs = sorted(df.mechanism.unique())
        for metric in SINGLE_METRICS:
            if metric not in df:
                continue
            U = _unit_frame(df, metric, w)
            Sx = _setting_frame(df, metric, w)
            if BASELINE in U.index:
                for m in mechs:
                    if m != BASELINE:
                        add(w, "vs_uniform", m, BASELINE, metric, U.loc[m], U.loc[BASELINE], Sx.loc[m], Sx.loc[BASELINE])
            if {"chawla_lpip", "chawla_ubp"} <= set(U.index):
                add(w, "lpip_vs_ubp", "chawla_lpip", "chawla_ubp", metric, U.loc["chawla_lpip"], U.loc["chawla_ubp"],
                    Sx.loc["chawla_lpip"], Sx.loc["chawla_ubp"])
        for mo, mp in ORACLE_PAIRS:
            if mo not in df or mp not in df:
                continue
            Uo, Up = _unit_frame(df, mo, w), _unit_frame(df, mp, w)
            So, Sp = _setting_frame(df, mo, w), _setting_frame(df, mp, w)
            for m in mechs:
                add(w, "oracle_vs_operational", m, m, mo, Uo.loc[m], Up.loc[m], So.loc[m], Sp.loc[m], metric_b=mp)
    return pd.DataFrame(rows)


def per_family(runs: dict) -> pd.DataFrame:
    cols = [c for c in ("revenue_strategic_op_norm", "welfare_strategic_op_norm", "served_strategic_op",
                        "comb_violation_rate_op", "revenue_naive_norm") ]
    out = []
    for w, df in runs.items():
        c = [x for x in cols if x in df]
        if w in ("sqlshare_15", "sqlshare_14"):
            g = df.groupby(["family", "mechanism", "cid"])[c].mean().groupby(["family", "mechanism"]).mean()
        else:
            g = df.groupby(["family", "mechanism"])[c].mean()
        out.append(g.reset_index().assign(workload=w))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def arms_sizes(s: Settings, tpch: Path | None) -> pd.DataFrame:
    """Comment 6: test arms and training sizes. ``revenue_vs_test_optimum`` divides the test
    revenue by the per-asset optimum on the test buyers (ref_monopoly_test), a reference
    that does not depend on the training sample."""
    frames = []
    mm = _read(s.results_dir / "main_market" / "runs.csv")
    if mm is not None:
        frames.append(mm.assign(workload="sqlshare_main"))
    if tpch is not None and (tpch / "revised" / "runs.csv").exists():
        frames.append(pd.read_csv(tpch / "revised" / "runs.csv").assign(workload="tpch"))
    rows = []
    for df in frames:
        df = df.copy()
        df["demand_arm"] = df.arm.str.split("_n").str[0]
        if "ref_monopoly_test" in df:
            df["revenue_vs_test_optimum"] = df.revenue_naive / df.ref_monopoly_test.replace(0, np.nan)
        cols = [c for c in ("revenue_naive_norm", "revenue_vs_test_optimum", "generalization_gap", "kendall_tau",
                            "demand_tv", "revenue_train_norm") if c in df]
        g = df.groupby(["workload", "demand_arm", "n_train", "mechanism"])[cols].agg(["mean", "std", "count"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def funnel(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    f = _read(s.work_dir / "funnel.csv")
    if f is not None:
        for r in f.itertuples():
            rows.append({"step": r.stage, "count": int(r.n), "note": "release funnel (prepare)"})
    lat = s.results_dir / "lattice"
    comp = _read(lat / "components.csv")
    summ = read_json(lat / "summary.json") if (lat / "summary.json").exists() else {}
    if comp is not None and summ:
        st = comp[comp.structural]
        surv = int(summ.get("survivors_total", 0))
        rows += [
            {"step": "survivors (non-empty answers)", "count": surv, "note": ""},
            {"step": "in the structural markets (lattice)", "count": int(st.survivors.sum()),
             "note": f"{surv - int(st.survivors.sum())} survivors are in markets below the lattice size threshold"},
            {"step": "materialized assets in structural markets", "count": int(st.materialized.sum()),
             "note": f"{int(st.survivors.sum() - st.materialized.sum())} excluded in those markets "
                     f"(row cap over all survivors: {summ.get('row_cap_exclusions')})"},
        ]
    struct = _read(s.results_dir / "markets" / "structure.csv")
    mr = _read(s.results_dir / "markets" / "runs.csv")
    if struct is not None:
        small = mr[mr.status == "too_small"] if mr is not None and "status" in mr else pd.DataFrame()
        rows.append({"step": "assets in markets with the full protocol", "count": int(struct.assets.sum()),
                     "note": f"{len(struct)} markets with at least {s.market_min_assets} assets; "
                             f"{len(small)} smaller markets left out "
                             f"(cids {', '.join(map(str, small.cid.tolist())) if len(small) else '-'})"})
    fam = pd.DataFrame()
    if summ:
        fc = summ.get("family_counts", {})
        tot = max(sum(fc.values()), 1)
        fam = pd.DataFrame([{"family": k, "verified_pairs": v, "share_pct": round(100 * v / tot, 2)} for k, v in fc.items()])
        fam = pd.concat([fam, pd.DataFrame([{"family": "projection+selection",
                                              "verified_pairs": fc.get("projection", 0) + fc.get("selection", 0),
                                              "share_pct": round(100 * (fc.get("projection", 0) + fc.get("selection", 0)) / tot, 2)}])],
                        ignore_index=True)
    return pd.DataFrame(rows), fam


def costs(s: Settings, runs: dict, tpch: Path | None, support_seconds: list) -> pd.DataFrame:
    rows = list(support_seconds)
    for w, df in runs.items():
        if w == "sqlshare_14":
            continue
        for mech, g in df.groupby("mechanism"):
            row = {"workload": w, "item": "price fitting", "mechanism": mech,
                   "seconds_mean": float(g.fit_seconds.mean()) if "fit_seconds" in g else np.nan}
            rows.append(row)
            for c, nm in (("plan_seconds_oracle", "planning (oracle)"), ("plan_seconds_operational", "planning (operational)")):
                if c in g:
                    rows.append({"workload": w, "item": nm, "mechanism": mech, "seconds_mean": float(g[c].mean())})
    if tpch is not None:
        for sub, kind in (("revised", "qirana_updates"), ("revised_delete", "deletions")):
            p = tpch / sub / "build.json"
            if p.exists():
                b = json.loads(p.read_text(encoding="utf-8"))
                for k, v in b.get("costs", {}).items():
                    rows.append({"workload": "tpch", "item": f"{k.replace('_seconds', '')} ({kind})", "mechanism": "-",
                                 "seconds_mean": float(v)})
    return pd.DataFrame(rows)


def support_seconds(s: Settings) -> list:
    """Support build cost of every market, from the cached support sets of the revised run."""
    from .studies import _Markets
    rows = []
    mk = _Markets(s)
    try:
        for rank, cid in mk.full_markets():
            m = mk.inputs(rank, cid)
            if m is None:
                continue
            for name, (_, res) in m["supports"].items():
                rows.append({"workload": f"sqlshare_{cid}", "item": f"support build ({name})", "mechanism": "-",
                             "seconds_mean": float(res.seconds), "draws": int(res.draws), "accepted": int(res.n)})
    finally:
        mk.close()
    return rows


def causes(s: Settings, tpch: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comment 4b: every verified plan with gain 1.00 (a free rebuild of a priced target),
    by cause: (1) target not for sale (infinite price), (2) mechanism without a bundle
    guarantee (uniform entropy gain, or a baseline), (3) instance-specific derivation: the
    target's conflict set is not inside the sources' union on the pricing support, so the
    sources do not determine the target, (4) otherwise (would indicate a defect)."""
    items = []
    mm = s.results_dir / "main_market"
    if (mm / "plans_log.csv").exists() and (mm / "pricing_support.json").exists():
        sup = read_json(mm / "pricing_support.json")
        inc = {str(k): set(v) for k, v in sup["incidence"].items()}
        items.append(("sqlshare_main", pd.read_csv(mm / "plans_log.csv"), inc))
    if tpch is not None and (tpch / "revised" / "plans_log.csv").exists() and (tpch / "revised" / "market.pkl").exists():
        import pickle
        import sys
        # the TPC-H market file holds catalog objects of tpch/modules (probe_tpch_lattice.Asset)
        mods = str(Path(__file__).resolve().parents[2] / "tpch" / "modules")
        if mods not in sys.path:
            sys.path.insert(0, mods)
        with open(tpch / "revised" / "market.pkl", "rb") as fh:
            mk = pickle.load(fh)
        inc = {str(i): set(e) for i, e in enumerate(mk["edges"])}
        items.append(("tpch", pd.read_csv(tpch / "revised" / "plans_log.csv"), inc))
    prow = []
    for w, plans, inc in items:
        p = plans[(plans.verified.astype(bool)) & (plans.gain >= 0.999)]
        if "arm" in p:
            p = p[p.arm == "same"]
        for (mech, att, t, srcs), g in p.groupby(["mechanism", "attacker", "target", "sources"]):
            src = [x for x in str(srcs).split(";") if x]
            covered = inc.get(str(t), set()) <= set().union(*[inc.get(x, set()) for x in src]) if src else False
            if not np.isfinite(g.price.astype(float)).all():
                cause = "target not for sale (infinite price)"
            elif mech in INFO_ONLY:
                cause = "no bundle guarantee (information-arbitrage-free only)"
            elif mech not in ARBITRAGE_FREE:
                cause = "baseline without an arbitrage guarantee"
            elif not covered:
                cause = "instance-specific derivation (refuted by the pricing support)"
            else:
                cause = "unexplained (conflict sets cover the target)"
            prow.append({"workload": w, "mechanism": mech, "attacker": att, "target": t, "sources": srcs,
                         "runs": len(g), "target_conflict_set": len(inc.get(str(t), set())),
                         "covered_by_sources": bool(covered), "cause": cause})
    plans_df = pd.DataFrame(prow)
    if plans_df.empty:
        return pd.DataFrame(), plans_df
    summ = plans_df.groupby(["workload", "attacker", "cause"]).agg(
        distinct_plans=("target", "size"), distinct_targets=("target", "nunique"),
        mechanisms=("mechanism", lambda x: ";".join(sorted(set(x))))).reset_index()
    return summ, plans_df


def relations_summary(s: Settings) -> pd.DataFrame:
    r = _read(s.results_dir / "extract" / "relations.csv")
    if r is None:
        return pd.DataFrame()
    r["lattice"] = r.lattice.astype(bool)
    r["query_text"] = r.query_text.astype(bool)
    rows = []
    for cid, g in r.groupby("cid"):
        row = {"cid": cid, "lattice_pairs": int(g.lattice.sum()), "query_text_pairs": int(g.query_text.sum()),
               "both": int((g.lattice & g.query_text).sum())}
        for col in [c for c in g.columns if c.startswith("refuting_")]:
            ok = g[col].notna()
            row[f"lattice_refuted_{col[9:]}"] = int((g.lattice & ok & (g[col] > 0)).sum())
            row[f"query_text_refuted_{col[9:]}"] = int((g.query_text & ok & (g[col] > 0)).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    tot = df.drop(columns=["cid"]).sum(numeric_only=True)
    return pd.concat([df, pd.DataFrame([{"cid": "all", **tot.to_dict()}])], ignore_index=True)


# ---------------------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------------------
def _study_dirs(s: Settings, tpch_studies: Path | None) -> dict:
    d = {"sqlshare": s.results_dir / "studies"}
    if tpch_studies is not None:
        d["tpch"] = tpch_studies
    return d


def study_a_contrasts(dirs: dict) -> pd.DataFrame:
    """LPIP minus UBP per study-A cell, on the Chawla et al. normalization (revenue over the
    sum of valuations) and on the main protocol's (revenue over the training monopoly
    reference); intervals over workloads, counts over settings."""
    from .study_core import add_main_norm
    frames = [pd.read_csv(p / "A" / "study_A_runs.csv") for p in dirs.values() if (p / "A" / "study_A_runs.csv").exists()]
    if not frames:
        return pd.DataFrame()
    a = add_main_norm(pd.concat(frames, ignore_index=True))
    cell = ["workload", "support", "valuation", "family", "buyers", "lp_cap", "seed", "protocol"]
    rows = []
    for metric in ("revenue_over_values", "revenue_norm_main"):
        piv = a.pivot_table(index=cell, columns="mechanism", values=metric).reset_index()
        if not {"chawla_lpip", "chawla_ubp"} <= set(piv.columns):
            continue
        piv["lpip_minus_ubp"] = piv.chawla_lpip - piv.chawla_ubp
        for key, g in piv.groupby(["support", "valuation", "buyers", "lp_cap", "protocol"]):
            per_w = g.groupby("workload").lpip_minus_ubp.mean()
            mean, lo, hi = boot_ci(per_w.to_numpy())
            pos, neg, tie = _signs(g.lpip_minus_ubp)
            rows.append(dict(zip(["support", "valuation", "buyers", "lp_cap", "protocol"], key), metric=metric,
                             workloads=len(per_w), mean_diff=mean, ci_low=lo, ci_high=hi,
                             workloads_lpip_higher=int((per_w > 1e-12).sum()),
                             settings=pos + neg + tie, settings_lpip_higher=pos, settings_ubp_higher=neg))
    return pd.DataFrame(rows)


def study_b_table(dirs: dict) -> pd.DataFrame:
    p = dirs["sqlshare"] / "B"
    if not (p / "study_B_runs.csv").exists():
        return pd.DataFrame()
    sp = pd.read_csv(p / "study_B_supports.csv")
    r = pd.read_csv(p / "study_B_runs.csv")
    keys = ["sampler", "condition", "currency"]
    q = sp.groupby(keys)[["support_n", "assets_priceable", "distinct_price_levels", "lattice_pairs_refuted",
                          "spent_seconds", "spent_queries"]].mean().reset_index()
    sb = [m for m in ("qirana_weighted_coverage", "qirana_shannon", "qirana_uniform_entropy_gain", "chawla_lpip",
                      "chawla_xos") if m in set(r.mechanism)]
    cols = [c for c in ("revenue_strategic_op_norm", "welfare_strategic_op_norm", "comb_violation_rate_op") if c in r]
    d = r[r.mechanism.isin(sb)].groupby(keys + ["mechanism"])[cols].mean().reset_index()
    return d.merge(q, on=keys, how="left")


def study_c_comparisons(dirs: dict) -> pd.DataFrame:
    frames = [pd.read_csv(p / "C" / "study_C_matched.csv") for p in dirs.values()
              if (p / "C" / "study_C_matched.csv").exists()]
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    m["ok"] = m.reached.astype(bool) & ~m.at_grid_edge.astype(bool)
    rows = []
    for (w, view, match, lvl), g in m.groupby(["workload", "view", "match", "level"]):
        piv = g[g.ok].pivot_table(index=["family", "seed"], columns="mechanism", values="welfare_norm")
        if BASELINE not in piv:
            continue
        for mech in piv.columns:
            if mech == BASELINE:
                continue
            d = (piv[mech] - piv[BASELINE]).dropna()
            mean, lo, hi = boot_ci(d.to_numpy())
            pos, neg, tie = _signs(d)
            rows.append({"workload": w, "view": view, "match": match, "level": lvl, "mechanism": mech,
                         "units_matched_both": len(d), "welfare_minus_uniform": mean, "ci_low": lo, "ci_high": hi,
                         "settings_mech_higher": pos, "settings_uniform_higher": neg, "settings_equal": tie})
    return pd.DataFrame(rows)


def study_d_comparisons(dirs: dict) -> pd.DataFrame:
    rows = []
    for p in dirs.values():
        f = p / "D" / "study_D_runs.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        for metric in SINGLE_METRICS:
            if metric not in r:
                continue
            piv = r.pivot_table(index=["workload", "mechanism", "family", "seed"], columns="donors", values=metric)
            if not {"distinct", "frequency"} <= set(piv.columns):
                continue
            piv["d"] = piv.frequency - piv.distinct
            for (w, mech), g in piv.groupby(level=[0, 1]):
                mean, lo, hi = boot_ci(g.d.to_numpy())
                pos, neg, tie = _signs(g.d)
                rows.append({"workload": w, "mechanism": mech, "metric": metric, "frequency_minus_distinct": mean,
                             "ci_low": lo, "ci_high": hi, "settings_frequency_higher": pos,
                             "settings_distinct_higher": neg, "settings_equal": tie})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def stage_analyze(s: Settings, tpch: Path | None = None, tpch_studies: Path | None = None,
                  with_support_costs: bool = True) -> dict:
    if s.protocol != "revised":
        raise ValueError("the analysis is for the revised protocol (--protocol revised)")
    t0 = time.perf_counter()
    out = s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tpch = Path(tpch) if tpch else None
    tpch_studies = Path(tpch_studies) if tpch_studies else None
    runs = load_runs(s, tpch)
    LOG.info("analysis: workloads %s", sorted(runs))
    fun, fam = funnel(s)
    write_csv(fun, out / "funnel.csv")
    write_csv(fam, out / "lattice_families.csv")
    mr = main_results(runs)
    write_csv(mr, out / "main_results.csv")
    comp = comparisons(runs)
    write_csv(comp, out / "comparisons.csv")
    write_csv(per_family(runs), out / "per_family.csv")
    write_csv(arms_sizes(s, tpch), out / "arms_sizes.csv")
    sup_rows = support_seconds(s) if with_support_costs else []
    write_csv(costs(s, runs, tpch, sup_rows), out / "costs.csv")
    cs, cp = causes(s, tpch)
    write_csv(cs, out / "causes.csv")
    write_csv(cp, out / "causes_plans.csv")
    rel = relations_summary(s)
    write_csv(rel, out / "relations_summary.csv")
    dirs = _study_dirs(s, tpch_studies)
    sa, sb, sc, sd = study_a_contrasts(dirs), study_b_table(dirs), study_c_comparisons(dirs), study_d_comparisons(dirs)
    write_csv(sa, out / "study_A_contrasts.csv")
    write_csv(sb, out / "study_B_table.csv")
    write_csv(sc, out / "study_C_comparisons.csv")
    write_csv(sd, out / "study_D_comparisons.csv")
    numbers = {"settings": {"bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED, "level": LEVEL,
                            "baseline": BASELINE, "workloads": sorted(runs)},
               "funnel": fun.to_dict(orient="records"),
               "lattice_families": fam.to_dict(orient="records"),
               "relations_total": rel[rel.cid == "all"].to_dict(orient="records") if len(rel) else [],
               "causes": cs.to_dict(orient="records") if len(cs) else [],
               "seconds": round(time.perf_counter() - t0, 1)}
    write_json(numbers, out / "paper_numbers.json")
    LOG.info("analysis: %s (%.0f s)", out, time.perf_counter() - t0)
    return numbers

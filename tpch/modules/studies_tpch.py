"""Revision 3, Part 7c: studies A, C and D on TPC-H (study B is SQLShare only).

    python tpch/run_tpch.py --study A,C,D --out <folder with revised and revised_delete>

Reads the finished revised TPC-H runs (``<out>/revised`` with Qirana updates and
``<out>/revised_delete`` with row deletions; their market.pkl holds the catalog, answers,
support, scores and measured timings) and writes ``<study-out>/{A,C,D}``. The revised runs
are not changed. Each unit of work writes its own file, so a stopped run continues.

  A  the Chawla et al. factorial (see sqlshare_rerun/study_core.py) on both supports.
  C  the main-arm runs refitted with the price-multiplier sweep (naive and operational
     views); the multiplier-1 rows must equal the revised runs (checked).
  D  a Qirana support whose new values are copied from a random lineitem row
     (frequency-weighted donors), same seed and size; main-arm runs refitted on it and
     compared with the revised runs (distinct-value donors).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import revised_tpch as R
from sqlshare_rerun import pricing as P
from sqlshare_rerun import study_core as C
from sqlshare_rerun import studies as SS


def _load_market(folder):
    with open(os.path.join(folder, "market.pkl"), "rb") as fh:
        return pickle.load(fh)


def _settings_of(folder) -> R.TpchSettings:
    with open(os.path.join(folder, "build.json"), encoding="utf-8") as fh:
        st = json.load(fh)["settings"]
    st = {k: (tuple(v) if isinstance(v, list) else v) for k, v in st.items()}
    return R.TpchSettings(**st)


def study_a(in_dir, out, workers, log=print):
    for sub, support in (("revised", "updates"), ("revised_delete", "deletions")):
        folder = os.path.join(in_dir, sub)
        if not os.path.exists(os.path.join(folder, "build.done")):
            log(f"study A: {folder} not found; the {support} support is skipped")
            continue
        s = _settings_of(folder)
        mk = _load_market(folder)
        SS.run_a_workload(Path(out) / "_work", Path(out), "tpch", support, mk["edges"], mk["support_n"], mk["info"],
                          workers, s.seeds, s.cip_eps, s.n_buyers)
        log(f"study A: tpch ({support}) done")
    SS.collect_a(Path(out))


def _main_tasks(s: R.TpchSettings):
    return [(s.main_arm, f, sd, s.n_buyers, True) for f in R.FAMILIES for sd in s.seeds]


def _run_tasks(market, mpath, tdir, s, workers, label):
    os.makedirs(tdir, exist_ok=True)
    tmp = mpath + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(market, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, mpath)
    tasks = [(mpath, tdir) + t for t in _main_tasks(s)]
    todo = [t for t in tasks if not os.path.exists(os.path.join(tdir, R.task_name(*t[2:5]) + ".csv"))]
    if todo:
        from sqlshare_rerun.parallel import run_pool
        run_pool(R.run_task, todo, workers, label)
    return [R.task_name(*t[2:5]) for t in tasks]


def study_c(in_dir, out, workers, log=print):
    folder = os.path.join(in_dir, "revised")
    s = _settings_of(folder)
    mk = _load_market(folder)
    mk["cfg"] = dict(mk["cfg"], sweep=True)
    tdir = os.path.join(out, "tasks")
    names = _run_tasks(mk, os.path.join(out, "market_sweep.pkl"), tdir, s, workers, "study C TPC-H")
    runs = pd.concat([pd.read_csv(os.path.join(tdir, n + ".csv")) for n in names], ignore_index=True)
    matched = pd.concat([pd.read_csv(os.path.join(tdir, n + ".matched.csv")) for n in names], ignore_index=True)
    curve = pd.concat([pd.read_csv(os.path.join(tdir, n + ".curve.csv")) for n in names], ignore_index=True)
    for df in (runs, matched, curve):
        df.insert(0, "workload", "tpch")
    runs.to_csv(os.path.join(out, "study_C_runs.csv"), index=False)
    matched.to_csv(os.path.join(out, "study_C_matched.csv"), index=False)
    curve.to_csv(os.path.join(out, "study_C_curves.csv"), index=False)
    matched["matched"] = matched.reached.astype(bool) & ~matched.at_grid_edge.astype(bool)
    keys = ["workload", "view", "match", "level", "mechanism"]
    g = matched.groupby(keys).agg(runs=("matched", "size"), matched_share=("matched", "mean")).reset_index()
    v = matched[matched.matched].groupby(keys).agg(
        welfare_norm=("welfare_norm", "mean"), revenue_norm=("revenue_norm", "mean"),
        served=("served", "mean"), multiplier=("multiplier", "mean")).reset_index()
    g.merge(v, how="left", on=keys).to_csv(os.path.join(out, "study_C_summary.csv"), index=False)
    # the multiplier-1 rows must equal the revised runs (same market file, same seeds)
    ref = pd.read_csv(os.path.join(folder, "runs.csv"))
    ref = ref[(ref.arm == s.main_arm) & (ref.n_train == s.n_buyers)]
    j = runs.merge(ref, on=["family", "seed", "mechanism"], suffixes=("", "_main"))
    rows = []
    for mech, gg in j.groupby("mechanism"):
        row = {"workload": "tpch", "mechanism": mech, "runs": len(gg)}
        for c in ("revenue_naive_norm", "revenue_strategic_op_norm", "welfare_strategic_op_norm",
                  "comb_violation_rate_op"):
            if c in gg and c + "_main" in gg:
                row[f"max_abs_diff_{c}"] = float(np.nanmax(np.abs(gg[c] - gg[c + "_main"])))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "study_C_consistency.csv"), index=False)
    log("study C: TPC-H done")


def study_d(in_dir, out, workers, con, log=print):
    folder = os.path.join(in_dir, "revised")
    s = _settings_of(folder)
    sf = dataclasses.replace(s, qirana_donors="frequency")
    mk = _load_market(folder)
    sup_path = os.path.join(out, "support_frequency.pkl")
    if os.path.exists(sup_path):
        with open(sup_path, "rb") as fh:
            edges, fps, info = pickle.load(fh)
    else:
        t0 = time.perf_counter()
        edges, fps, info = R.build_support(con, mk["assets"], sf)
        info["seconds"] = time.perf_counter() - t0
        tmp = sup_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump((edges, fps, info), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, sup_path)
    q = P.qirana_prices(edges, info["n"], changed=fps)
    mf = dict(mk)
    mf["edges"], mf["support_n"] = edges, info["n"]
    mf["scores"] = dict(mk["scores"], qirana_weighted_coverage=q["qirana_weighted_coverage"],
                        qirana_uniform_entropy_gain=q["qirana_uniform_entropy_gain"],
                        qirana_shannon=np.maximum(q["qirana_shannon"], 0.0))
    tdir = os.path.join(out, "tasks")
    names = _run_tasks(mf, os.path.join(out, "market_frequency.pkl"), tdir, sf, workers, "study D TPC-H")
    runs_f = pd.concat([pd.read_csv(os.path.join(tdir, n + ".csv")) for n in names], ignore_index=True)
    base = pd.read_csv(os.path.join(folder, "runs.csv"))
    base = base[(base.arm == s.main_arm) & (base.n_train == s.n_buyers)]
    runs = pd.concat([base.assign(donors="distinct"), runs_f.assign(donors="frequency")], ignore_index=True)
    runs.insert(0, "workload", "tpch")
    runs.to_csv(os.path.join(out, "study_D_runs.csv"), index=False)
    pairs_named = [(i, j) for i, j in mk["pairs"]]
    keys = list(range(len(mk["assets"])))
    srows = []
    with open(os.path.join(folder, "pricing_support.json"), encoding="utf-8") as fh:
        base_info = json.load(fh)
    for donors, eds, inf in (("distinct", mk["edges"], base_info), ("frequency", edges, info)):
        inc = {k: sorted(e) for k, e in zip(keys, eds) if len(e)}
        st = C.support_stats(keys, inc, inf["n"], pairs_named)
        srows.append({"workload": "tpch", "donors": donors, "rejected": inf.get("rejected_draws"),
                      "build_seconds": inf.get("seconds"), **st})
    pd.DataFrame(srows).to_csv(os.path.join(out, "study_D_supports.csv"), index=False)
    cols = [c for c in ("revenue_naive_norm", "revenue_strategic_op_norm", "welfare_strategic_op_norm",
                        "comb_violation_rate_op", "mono_violation_rate") if c in runs]
    runs.groupby(["workload", "donors", "mechanism"])[cols].mean().reset_index().to_csv(
        os.path.join(out, "study_D_summary.csv"), index=False)
    log("study D: TPC-H done")


def run_studies(in_dir, study_out, which, workers, con=None, log=print):
    times = {}
    for w in which:
        t0 = time.perf_counter()
        out = os.path.join(study_out, w)
        os.makedirs(os.path.join(out, "tasks"), exist_ok=True)
        if w == "A":
            study_a(in_dir, out, workers, log)
        elif w == "C":
            study_c(in_dir, out, workers, log)
        elif w == "D":
            if con is None:
                raise RuntimeError("study D needs the TPC-H database")
            study_d(in_dir, out, workers, con, log)
        else:
            raise ValueError(f"study {w} does not run on TPC-H (A, C and D do)")
        times[w] = round(time.perf_counter() - t0, 1)
        log(f"study {w} finished ({times[w]:.0f}s)")
    path = os.path.join(study_out, "studies_summary.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
    old.setdefault("seconds", {}).update(times)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=1)
    return times

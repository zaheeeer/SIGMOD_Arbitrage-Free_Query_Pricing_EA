"""The recomposing adversary and the fit, freeze, and evaluate runs.

``plan_all`` is the historical ``sqlshare_plan_all``: for every set-semantics target it
solves a min-cost row cover and accepts the plan only if it undercuts the posted price
and its re-executed union equals the target answer. A cover can undercut the price only
if every source it uses is itself cheaper than the target, so targets whose cheaper
sources cannot cover all rows are skipped without an ILP. This is exact: it changes no
outcome, only the number of solver calls.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .lattice import frames_equal
from .pricing import (audit_monotone, buyer_arms, chawla_all, draw_buyers, enforce_monotone,
                      monopoly_prices, scaled_prices, score)
from .operational import PurchasedAnswers, compare_rebuilt, execute_plan, plan_operational
from .parallel import run_pool
from .util import load_pickle

INF = float("inf")


def min_cost_row_cover(contrib: dict, n_rows: int, prices: dict, time_limit=5.0):
    from scipy.optimize import Bounds, LinearConstraint, milp
    if not contrib or n_rows == 0:
        return INF, []
    src = sorted(contrib)
    A = np.zeros((n_rows, len(src)))
    for c, s in enumerate(src):
        A[contrib[s], c] = 1.0
    if not np.all(A.sum(axis=1) > 0):
        return INF, []
    cost = np.array([prices[s] for s in src], dtype=float)
    if not np.all(np.isfinite(cost)):
        keep = np.isfinite(cost)
        if not keep.any():
            return INF, []
        A, cost, src = A[:, keep], cost[keep], [s for s, k in zip(src, keep) if k]
        if not np.all(A.sum(axis=1) > 0):
            return INF, []
    res = milp(c=cost, constraints=[LinearConstraint(A, lb=np.ones(n_rows), ub=np.inf)],
               integrality=np.ones(len(src)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if not res.success or res.x is None:
        return INF, []
    chosen = [src[k] for k in range(len(src)) if res.x[k] > 0.5]
    return float(res.fun), chosen


def verify_row_cover(answers, target, plan) -> bool:
    if not plan:
        return False
    T = answers[target]
    cols = list(T.columns)
    got = pd.concat([answers[s].loc[:, cols] for s in plan], ignore_index=True).drop_duplicates()
    return frames_equal(got.reset_index(drop=True), T.reset_index(drop=True))


def plan_all(p, keys, ki, contrib, nrows, answers, time_limit=5.0, exact_formulation=True,
             return_plans=False):
    """Effective prices under the recomposing adversary.

    Returns (cost vector, plan sizes, plans verified, plans failed). ``exact_formulation``
    solves the ILP over all candidate sources, as the historical run did, whenever a
    cheaper cover is possible; otherwise only over the cheaper sources (same optimum).
    """
    cost = np.asarray(p, float).copy()
    psize = np.ones(len(keys), int)
    vok = vbad = 0
    plans = {}
    pm = {k: float(p[ki[k]]) for k in keys}
    for k, c_ in contrib.items():
        pk = pm[k]
        cheap = {s: idx for s, idx in c_.items() if pm[s] < pk - 1e-9}
        if not cheap:
            continue
        covered = np.zeros(nrows[k], bool)
        for idx in cheap.values():
            covered[idx] = True
        if not covered.all():
            continue
        cc_, plan = min_cost_row_cover(c_ if exact_formulation else cheap, nrows[k], pm, time_limit)
        if plan and cc_ < pk - 1e-9:
            try:
                ok = verify_row_cover(answers, k, plan)
            except Exception:
                ok = False
            plans[k] = (float(cc_), list(plan), bool(ok))
            if ok:
                cost[ki[k]] = cc_
                psize[ki[k]] = len(plan)
                vok += 1
            else:
                vbad += 1
    if return_plans:
        return cost, psize, vok, vbad, plans
    return cost, psize, vok, vbad


# ----------------------------------------------------------------------------------------
# one (arm, family, seed) run of the full protocol
# ----------------------------------------------------------------------------------------
_MARKET = {}


def _market(path):
    if path not in _MARKET:
        _MARKET.clear()
        _MARKET[path] = load_pickle(Path(path))
    return _MARKET[path]


def run_task(args):
    """One (arm, family, seed) run. Revision 3 adds two optional fields: the training size
    and whether the attackers plan (the D7 study arms only measure revenue out of sample)."""
    path, arm, family, seed = args[:4]
    d = _market(path)
    cfg = d["cfg"]
    n_train = int(args[4]) if len(args) > 4 else cfg["n_buyers"]
    planned = bool(args[5]) if len(args) > 5 else True
    keys, ki, n = d["keys"], d["ki"], len(d["keys"])
    info = d["info"][family]
    demand_arm = arm.split("_n")[0]
    shift_stats = {}
    if cfg["demand"] == "arms":
        (tr_t, tr_v), (te_t, te_v), shift_stats = buyer_arms(info, 1000 * seed + 7, demand_arm,
                                                             n_train, cfg["n_buyers"])
    else:
        (tr_t, tr_v), (te_t, te_v) = draw_buyers(cfg["demand"], arm, info, 1000 * seed + 7,
                                                 n_train, cfg["n_buyers"])
    prices, alphas, fit_s = {}, {}, {}
    for name, sc in d["scale_scores"].items():
        t0 = time.perf_counter()
        prices[name], alphas[name] = scaled_prices(cfg["fit"], tr_t, tr_v, sc)
        fit_s[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mono = monopoly_prices(tr_t, tr_v, n)
    prices["monopoly_per_asset"] = mono
    fit_s["monopoly_per_asset"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prices["monotone_constrained"] = enforce_monotone(mono, d["pairs"])
    fit_s["monotone_constrained"] = time.perf_counter() - t0
    ch_s = {}
    prices.update(chawla_all(d["edges"], tr_t, tr_v, d["support_n"], max_lps=cfg["max_lps"],
                             eps=cfg["cip_eps"], seconds=ch_s, compact=bool(cfg.get("compact_lp"))))
    fit_s.update(ch_s)
    ref_train = score(tr_t, tr_v, prices["monopoly_per_asset"])["revenue"]
    # Test revenues are compared with the training reference scaled to the test size
    # (revision 3, Part 4 fix): with n_train != n_test the unscaled reference was off by
    # n_test / n_train. When the two sizes are equal the factor is exactly 1.0.
    size_ratio = cfg["n_buyers"] / n_train
    ref = ref_train * size_ratio
    # Part 5: the per-asset optimum on the test buyers themselves (monopoly prices fitted
    # and scored on the test sample). It does not depend on the training sample, so it is
    # a common reference for comparing test arms and training sizes in the analysis.
    ref_test_opt = score(te_t, te_v, monopoly_prices(te_t, te_v, n))["revenue"] if cfg.get("record_costs") else np.nan
    rows, plan_rows, sweep_m, sweep_c = [], [], [], []
    op = d.get("op")
    do_sweep = bool(cfg.get("sweep"))             # revision 3, study C
    for name, p in prices.items():
        t0 = time.perf_counter()
        if planned:
            cost, psz, vok, vbad, oplans = plan_all(p, keys, ki, d["contrib"], d["nrows"], d["answers"],
                                                    cfg["ilp_time_limit"], cfg["exact_formulation"],
                                                    return_plans=True)
        else:
            cost, vok, vbad, oplans = np.asarray(p, float).copy(), 0, 0, {}
        plan_s = time.perf_counter() - t0
        if cfg.get("log_plans"):
            for t, (c_, plan, ok) in oplans.items():
                plan_rows.append(_plan_row(arm, family, seed, name, "oracle", t, p[ki[t]], c_, plan, ok))
        eff = np.minimum(p, cost)
        nai = score(te_t, te_v, p)
        stra = score(te_t, te_v, eff)
        ins = score(tr_t, tr_v, p)["revenue"]
        ma = audit_monotone(p, d["pairs"])
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = np.where(np.isinf(p), np.where(np.isfinite(eff), 1.0, 0.0),
                            np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0))
        a = alphas.get(name)
        rows.append({
            "arm": arm, "family": family, "seed": seed, "mechanism": name,
            "revenue_train_norm": ins / max(ref_train, 1e-12),
            "revenue_naive_norm": nai["revenue"] / max(ref, 1e-12),
            "revenue_strategic_norm": stra["revenue"] / max(ref, 1e-12),
            "surplus_naive_norm": nai["surplus"] / max(ref, 1e-12),
            "welfare_naive_norm": nai["welfare"] / max(ref, 1e-12),
            "surplus_strategic_norm": stra["surplus"] / max(ref, 1e-12),
            "welfare_strategic_norm": stra["welfare"] / max(ref, 1e-12),
            "generalization_gap": 1 - nai["revenue"] / max(ins * size_ratio, 1e-12),
            "mono_violation_rate": ma["violation_rate"],
            "comb_violation_rate": float(np.mean(gain > 1e-6)),
            "comb_mean_gain": float(np.mean(gain)),
            "comb_max_gain": float(gain.max()),
            "served_naive": nai["served"], "served_strategic": stra["served"],
            "plans_verified": int(vok), "plans_failed": int(vbad),
            "alpha": a if a is not None else np.nan,
            "alpha_at_grid_max": bool(a is not None and cfg["fit"] == "grid60" and a >= 60.0 - 1e-9),
            "unsellable_assets": int(np.isinf(p).sum()),
            **({"n_train": n_train, "n_test": int(cfg["n_buyers"]), "planned": planned,
                "revenue_train": ins, "revenue_naive": nai["revenue"],
                "revenue_strategic": stra["revenue"], "ref_monopoly_train": ref_train,
                "ref_monopoly_test": ref_test_opt,
                "fit_seconds": fit_s.get(name, np.nan),
                "plan_seconds_oracle": plan_s if planned else np.nan,
                **shift_stats} if cfg.get("record_costs") else {}),
        })
        eff_hold = {}
        if op is not None and planned:
            rows[-1].update(_operational_metrics(d, op, cfg, name, p, te_t, te_v, ref, arm, family, seed,
                                                 plan_rows, eff_hold))
        if do_sweep:
            from .study_core import sweep
            views = {"naive": p}
            if planned:
                views["oracle"] = eff
            if "operational" in eff_hold:
                views["operational"] = eff_hold["operational"]
            m_rows, c_rows = sweep(views, te_t, te_v, ref, {"arm": arm, "family": family, "seed": seed,
                                                            "mechanism": name})
            sweep_m += m_rows
            sweep_c += c_rows
    if cfg.get("log_plans"):
        rows.append({"_plans": plan_rows})
    if do_sweep:
        rows.append({"_sweep": sweep_m, "_curve": sweep_c})
    return rows


def _plan_row(arm, family, seed, mech, attacker, t, price, cost, plan, ok):
    price = float(price)
    gain = 1.0 if math.isinf(price) else (price - cost) / price if price > 0 else 0.0
    return {"arm": arm, "family": family, "seed": seed, "mechanism": mech, "attacker": attacker,
            "target": t, "price": price, "cost": float(cost), "gain": gain, "n_sources": len(plan),
            "sources": ";".join(map(str, plan)), "verified": bool(ok)}


_VERIFIED = {}


def _operational_metrics(d, op, cfg, name, p, te_t, te_v, ref, arm, family, seed, plan_rows, eff_hold=None):
    """Revision 3 (D3): the operational buyer plans from query texts and prices only; each
    plan is executed on the purchased answers alone and then verified against the target."""
    keys, ki = d["keys"], d["ki"]
    pm = {k: float(p[ki[k]]) for k in keys}
    t0 = time.perf_counter()
    plans = plan_operational(pm, op["structures"],
                             lambda c, n, pr: min_cost_row_cover(c, n, pr, cfg["ilp_time_limit"]))
    cost = np.full(len(keys), np.inf)
    ok_n = bad_n = 0
    for t, (c_, plan) in plans.items():
        vkey = (t, tuple(sorted(map(str, plan))))
        if vkey not in _VERIFIED:
            try:
                got = execute_plan(op["forms"][t], plan, op["forms"], op["structures"][t][2],
                                   PurchasedAnswers(d["answers"], plan))
                _VERIFIED[vkey] = compare_rebuilt(got, d["answers"][t])
            except Exception as e:
                _VERIFIED[vkey] = (False, f"error {type(e).__name__}: {str(e)[:120]}")
        ok, note = _VERIFIED[vkey]
        if ok:
            cost[ki[t]] = c_
            ok_n += 1
        else:
            bad_n += 1
        if cfg.get("log_plans"):
            plan_rows.append(dict(_plan_row(arm, family, seed, name, "operational", t, p[ki[t]], c_, plan, ok),
                                  verify_note=note))
    eff = np.minimum(p, cost)
    if eff_hold is not None:
        eff_hold["operational"] = eff
    stra = score(te_t, te_v, eff)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(np.isinf(p), np.where(np.isfinite(eff), 1.0, 0.0),
                        np.where(p > 0, (p - eff) / np.maximum(p, 1e-12), 0.0))
    sem = audit_monotone(p, op["sem_pairs"]) if op["sem_pairs"] else {"violation_rate": 0.0}
    return {
        "revenue_strategic_op_norm": stra["revenue"] / max(ref, 1e-12),
        "surplus_strategic_op_norm": stra["surplus"] / max(ref, 1e-12),
        "welfare_strategic_op_norm": stra["welfare"] / max(ref, 1e-12),
        "served_strategic_op": stra["served"],
        "comb_violation_rate_op": float(np.mean(gain > 1e-6)),
        "comb_mean_gain_op": float(np.mean(gain)),
        "comb_max_gain_op": float(gain.max()),
        "plans_verified_op": int(ok_n), "plans_failed_op": int(bad_n),
        "mono_violation_rate_sem": sem["violation_rate"],
        **({"plan_seconds_operational": time.perf_counter() - t0} if cfg.get("record_costs") else {}),
    }


def run_protocol(market_pickle: Path, arms, families, seeds, workers: int, stall_s: float = 1800.0,
                 with_plans: bool = False, tasks=None):
    """``tasks`` (revision 3) is an explicit list of (arm, family, seed, n_train, planned)."""
    if tasks is None:
        tasks = [(str(market_pickle), a, f, s) for a in arms for f in families for s in seeds]
    else:
        tasks = [(str(market_pickle),) + tuple(t) for t in tasks]
    try:
        out = run_pool(run_task, tasks, workers, "fit, freeze, and evaluate runs", stall_s)
    finally:
        _MARKET.clear()
        _VERIFIED.clear()
    runs = pd.DataFrame([r for rows in out for r in rows if "_plans" not in r and "_sweep" not in r])
    if not with_plans:
        return runs
    plans = pd.DataFrame([p for rows in out for r in rows if "_plans" in r for p in r["_plans"]])
    return runs, plans

"""Revision 3, Part 7c: studies A-D on the synthetic SQLShare release."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import study_core as C                          # noqa: E402
from sqlshare_rerun.cli import _args, run                           # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402

EXTRA = dict(SELFTEST_SETTINGS, demand_arms=("same",), train_sizes=(), max_lps=0)


@pytest.fixture(scope="module")
def studied(tmp_path_factory):
    root = tmp_path_factory.mktemp("stu")
    (root / "data").mkdir()
    build_fixture(root / "data" / "sqlshare_data_release1.zip")
    assert run(_args(["all", "--protocol", "revised", "--root", str(root)]), **EXTRA) == 0
    cache = root / "work" / "sample_cache_revised"
    before = {p.name for p in cache.glob("*.pkl")}
    struct = pd.read_csv(root / "results" / "revised" / "markets" / "structure.csv")
    extra = {"study_b_markets": tuple(int(c) for c in struct.cid), "study_b_seeds": (0,), "study_b_n": 25}
    assert run(_args(["study", "--protocol", "revised", "--root", str(root)]), **EXTRA, extra=extra) == 0
    after = {p.name for p in cache.glob("*.pkl")}
    return root, before, after


def _st(root, w, f):
    return pd.read_csv(root / "results" / "revised" / "studies" / w / f)


def test_main_supports_are_reused_not_redrawn(studied):
    root, before, after = studied
    assert before <= after
    new = after - before
    assert all(n.startswith(("studyB_", "qirana_")) for n in new)       # B draws and the D support only


def test_study_a_reproduces_the_main_runs(studied):
    root, _, _ = studied
    a = _st(root, "A", "study_A_runs.csv")
    main = pd.read_csv(root / "results" / "revised" / "main_market" / "runs.csv")
    main = main[main.arm == "same"]
    x = a[(a.support == "updates") & (a.valuation == "ours") & (a.buyers == "popularity") & (a.lp_cap == 0)
          & (a.protocol == "held_out") & (a.workload == a.workload.iloc[0])]
    j = x.merge(main, on=["family", "seed", "mechanism"])
    assert len(j) == len(x) > 0
    assert np.allclose(j.revenue, j.revenue_naive)
    xi = a[(a.support == "updates") & (a.valuation == "ours") & (a.buyers == "popularity") & (a.lp_cap == 0)
           & (a.protocol == "in_sample") & (a.workload == a.workload.iloc[0])]
    ji = xi.merge(main, on=["family", "seed", "mechanism"])
    assert np.allclose(ji.revenue, ji.revenue_train)
    assert {"deletions", "updates"} <= set(a.support)
    per = a[(a.buyers == "per_query") & (a.protocol == "in_sample")]
    assert np.allclose(per.revenue_over_values, per.revenue_over_monopoly)   # monopoly = sum of values
    s = _st(root, "A", "study_A_summary.csv")
    assert s.rank_in_cell.min() == 1


def test_study_c_equals_the_main_results_at_multiplier_one(studied):
    root, _, _ = studied
    c = _st(root, "C", "study_C_consistency.csv")
    cols = [k for k in c.columns if k.startswith("max_abs_diff")]
    other = c[c.mechanism != "compute_metered"]
    assert len(other) > 0 and (other[cols].to_numpy() < 1e-9).all()
    m = _st(root, "C", "study_C_matched.csv")
    assert set(m.view) == {"naive", "oracle", "operational"}
    s = _st(root, "C", "study_C_summary.csv")
    assert {"matched_share", "welfare_norm"} <= set(s.columns)


def test_study_b_budgets(studied):
    root, _, _ = studied
    sp = _st(root, "B", "study_B_supports.csv")
    assert set(sp.sampler) == {"size_deletions", "readership_deletions", "qirana_updates"}
    eq = sp[sp.condition == "equal_size"]
    assert (eq.support_n == 25).all()
    bud = sp[sp.condition != "equal_size"]
    sec = bud[bud.currency == "seconds"]
    q = bud[bud.currency == "queries"]
    assert (sec.spent_seconds <= sec.budget_seconds + 1e-9).all()
    assert (q.spent_queries <= q.budget_queries).all()
    r = _st(root, "B", "study_B_runs.csv")
    assert (r.plans_failed_op == 0).all()


def test_study_d_compares_donors(studied):
    root, _, _ = studied
    s = _st(root, "D", "study_D_supports.csv")
    assert list(s.donors) == ["distinct", "frequency"]
    r = _st(root, "D", "study_D_runs.csv")
    base = r[r.donors == "distinct"]
    main = pd.read_csv(root / "results" / "revised" / "main_market" / "runs.csv")
    main = main[main.arm == "same"]
    j = base.merge(main, on=["family", "seed", "mechanism"], suffixes=("", "_m"))
    keep = j.mechanism != "compute_metered"
    assert np.allclose(j[keep].revenue_strategic_op_norm, j[keep].revenue_strategic_op_norm_m)


def test_sweep_matching_rules():
    t = np.array([0, 1, 0, 1])
    v = np.array([10.0, 10.0, 4.0, 4.0])
    m, cur = C.sweep({"naive": np.array([4.0, 8.0])}, t, v, 10.0, {"mechanism": "x"})
    cov = {r["level"]: r for r in m if r["match"] == "coverage"}
    # buyers stop at m = 2.5, 1.25, 1.0 and 0.5: served is 1 up to 0.5, 0.75 up to 1.0,
    # 0.5 up to 1.25, 0.25 up to 2.5 and 0 above
    assert cov[0.75]["multiplier"] == 1.0 and cov[0.5]["multiplier"] == 1.25 and cov[0.25]["multiplier"] == 2.5
    rev = {r["level"]: r for r in m if r["match"] == "revenue"}
    # up to m = 0.5 all four buy: revenue / 10 = 2.4 m, so 0.3 is first reached at m = 0.13
    assert rev[0.3]["multiplier"] == 0.13 and rev[0.3]["at_grid_edge"] is False
    assert len(cur) == len(C.CURVE_GRID)


def test_budget_prefix_helpers():
    trace = [(0, 1.0, 3), (-1, 0.5, 2), (1, 1.0, 3), (2, 2.0, 1)]
    assert C.cost_of_first(trace, 2) == (2.5, 8, 3)
    assert C.accepted_within(trace, 2.5, "seconds") == 2
    assert C.accepted_within(trace, 8, "queries") == 2
    assert C.accepted_within(trace, 9, "queries") == 3
    assert C.accepted_within(trace, 0.5, "seconds") == 0


def test_analysis_tables(studied):
    root, _, _ = studied
    assert run(_args(["analyze", "--protocol", "revised", "--root", str(root)]), **EXTRA) == 0
    out = root / "results" / "revised" / "analysis"
    for f in ("funnel.csv", "lattice_families.csv", "main_results.csv", "comparisons.csv", "per_family.csv",
              "arms_sizes.csv", "costs.csv", "relations_summary.csv", "study_A_contrasts.csv",
              "study_C_comparisons.csv", "study_D_comparisons.csv", "paper_numbers.json"):
        assert (out / f).exists(), f
    c = pd.read_csv(out / "comparisons.csv")
    assert {"sqlshare_main", "sqlshare_15"} <= set(c.workload)
    assert (c.ci_low <= c.mean_diff + 1e-12).all() and (c.mean_diff <= c.ci_high + 1e-12).all()
    assert (c.settings == c.settings_a_higher + c.settings_b_higher + c.settings_equal).all()
    costs = pd.read_csv(out / "costs.csv")
    assert costs.item.str.startswith("support build").any()


def test_bootstrap_interval():
    from sqlshare_rerun.analysis import boot_ci
    m, lo, hi = boot_ci(np.ones(20))
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, 400)
    m, lo, hi = boot_ci(x)
    assert lo < m < hi and abs((hi - lo) - 2 * 1.96 / 20) < 0.05

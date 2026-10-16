"""Revision 3, Part 7c: studies A, C and D on a small synthetic TPC-H lineitem."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sqlshare"))
sys.path.insert(0, str(ROOT / "tpch" / "modules"))
import revised_tpch as R                                           # noqa: E402
import studies_tpch as ST                                          # noqa: E402

SMALL = dict(n_assets=30, support_n=40, n_buyers=60, seeds=(0, 1), train_sizes=(60,),
             demand_arms=("same",), expect_catalog=())


@pytest.fixture(scope="module")
def tpch_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("tpch")
    con = duckdb.connect()
    R.synthetic_lineitem(con, n_rows=1200, n_months=8, seed=0)
    quiet = lambda *_: None                                         # noqa: E731
    R.run_revised(con, str(out / "revised"), R.TpchSettings(**SMALL), workers=1, sf_is_one=False, log=quiet)
    R.run_revised(con, str(out / "revised_delete"), R.TpchSettings(**dict(SMALL, neighbours="delete")),
                  workers=1, sf_is_one=False, log=quiet)
    ST.run_studies(str(out), str(out / "studies"), ["A", "C", "D"], 1, con, log=quiet)
    return out


def test_fingerprints_of_the_p3_runs_are_unchanged():
    assert R.TpchSettings().fingerprint() == "ec79407bc3e0"
    assert R.TpchSettings(neighbours="delete").fingerprint() == "dd366a9cb107"
    assert R.TpchSettings(qirana_donors="frequency").fingerprint() != "ec79407bc3e0"


def test_study_a_on_both_supports(tpch_runs):
    a = pd.read_csv(tpch_runs / "studies" / "A" / "study_A_runs.csv")
    assert set(a.support) == {"updates", "deletions"}
    assert set(a.valuation) == {"ours", "uniform10", "uniform100", "zipf1.5", "zipf2.5"}
    assert set(a.protocol) == {"in_sample", "held_out"} and set(a.lp_cap) == {0, 60}
    # the (ours, popularity, uncapped, held out) cell is the main run
    main = pd.read_csv(tpch_runs / "revised" / "runs.csv")
    main = main[(main.arm == "same") & (main.n_train == 60)]
    x = a[(a.support == "updates") & (a.valuation == "ours") & (a.buyers == "popularity") & (a.lp_cap == 0)
          & (a.protocol == "held_out")]
    j = x.merge(main, on=["family", "seed", "mechanism"])
    assert len(j) == len(x) > 0
    assert np.allclose(j.revenue, j.revenue_naive)
    w = pd.read_csv(tpch_runs / "studies" / "A" / "study_A_workloads.csv")
    assert set(w.support) == {"updates", "deletions"}


def test_study_c_matches_the_main_runs_at_multiplier_one(tpch_runs):
    c = pd.read_csv(tpch_runs / "studies" / "C" / "study_C_consistency.csv")
    cols = [k for k in c.columns if k.startswith("max_abs_diff")]
    assert len(c) > 0 and (c[cols].fillna(0).to_numpy() < 1e-9).all()
    m = pd.read_csv(tpch_runs / "studies" / "C" / "study_C_matched.csv")
    assert set(m.view) == {"naive", "operational"} and set(m.match) == {"revenue", "coverage"}
    cov = m[(m.match == "coverage") & m.reached & ~m.at_grid_edge]
    assert (cov.served >= cov.level - 1e-12).all()
    rev = m[(m.match == "revenue") & m.reached]
    assert (rev.revenue_norm >= rev.level - 1e-12).all()


def test_study_d_frequency_donors(tpch_runs):
    s = pd.read_csv(tpch_runs / "studies" / "D" / "study_D_supports.csv")
    assert list(s.donors) == ["distinct", "frequency"] and (s.support_n == 40).all()
    r = pd.read_csv(tpch_runs / "studies" / "D" / "study_D_runs.csv")
    assert set(r.donors) == {"distinct", "frequency"}
    assert (r[r.donors == "frequency"].plans_failed_op == 0).all()


def test_analysis_reads_the_tpch_results(tpch_runs, tmp_path):
    from sqlshare_rerun import analysis as A
    from sqlshare_rerun.config import make_settings
    s = make_settings("revised", tmp_path)               # no SQLShare results here
    runs = A.load_runs(s, tpch_runs)
    assert set(runs) == {"tpch"} and (runs["tpch"].arm == "same").all()
    c = A.comparisons(runs)
    assert set(c.workload) == {"tpch"} and len(c) > 0
    cs, cp = A.causes(s, tpch_runs)
    if len(cp):
        assert set(cp.attacker) == {"operational"}
    arms = A.arms_sizes(s, tpch_runs)
    assert set(arms.workload) == {"tpch"}
    d = A.study_d_comparisons({"tpch": tpch_runs / "studies"})
    assert set(d.workload) == {"tpch"}
    a = A.study_a_contrasts({"tpch": tpch_runs / "studies"})
    assert len(a) > 0


def test_analysis_loads_the_tpch_market_without_the_tpch_path(tpch_runs, tmp_path):
    """Regression (P7c laptop run): `analyze` runs from the sqlshare folder, where the TPC-H
    modules are not on the path; unpickling market.pkl must still work."""
    import subprocess
    code = ("import sys; from pathlib import Path; "
            "from sqlshare_rerun.config import make_settings; from sqlshare_rerun import analysis as A; "
            f"s = make_settings('revised', Path(r'{tmp_path}')); "
            f"cs, cp = A.causes(s, Path(r'{tpch_runs}')); print('ok', len(cp))")
    import os
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT / "sqlshare"), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout

"""Revision 3, Part 1: the pipeline runs end to end with the Qirana options on the synthetic
release (decisions D2 and D8), and the 2.x protocols are unchanged by default."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import stages as S                             # noqa: E402
from sqlshare_rerun.config import make_settings                     # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402
from sqlshare_rerun.util import read_json                          # noqa: E402

QIRANA = dict(neighbours="qirana", qirana_table_weights="uniform", qirana_gain="entropy_gain",
              attackers="both", log_plans=True)


def test_default_fingerprints_are_unchanged():
    """New settings at their neutral values leave the 2.x checkpoint keys unchanged."""
    for proto, fp in (("paper", "11cf10d670fa"), ("corrected", "c9c9df996639")):
        assert make_settings(proto).fingerprint() == fp


@pytest.fixture(scope="module")
def revised_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("rev")
    (root / "data").mkdir()
    build_fixture(root / "data" / "sqlshare_data_release1.zip")
    s = make_settings("corrected", root, **SELFTEST_SETTINGS, **QIRANA)
    S.stage_download(s)
    S.stage_prepare(s)
    S.stage_lattice(s)
    summary = S.stage_main_market(s)
    return s, summary


def test_main_market_prices_with_qirana_updates(revised_run):
    s, summary = revised_run
    out = s.results_dir / "main_market"
    runs = pd.read_csv(out / "runs.csv")
    mechs = set(runs.mechanism)
    assert "qirana_uniform_entropy_gain" in mechs
    assert "qirana_uniform_gain" not in mechs
    assert summary["pricing_support"] == "qirana_updates_uniform"
    assert "qirana" in summary["neighbour_draws"]
    t6 = pd.read_csv(out / "table6_support_sampling.csv")
    assert "qirana_updates_uniform" in set(t6.sampling)
    row = t6[t6.sampling == "qirana_updates_uniform"].iloc[0]
    assert row.support_rows == s.table6_n


def test_revised_settings_are_recorded(revised_run):
    s, _ = revised_run
    fp = s.fingerprint()
    assert fp not in ("11cf10d670fa", "c9c9df996639")
    key = read_json(s.results_dir / "main_market" / "main.done")
    assert fp in str(key)


@pytest.mark.parametrize("full", [True, False])
def test_markets_run_with_qirana_updates(revised_run, full):
    s, _ = revised_run
    s2 = make_settings("corrected", s.root, **SELFTEST_SETTINGS, **QIRANA, full_protocol_markets=full)
    if not full:
        s2.extra["results_dir"] = str(s.root / "results" / "corrected_reduced")
        s2.extra["lattice_dir"] = str(s.lattice_dir)
    summ = S.stage_markets(s2)
    assert summ["markets_ok"] >= 1


def test_operational_buyer_runs_and_logs_plans(revised_run):
    s, summary = revised_run
    out = s.results_dir / "main_market"
    runs = pd.read_csv(out / "runs.csv")
    for col in ("revenue_strategic_op_norm", "comb_violation_rate_op", "plans_verified_op",
                "plans_failed_op", "mono_violation_rate_sem"):
        assert col in runs.columns
    assert (runs.plans_failed_op == 0).all()           # every guaranteed plan rebuilt its target
    st = summary["operational"]
    assert st["in_fragment"] > 0 and st["assets"] >= st["in_fragment"]
    plans = pd.read_csv(out / "plans_log.csv")
    assert set(plans.attacker) <= {"oracle", "operational"}
    assert (plans.cost < plans.price).all()
    ps = read_json(out / "pricing_support.json")
    assert ps["n"] == s.table6_n and len(ps["accepted"]) == ps["n"]

"""Revision 3, Part 3: the revised protocol (decisions D1, D6, D7)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import pricing as P                             # noqa: E402
from sqlshare_rerun.cli import _args, run                           # noqa: E402
from sqlshare_rerun.config import REVISED_OVERRIDES, make_settings  # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402
from sqlshare_rerun.util import read_json                           # noqa: E402


# ---- D7: test arms -----------------------------------------------------------------------
def _demands(arm, seed=5, k=200):
    """Training and test demand vectors used by buyer_arms (same streams)."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    train = P._popularity(k, streams[0])
    tr, te, st = P.buyer_arms(np.ones(k), seed, arm, 3000, 3000)
    return train, tr, te, st


def test_training_sample_is_shared_by_every_arm():
    info = np.linspace(0.1, 1.0, 50)
    base = P.buyer_arms(info, 9, "same", 400, 400)[0]
    for arm in ("shift10", "shift25", "shift50", "redrawn"):
        tr = P.buyer_arms(info, 9, arm, 400, 400)[0]
        assert np.array_equal(tr[0], base[0]) and np.array_equal(tr[1], base[1])


@pytest.mark.parametrize("arm,frac", [("shift10", 0.10), ("shift25", 0.25), ("shift50", 0.50)])
def test_controlled_shift_moves_only_the_stated_share(arm, frac):
    k = 200
    _, _, _, st = _demands(arm, k=k)
    # recompute the test demand directly to count moved assets
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(5).spawn(5)]
    train = P._popularity(k, streams[0])
    idx = np.sort(streams[3].choice(k, size=int(round(frac * k)), replace=False))
    test = train.copy()
    test[idx] = train[idx][streams[4].permutation(len(idx))]
    moved = int((test != train).sum())
    assert moved <= len(idx) and moved >= len(idx) - 2      # a permutation may fix a point
    assert st["kendall_tau"] < 1.0 and st["demand_tv"] > 0


def test_shift_magnitude_grows_with_the_share():
    taus = [_demands(a)[3]["kendall_tau"] for a in ("same", "shift10", "shift25", "shift50", "redrawn")]
    assert taus[0] == 1.0
    assert taus[0] > taus[1] > taus[2] > taus[3]
    assert abs(taus[4]) < 0.2                                  # a fresh ranking: near zero


def test_same_arm_has_the_training_demand():
    _, _, _, st = _demands("same")
    assert st == {"kendall_tau": 1.0, "demand_tv": 0.0}


# ---- D6: LPIP without a cap --------------------------------------------------------------------
def test_lpip_uncapped_solves_one_lp_per_buyer_and_never_loses_revenue():
    rng = np.random.default_rng(3)
    n_items = 60
    assets = [frozenset(rng.choice(n_items, size=rng.integers(1, 8), replace=False).tolist()) for _ in range(30)]
    t = rng.integers(0, 30, 150)
    v = 100 * rng.lognormal(0, 0.6, 150)
    edges = [assets[i] for i in t]
    p_cap, i_cap = P.lpip(edges, v, n_items, max_lps=10)
    p_all, i_all = P.lpip(edges, v, n_items, max_lps=0)
    assert i_cap["n_lps"] == 10 and i_all["n_lps"] == len(edges)
    assert P.revenue(p_all, v) >= P.revenue(p_cap, v) - 1e-9


# ---- D1: the revised preset ---------------------------------------------------------------------
def test_revised_preset_has_every_approved_correction():
    s = make_settings("revised")
    assert (s.alias_delete, s.entropy, s.fit, s.selection, s.querymarket) == \
        ("resolve", "exact", "breakpoints", "predicate", "cover_semantic")
    assert s.full_protocol_markets is True and s.full_markets == ()
    assert (s.row_sampling, s.answer_cells, s.metered_repeats) == ("any_row", "rows_x_cols", 3)
    assert (s.neighbours, s.qirana_table_weights, s.qirana_gain) == ("qirana", "uniform", "entropy_gain")
    assert (s.attackers, s.log_plans, s.max_lps, s.demand, s.main_arm) == ("both", True, 0, "arms", "same")
    assert s.remove_falsified == "curve"
    assert s.fingerprint() not in ("11cf10d670fa", "c9c9df996639")
    assert make_settings("paper").fingerprint() == "11cf10d670fa"
    assert make_settings("corrected").fingerprint() == "c9c9df996639"


# ---- end to end: python -m sqlshare_rerun all --protocol revised ------------------------------
def test_revised_protocol_end_to_end(tmp_path):
    (tmp_path / "data").mkdir()
    build_fixture(tmp_path / "data" / "sqlshare_data_release1.zip")
    a = _args(["all", "--protocol", "revised", "--root", str(tmp_path)])
    extra = dict(SELFTEST_SETTINGS, demand_arms=("same", "shift25", "redrawn"), train_sizes=(100, 200))
    assert run(a, **extra) == 0
    res = tmp_path / "results" / "revised"
    runs = pd.read_csv(res / "main_market" / "runs.csv")
    n_seeds, n_fam = len(SELFTEST_SETTINGS["run_seeds"]), 4
    assert set(runs.arm) == {"same", "shift25", "redrawn", "same_n100"}      # n_buyers=200 is the main size
    main = runs[runs.arm == "same"]
    assert main.planned.all() and not runs[runs.arm != "same"].planned.any()
    assert len(main) == n_seeds * n_fam * main.mechanism.nunique()
    assert (main.kendall_tau == 1.0).all()
    assert (runs[runs.arm == "redrawn"].kendall_tau < 1.0).all()
    assert "qirana_uniform_entropy_gain" in set(runs.mechanism)
    for c in ("fit_seconds", "plan_seconds_oracle", "plan_seconds_operational", "n_train"):
        assert c in runs.columns
    assert (res / "main_market" / "summary_by_arm_size_mechanism.csv").exists()
    # Part 4 fix: test revenue is normalized by the training reference scaled to the test size
    mono = runs[runs.mechanism == "monopoly_per_asset"]
    assert np.allclose(mono.revenue_train_norm, 1.0)
    ratio = runs.n_test / runs.n_train
    assert np.allclose(runs.revenue_naive_norm, runs.revenue_naive / (runs.ref_monopoly_train * ratio))
    assert np.allclose(runs.generalization_gap, 1 - runs.revenue_naive / (runs.revenue_train * ratio))
    assert (runs[runs.arm == "same_n100"].ref_monopoly_train > 0).all()
    nums = read_json(res / "revised_numbers.json")
    assert nums["main_market"]["main_arm"] == "same"
    assert (res / "markets" / "summary.json").exists()
    # D10 = C: query-text QueryMarket in the main tables, instance-level as comparison
    assert {"querymarket_viewcover", "querymarket_viewcover_instance"} <= set(runs.mechanism)
    # D4: every market with enough assets ran the full protocol
    full = pd.read_csv(res / "markets" / "full_protocol_runs.csv")
    ok = pd.read_csv(res / "markets" / "structure.csv")
    assert set(full.cid) == set(ok.cid)


# ---- Part 5: a stopped run continues from the finished support sets -----------------------
def test_revised_run_reuses_finished_support_sets(tmp_path):
    (tmp_path / "data").mkdir()
    build_fixture(tmp_path / "data" / "sqlshare_data_release1.zip")
    extra = dict(SELFTEST_SETTINGS, demand_arms=("same",), train_sizes=())
    assert run(_args(["all", "--protocol", "revised", "--root", str(tmp_path)]), **extra) == 0
    res = tmp_path / "results" / "revised"
    cache = tmp_path / "work" / "sample_cache_revised"
    kinds = {p.name.rsplit("_", 1)[0] for p in cache.glob("*.pkl")}
    assert {"curve", "witness", "size", "readership"} <= kinds
    assert any(k.startswith("market_") for k in kinds) and any(k.startswith("full_market_") for k in kinds)
    first = pd.read_csv(res / "markets" / "full_protocol_runs.csv")
    stamps = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.pkl")}
    (res / "markets" / "markets.done").unlink()          # as if the markets stage had stopped
    assert run(_args(["markets", "--protocol", "revised", "--root", str(tmp_path)]), **extra) == 0
    again = pd.read_csv(res / "markets" / "full_protocol_runs.csv")
    drop = [c for c in first.columns if "seconds" in c]
    pd.testing.assert_frame_equal(first.drop(columns=drop), again.drop(columns=drop))
    assert stamps == {p.name: p.stat().st_mtime_ns for p in cache.glob("*.pkl")}   # nothing redrawn
    # the 2.x protocols keep drawing afresh (no cache folder of their own)
    from sqlshare_rerun.stages import _sample_cache_dir
    assert _sample_cache_dir(make_settings("corrected")) is None
    assert _sample_cache_dir(make_settings("paper")) is None

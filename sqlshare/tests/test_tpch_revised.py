"""Revision 3, Part 4: the revised protocol on TPC-H (tpch/modules/revised_tpch.py).

The tests use small synthetic lineitem tables with the TPC-H column names and types, so
they need neither the DuckDB tpch extension nor the 1 GB dataset.

1. Exact conflict sets: the change computed for every asset on a neighbour equals a
   re-execution of the asset's SQL on the changed database, and two neighbours share a
   fingerprint exactly when the asset returns the same answer on both.
2. Qirana neighbours: keys never change, both update kinds occur, every neighbour
   changes the database, draws are reproducible.
3. Operational buyer: plans are built from asset definitions and prices only, hold on
   other database instances, are the cheapest guaranteed plans, and execute on purchased
   answers only.
4. End to end: tasks, columns, normalization, resume, settings guard; the 2.x TPC-H
   modules are byte-identical to the published artifact.
"""
import hashlib
import inspect
import itertools
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sqlshare"))
sys.path.insert(0, str(ROOT / "tpch" / "modules"))
import probe_tpch_lattice as L                                     # noqa: E402
import revised_tpch as R                                           # noqa: E402
from sqlshare_rerun import pricing as P                           # noqa: E402
from sqlshare_rerun.operational import PurchasedAnswers            # noqa: E402


def _db(n_rows=400, n_months=4, seed=0):
    con = duckdb.connect()
    R.synthetic_lineitem(con, n_rows=n_rows, n_months=n_months, seed=seed)
    return con


def _catalog(con, n=30):
    months = L.month_universe(con)
    A, _ = L.build_assets_n(months, n, seed=0)
    views = [L.Asset(t, frozenset(s["group_dims"]), frozenset({m}))
             for t, s in L.TEMPLATES.items() for m in months]
    return list(A) + [v for v in views if v not in set(A)]


def _exact(con, a):
    """Exact answer (Decimal values) of an asset, as a sorted tuple of rows."""
    return tuple(sorted(con.execute(L.asset_sql(a)).fetchall(), key=repr))


def _apply(con, changes):
    """Apply [(old, new)] row changes (new None = delete) inside the open transaction."""
    for old, new in changes:
        key = f"l_orderkey = {old['l_orderkey']} AND l_linenumber = {old['l_linenumber']}"   # keys never change
        rid = con.execute(f"SELECT min(rowid) FROM lineitem WHERE {key}").fetchone()[0]
        if new is None:
            con.execute(f"DELETE FROM lineitem WHERE rowid = {rid}")
        else:
            cols = [c for c in new if new[c] != old[c]]
            if cols:
                sets = ", ".join(f"{c} = ?" for c in cols)
                con.execute(f"UPDATE lineitem SET {sets} WHERE rowid = {rid}", [new[c] for c in cols])


# ---- 1. exact conflict sets and fingerprints --------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conflict_sets_and_fingerprints_match_reexecution(seed):
    con = _db(n_rows=60, n_months=3, seed=seed)       # small: many cells hold one row
    assets = _catalog(con, 20)
    base = [_exact(con, a) for a in assets]
    eng = R.DeltaEngine(assets, R.base_counts(con))
    gen = R.LineitemNeighbours(con, "lineitem", 0.5)
    rng = np.random.default_rng(seed)
    cols = gen.cols
    specs = []
    while len(specs) < 60:
        ch = gen.changes(gen.draw(rng))
        if ch is not None:
            specs.append(ch)
    for r in rng.choice(gen.count, size=20, replace=False):
        old = dict(zip(cols, con.execute(f"SELECT * FROM lineitem WHERE rowid = {int(r)}").fetchone()))
        specs.append([(old, None)])
    answers, states = [], set()
    for ch in specs:
        rows = [(R.make_row(o), None if n is None else R.make_row(n)) for o, n in ch]
        fp = eng.neighbour(rows)
        for ai in range(len(assets)):
            raw = eng.asset_change(ai, rows)
            if raw:
                states |= {e[1][0] for e in raw}
        con.execute("BEGIN TRANSACTION")
        _apply(con, ch)
        got = [_exact(con, a) for a in assets]
        con.execute("ROLLBACK")
        changed = {ai for ai in range(len(assets)) if got[ai] != base[ai]}
        assert changed == set(fp), (ch, changed ^ set(fp))
        answers.append((fp, got))
    # equal fingerprints <=> equal answers, per asset, over the neighbours that change it
    for ai in range(len(assets)):
        seen = [(fp[ai], got[ai]) for fp, got in answers if ai in fp]
        for (f1, g1), (f2, g2) in itertools.combinations(seen, 2):
            assert (f1 == f2) == (g1 == g2)
    assert "val" in states


def test_a_group_that_loses_its_last_row_disappears():
    con = _db(n_rows=60, n_months=3, seed=0)
    assets = _catalog(con, 20)
    bc = R.base_counts(con)
    (m, rf, ls), _ = next((k, c) for k, c in sorted(bc["q1"].items(), key=repr) if c == 1)
    cols = [d[0] for d in con.execute("SELECT * FROM lineitem LIMIT 0").description]
    old = dict(zip(cols, con.execute(
        "SELECT * FROM lineitem WHERE CAST(date_trunc('month', l_shipdate) AS DATE) = ? "
        "AND l_returnflag = ? AND l_linestatus = ?", [m, rf, ls]).fetchone()))
    eng = R.DeltaEngine(assets, bc)
    finest = next(i for i, a in enumerate(assets)
                  if a.template == "q1" and a.group == frozenset(L.TEMPLATES["q1"]["group_dims"]) and m in a.months)
    raw = eng.asset_change(finest, [(R.make_row(old), None)])
    assert raw is not None and raw[0][1] == ("gone",)
    base = _exact(con, assets[finest])
    con.execute("BEGIN TRANSACTION")
    _apply(con, [(old, None)])
    after = _exact(con, assets[finest])
    con.execute("ROLLBACK")
    assert len(after) == len(base) - 1


def test_changes_to_unread_columns_change_no_asset():
    con = _db()
    assets = _catalog(con)
    eng = R.DeltaEngine(assets, R.base_counts(con))
    old = dict(zip([d[0] for d in con.execute("SELECT * FROM lineitem LIMIT 0").description],
                   con.execute("SELECT * FROM lineitem WHERE rowid = 5").fetchone()))
    new = dict(old, l_comment="zzz", l_tax=old["l_tax"], l_shipmode="RAIL")
    assert eng.neighbour([(R.make_row(old), R.make_row(new))]) == {}


def test_answer_partition_gives_shannon_prices():
    con = _db()
    assets = _catalog(con)
    s = R.TpchSettings(support_n=50, expect_catalog=())
    edges, fps, info = R.build_support(con, assets, s)
    q = P.qirana_prices(edges, info["n"], changed=fps)
    for ai in range(len(assets)):
        assert q["qirana_shannon"][ai] == pytest.approx(P.exact_entropy(fps[ai], info["n"]))
        e = len(edges[ai])
        want = np.log(e) / np.log(info["n"]) if e > 1 else 0.0
        assert q["qirana_uniform_entropy_gain"][ai] == pytest.approx(want)
        assert q["qirana_weighted_coverage"][ai] == e


# ---- 2. Qirana neighbours ------------------------------------------------------------------
def test_qirana_neighbours_follow_the_published_rules():
    con = _db()
    gen = R.LineitemNeighbours(con, "lineitem", 0.5)
    assert "l_orderkey" not in gen.eligible and "l_linenumber" not in gen.eligible
    assert len(gen.eligible) == len(gen.cols) - 2
    s = R.TpchSettings(support_n=80, expect_catalog=())
    assets = _catalog(con)
    e1, f1, i1 = R.build_support(con, assets, s)
    e2, f2, i2 = R.build_support(con, assets, s)
    assert i1["accepted"] == i2["accepted"] and e1 == e2 and f1 == f2       # reproducible
    assert i1["kinds"]["row"] > 0 and i1["kinds"]["swap"] > 0
    assert i1["n"] == 80
    for spec in i1["accepted"]:
        assert spec[0] == "lineitem"
        assert gen.changes(tuple(spec[1:])) is not None                      # D' != D
        cols = spec[3] if spec[1] == "row" else spec[4]
        assert len(cols) >= 1


def test_deletion_support_is_available_for_comparison():
    con = _db()
    s = R.TpchSettings(support_n=40, neighbours="delete", expect_catalog=())
    edges, fps, info = R.build_support(con, _catalog(con), s)
    assert info["kinds"]["delete"] == 40 and info["n"] == 40


# ---- 3. operational buyer ------------------------------------------------------------------
def test_planner_never_sees_answers():
    params = set(inspect.signature(R.plan_target).parameters)
    assert params == {"ti", "t_months", "cands", "prices", "time_limit"}
    assert set(inspect.signature(R.op_structures).parameters) == {"assets"}
    held = PurchasedAnswers({0: pd.DataFrame(), 1: pd.DataFrame()}, [0])
    with pytest.raises(KeyError):
        held[1]


def test_single_source_plans_are_the_schema_pairs():
    con = _db()
    A = _catalog(con)
    st = R.op_structures(A)
    single = {(si, ti) for ti, c in st.items() for si, r, ms in c if ms == A[ti].months}
    schema = {(i, j) for i in range(len(A)) for j in range(len(A)) if i != j and L.determines(A[i], A[j])}
    assert single == schema


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_operational_plans_hold_on_other_instances_and_are_cheapest(seed):
    con1 = _db(n_rows=300, n_months=6, seed=0)
    A = _catalog(con1, 40)
    st = R.op_structures(A)
    rng = np.random.default_rng(seed)
    prices = rng.lognormal(0, 1, len(A))
    con2 = _db(n_rows=250, n_months=6, seed=100 + seed)               # another instance
    ans2 = {i: L.evaluate(con2, a) for i, a in enumerate(A)}
    n_plans = 0
    for ti in range(len(A)):
        got = R.plan_target(ti, A[ti].months, st[ti], prices)
        cheap = [c for c in st[ti] if prices[c[0]] < prices[ti] - 1e-9]
        if len(cheap) <= 10:                                          # brute force check
            best = np.inf
            for k in range(1, len(cheap) + 1):
                for sub in itertools.combinations(cheap, k):
                    whole = [ms for _, r, ms in sub if not r]
                    if any(a & b for a, b in itertools.combinations(whole, 2)):
                        continue
                    if set().union(*[ms for _, _, ms in sub]) != set(A[ti].months):
                        continue
                    best = min(best, sum(prices[si] for si, _, _ in sub))
            if best < prices[ti] - 1e-9:
                assert got is not None and got[0] == pytest.approx(best)
            else:
                assert got is None
        if got is None:
            continue
        n_plans += 1
        cost, plan = got
        assert cost < prices[ti]
        used = [m for _, _, ms in plan for m in ms]
        assert sorted(used) == sorted(A[ti].months)                   # every month once
        rebuilt = R.execute_plan(A[ti], plan, A, PurchasedAnswers(ans2, [si for si, _, _ in plan]))
        assert R.answers_equal(rebuilt, ans2[ti])
    assert n_plans > 0


# ---- 4. end to end --------------------------------------------------------------------------
SMALL = dict(n_assets=30, support_n=50, n_buyers=100, seeds=(0, 1), train_sizes=(50, 100),
             demand_arms=("same", "shift25", "redrawn"), expect_catalog=())


def test_revised_tpch_end_to_end_and_resume(tmp_path):
    con = _db(n_rows=1500, n_months=12)
    s = R.TpchSettings(**SMALL)
    out = tmp_path / "rev"
    runs = R.run_revised(con, str(out), s, workers=1, sf_is_one=False, log=lambda *_: None)
    n_fam, n_seed = len(R.FAMILIES), len(s.seeds)
    assert set(runs.arm) == {"same", "shift25", "redrawn", "same_n50"}
    assert len(runs) == 4 * n_fam * n_seed * runs.mechanism.nunique()
    mechs = set(runs.mechanism)
    assert {"qirana_uniform_entropy_gain", "qirana_shannon", "chawla_lpip", "querymarket_viewcover"} <= mechs
    assert "qirana_uniform_gain" not in mechs
    main = runs[runs.arm == "same"]
    assert main.planned.all() and not runs[runs.arm != "same"].planned.any()
    assert (main.plans_failed_op == 0).all() and main.plans_verified_op.sum() > 0
    assert (main.kendall_tau == 1.0).all() and (runs[runs.arm == "redrawn"].kendall_tau < 1).all()
    mono = runs[runs.mechanism == "monopoly_per_asset"]
    assert np.allclose(mono.revenue_train_norm, 1.0)
    ratio = runs.n_test / runs.n_train
    assert np.allclose(runs.revenue_naive_norm, runs.revenue_naive / (runs.ref_monopoly_train * ratio))
    for f in ("runs.csv", "plans_log.csv", "pricing_support.json", "build.json",
              "summary_by_arm_size_mechanism.csv", "summary_by_arm_size_family_mechanism.csv"):
        assert (out / f).exists()
    plans = pd.read_csv(out / "plans_log.csv")
    assert set(plans.attacker) == {"operational"} and plans.verified.all() and (plans.cost < plans.price).all()
    # resume: nothing is recomputed, the records are identical
    stamp = {p.name: p.stat().st_mtime_ns for p in (out / "tasks").glob("*.csv")}
    runs2 = R.run_revised(con, str(out), s, workers=1, sf_is_one=False, log=lambda *_: None)
    assert runs2.equals(runs)
    assert stamp == {p.name: p.stat().st_mtime_ns for p in (out / "tasks").glob("*.csv")}
    # a stopped run: delete two task files, rerun, same records
    for p in sorted((out / "tasks").glob("shift25__*.csv"))[:2]:
        p.unlink()
    runs3 = R.run_revised(con, str(out), s, workers=1, sf_is_one=False, log=lambda *_: None)
    pd.testing.assert_frame_equal(runs3.drop(columns=["fit_seconds"]), runs.drop(columns=["fit_seconds"]))
    # other settings may not reuse the folder
    with pytest.raises(RuntimeError):
        R.run_revised(con, str(out), R.TpchSettings(**dict(SMALL, support_n=51)), workers=1,
                      sf_is_one=False, log=lambda *_: None)


def test_answer_cells_count_rows_times_columns(tmp_path):
    con = _db(n_rows=800, n_months=6)
    s = R.TpchSettings(**SMALL)
    market, _, _ = R.build_market(con, s, sf_is_one=False, log=lambda *_: None)
    cells = np.array([len(market["answers"][i]) * len(market["answers"][i].columns)
                      for i in range(len(market["assets"]))], float)
    assert np.allclose(market["info"]["answer_cells"], cells / cells.max())


def test_main_run_settings():
    s = R.TpchSettings()
    assert (s.neighbours, s.fit, s.max_lps, s.main_arm, s.metered_repeats) == ("qirana", "breakpoints", 0, "same", 3)
    assert s.demand_arms == ("same", "shift10", "shift25", "shift50", "redrawn")
    assert s.train_sizes == (100, 200, 400, 800, 1600) and s.n_buyers == 800 and s.support_n == 400


LEGACY_SHA = {   # tpch/modules of the published artifact (git 5ee6219)
    "ports.py": "0f93462dcb55f214794553a8e8ebbf008c22f908c20bdf60e1daaa8e144a1b48",
    "ports_chawla.py": "bcac8156453628f5691abff471e5eaf6374da1c54d231397dd07a216bed4a456",
    "run_full.py": "9a8b08d20b567c1cef5b815f70cc43903dc08cc90ae053ec936765169ea92cd5",
    "tpc_experiment.py": "ec333f0b402283c7b985d4d5389b9059a4d19108fc9e0d1e7509591edbc2719b",
    "tpc_substrate.py": "0ab365ad3ea96915b2094eb2e37582a9de9b75691923ed4ca46cc65c996c43ba",
    "probe_tpch_lattice.py": "5907cdd49cd4e682a72c2bf769030e1d79428194bddb6a5c86503fc1a4645057",
}


@pytest.mark.parametrize("name", sorted(LEGACY_SHA))
def test_legacy_tpch_modules_are_unchanged(name):
    data = (ROOT / "tpch" / "modules" / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == LEGACY_SHA[name]

"""Revision 3, Part 1: Qirana fidelity (supervisor comment 1, decisions D2 and D8).

Every expected value in the first group of tests is computed by hand in the comments, from
the formulas of Deep and Koutris (SIGMOD 2017), Section 2.3:

    weighted coverage     p_wc  = sum_{i in C} w_i
    uniform entropy gain  p_ueg = log|C| / log|S|
    Shannon entropy       p_H   = -sum_{B in P_Q} w_B log w_B   (w_i = 1/|S|)

The second group checks the arbitrage properties the paper states for each function, and
the third checks the row/swap-update neighbours against an independent re-execution.
"""
import itertools
import math
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import pricing as P                            # noqa: E402
from sqlshare_rerun.support import SupportSampler, base_fingerprints  # noqa: E402

THREADS = int(os.environ.get("SQLSHARE_TEST_THREADS", "4"))


# ---- 1. hand-computed prices -----------------------------------------------------------
def test_hand_example_prices():
    # |S| = 4 support databases D0..D3.
    # Q_a: conflict set {0, 1}; answers on D0 and D1 differ from the true answer and from
    #      each other.            partition: {D2, D3}, {D0}, {D1}
    # Q_b: conflict set {0, 1, 2}; D0 and D1 give the SAME changed answer.
    #                              partition: {D3}, {D0, D1}, {D2}
    # Q_c: conflict set {}       partition: {D0, D1, D2, D3}
    # Q_d: conflict set {3}      partition: {D0, D1, D2}, {D3}
    edges = [frozenset({0, 1}), frozenset({0, 1, 2}), frozenset(), frozenset({3})]
    changed = [{0: "x", 1: "y"}, {0: "u", 1: "u", 2: "v"}, {}, {3: "z"}]
    p = P.qirana_prices(edges, 4, changed)
    # weighted coverage with unit weights = |C|
    assert p["qirana_weighted_coverage"].tolist() == [2.0, 3.0, 0.0, 1.0]
    # uniform entropy gain = log|C| / log 4:  log2/log4 = 0.5, log3/log4, 0 (|C|=0), 0 (|C|=1)
    assert p["qirana_uniform_entropy_gain"] == pytest.approx([0.5, math.log(3) / math.log(4), 0.0, 0.0])
    # Shannon (log base 2), probabilities = block size / 4
    h = lambda *b: -sum(x / 4 * math.log2(x / 4) for x in b)
    assert p["qirana_shannon"] == pytest.approx([h(2, 1, 1), h(1, 2, 1), 0.0, h(3, 1)])
    assert p["qirana_shannon"][0] == pytest.approx(1.5)     # 0.5*1 + 0.25*2 + 0.25*2
    assert p["qirana_shannon"][1] == pytest.approx(1.5)


def test_weights_are_used_by_weighted_coverage():
    p = P.qirana_prices([frozenset({0, 2})], 3, weights=[0.5, 7.0, 2.0])
    assert p["qirana_weighted_coverage"][0] == pytest.approx(2.5)


def test_changed_answers_must_match_conflict_sets():
    with pytest.raises(ValueError):
        P.qirana_prices([frozenset({0, 1})], 4, [{0: "x"}])


def test_historical_uniform_gain_was_coverage_over_a_constant():
    """The 2.x 'uniform gain' was |C|/|S|: weighted coverage divided by |S|, so after the
    scale fit it priced exactly like weighted coverage (supervisor comment 1)."""
    edges = [frozenset(range(k)) for k in (0, 1, 3, 7)]
    old = P.qirana_from_edges(edges, 10)
    assert np.allclose(old["qirana_uniform_gain"] * 10, old["qirana_weighted_coverage"])
    new = P.qirana_prices(edges, 10)
    # the published uniform entropy gain is a different (concave) function of |C|
    assert not np.allclose(new["qirana_uniform_entropy_gain"] * 10, new["qirana_weighted_coverage"])


# ---- 2. arbitrage properties stated by Deep and Koutris ---------------------------------
def _random_sets(rng, n, m):
    return [frozenset(np.flatnonzero(rng.random(n) < rng.uniform(0.05, 0.6)).tolist()) for _ in range(m)]


@pytest.mark.parametrize("seed", range(5))
def test_weighted_coverage_has_no_information_or_bundle_arbitrage(seed):
    rng = np.random.default_rng(seed)
    n = 30
    w = rng.uniform(0.1, 2.0, n)
    sets = _random_sets(rng, n, 25)
    p = P.qirana_prices(sets, n, weights=w)["qirana_weighted_coverage"]
    for (i, a), (j, b) in itertools.permutations(enumerate(sets), 2):
        if a <= b:                                   # information order
            assert p[i] <= p[j] + 1e-12
        bundle = P.qirana_prices([a | b], n, weights=w)["qirana_weighted_coverage"][0]
        assert bundle <= p[i] + p[j] + 1e-12          # C(Q1,Q2) = C(Q1) u C(Q2)


@pytest.mark.parametrize("seed", range(5))
def test_uniform_entropy_gain_has_no_information_arbitrage(seed):
    rng = np.random.default_rng(seed)
    sets = _random_sets(rng, 40, 25)
    p = P.qirana_prices(sets, 40)["qirana_uniform_entropy_gain"]
    for (i, a), (j, b) in itertools.permutations(enumerate(sets), 2):
        if a <= b:
            assert p[i] <= p[j] + 1e-12


def test_uniform_entropy_gain_exhibits_bundle_arbitrage():
    """Deep and Koutris: uniform entropy gain 'has no information arbitrage, but it does
    exhibit bundle arbitrage'. Two queries with one conflicting database each cost 0, their
    bundle conflicts on two databases and costs log 2 / log |S| > 0."""
    n = 16
    q1, q2 = frozenset({3}), frozenset({9})
    p = P.qirana_prices([q1, q2, q1 | q2], n)["qirana_uniform_entropy_gain"]
    assert p[0] == 0.0 and p[1] == 0.0
    assert p[2] == pytest.approx(math.log(2) / math.log(n))
    assert p[2] > p[0] + p[1]


@pytest.mark.parametrize("seed", range(5))
def test_shannon_entropy_is_monotone_and_subadditive_on_answer_partitions(seed):
    """Joint answers refine both partitions: H(Q1) <= H(Q1,Q2) <= H(Q1) + H(Q2)."""
    rng = np.random.default_rng(seed)
    n = 24
    for _ in range(20):
        a1 = rng.integers(0, 4, n)                  # answer label of Q1 on each support db
        a2 = rng.integers(0, 3, n)
        a1[0] = a2[0] = 0                           # support db 0 plays the true answer
        def changed(lbl):
            base = lbl[0]
            return {i: int(v) for i, v in enumerate(lbl) if v != base}
        joint = a1 * 10 + a2
        ch = [changed(a1), changed(a2), changed(joint)]
        edges = [frozenset(c) for c in ch]
        h = P.qirana_prices(edges, n, ch)["qirana_shannon"]
        assert h[0] <= h[2] + 1e-12 and h[1] <= h[2] + 1e-12
        assert h[2] <= h[0] + h[1] + 1e-12


# ---- 3. Qirana neighbours: row and swap updates ------------------------------------------
@pytest.fixture()
def upd_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "u.duckdb"))
    con.execute('CREATE SCHEMA "s"')
    con.execute('CREATE TABLE "s"."emp" (id INTEGER PRIMARY KEY, dept VARCHAR, sal INTEGER)')
    con.execute("INSERT INTO \"s\".\"emp\" SELECT i, 'd' || (i % 4), 100 + (i * 37) % 50 FROM range(60) r(i)")
    con.execute('CREATE TABLE "s"."site" AS SELECT i % 7 AS site, i % 3 AS kind FROM range(90) r(i)')
    yield con
    con.close()


def _assets():
    return {1: ("s", 'SELECT * FROM "s"."emp"'),
            2: ("s", 'SELECT id FROM "s"."emp"'),
            3: ("s", 'SELECT dept, sum(sal) AS t FROM "s"."emp" GROUP BY dept'),
            4: ("s", 'SELECT site, count(*) AS c FROM "s"."site" GROUP BY site'),
            5: ("s", 'SELECT * FROM "s"."site"')}


def _sampler(con, threads):
    assets = _assets()
    readers = {("s", "emp"): [1, 2, 3], ("s", "site"): [4, 5]}
    fp = base_fingerprints(con, ["s"], assets)
    targets = {("s", "emp"): ("s", "emp"), ("s", "site"): ("s", "site")}
    counts = {("s", "emp"): 60, ("s", "site"): 90}
    return SupportSampler(con, ["s"], assets, readers, fp, targets, counts, "any_row", threads,
                          neighbours="qirana", swap_ratio=0.5), readers


def _snapshot(con):
    return (con.execute('SELECT * FROM "s"."emp" ORDER BY id').fetchall(),
            con.execute('SELECT * FROM "s"."site" ORDER BY rowid').fetchall())


def test_update_neighbours_are_deterministic_and_rolled_back(upd_db):
    con = upd_db
    before = _snapshot(con)
    res = []
    for th in sorted({1, THREADS}):
        smp, readers = _sampler(con, th)
        res.append(smp.sample(sorted(readers), [1.0, 1.0], 80, 3, 4, record_fp=True))
    assert res[0].accepted == res[1].accepted
    assert res[0].incidence == res[1].incidence
    assert res[0].changed_fp == res[1].changed_fp
    assert _snapshot(con) == before                     # every update was rolled back


def test_every_neighbour_changes_the_database_and_keys_are_never_changed(upd_db):
    con = upd_db
    smp, readers = _sampler(con, 2)
    r = smp.sample(sorted(readers), [1.0, 1.0], 80, 3, 4, record_fp=True)
    assert r.n == 80
    on_emp = [i for i, (b, _) in enumerate(r.accepted) if b == ("s", "emp")]
    on_site = [i for i, (b, _) in enumerate(r.accepted) if b == ("s", "site")]
    # SELECT * of the updated table changes on every neighbour of that table ...
    assert sorted(r.incidence.get(1, [])) == on_emp
    # ... a bag of rows can be unchanged by a swap, so check the whole-table query of site
    # through row updates only
    rows_site = [i for i in on_site if r.accepted[i][1][0] == "row"]
    assert set(rows_site) <= set(r.incidence.get(5, []))
    # the primary key id is never updated, so the id list never changes
    assert r.incidence.get(2, []) == []
    kinds = [spec[0] for _, spec in r.accepted]
    assert 0 < kinds.count("swap") < len(kinds)        # both kinds occur (ratio 1:1)


def test_update_conflict_sets_match_an_independent_reexecution(upd_db, tmp_path):
    """Rebuild every accepted neighbour with plain SQL on a fresh copy of the database and
    recompute every answer: the conflict sets and changed answers must be identical."""
    con = upd_db
    smp, readers = _sampler(con, 2)
    r = smp.sample(sorted(readers), [1.0, 1.0], 40, 8, 9, record_fp=True)
    assets = _assets()
    base = {k: con.execute(sql).fetchall() for k, (_, sql) in assets.items()}
    cols = {("s", "emp"): ["dept", "sal"], ("s", "site"): ["site", "kind"]}   # id is the key
    for idx, (b, spec) in enumerate(r.accepted):
        name = f'"{b[0]}"."{b[1]}"'
        con.execute("BEGIN TRANSACTION")
        try:
            if spec[0] == "row":
                _, row, chosen, donors = spec
                for j, d in zip(chosen, donors):
                    c = cols[b][j]
                    v = con.execute(f'SELECT v FROM (SELECT DISTINCT "{c}" AS v FROM {name} '
                                    f'ORDER BY v NULLS LAST) LIMIT 1 OFFSET {d}').fetchone()[0]
                    con.execute(f'UPDATE {name} SET "{c}" = ? WHERE rowid = {row}', [v])
            else:
                _, r1, r2, chosen = spec
                names = [cols[b][j] for j in chosen]
                sel = ", ".join(f'"{c}"' for c in names)
                v1 = con.execute(f"SELECT {sel} FROM {name} WHERE rowid = {r1}").fetchone()
                v2 = con.execute(f"SELECT {sel} FROM {name} WHERE rowid = {r2}").fetchone()
                for rid, vals in ((r1, v2), (r2, v1)):
                    sets = ", ".join(f'"{c}" = ?' for c in names)
                    con.execute(f"UPDATE {name} SET {sets} WHERE rowid = {rid}", list(vals))
            for k in readers[b]:
                now = con.execute(assets[k][1]).fetchall()
                changed = sorted(now) != sorted(base[k])
                assert (idx in r.incidence.get(k, [])) == changed, (idx, spec, k)
        finally:
            con.execute("ROLLBACK")


def test_uniform_table_draw_follows_the_weights(upd_db):
    con = upd_db
    smp, readers = _sampler(con, 2)
    r = smp.sample(sorted(readers), [1.0, 1.0], 200, 21, 22)
    share = sum(1 for b, _ in r.accepted if b == ("s", "emp")) / r.n
    assert 0.38 < share < 0.62                          # uniform over the two relations

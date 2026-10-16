"""Revision 3, Part 2: the operational buyer (supervisor comment 3, decision D3).

1. Parsing of the query fragment.
2. Guarantees hold on EVERY database instance, not only the current one: plans are built
   once from query texts and then checked on many random instances.
3. The information boundary: planning never sees an answer, and executing a plan sees
   only the purchased answers (the "precise demonstration" the supervisor asks for).
"""
import inspect
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import operational as O                        # noqa: E402
from sqlshare_rerun.lattice import frames_equal                    # noqa: E402
from sqlshare_rerun.planner import min_cost_row_cover               # noqa: E402

COLS = ["station", "depth", "temp", "kind"]


def _resolve(schema, name):
    return {("s", "obs"): ("s", "obs"), ("s", "obs_alias"): ("s", "obs")}.get((schema.lower(), name.lower()))


def _columns(base):
    return COLS


def _forms(queries):
    return {k: O.parse_query(k, q, _resolve, _columns) for k, q in queries.items()}


def _db(seed, n=60):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "station": rng.choice(["S1", "S2", "S3", None], n, p=[0.35, 0.3, 0.25, 0.1]),
        "depth": rng.integers(0, 40, n).astype(float),
        "temp": np.round(rng.normal(10, 3, n), 1),
        "kind": rng.choice(["a", "b"], n),
    })
    df.loc[rng.random(n) < 0.1, "depth"] = np.nan
    con = duckdb.connect()
    con.execute('CREATE SCHEMA s')
    con.register("df", df)
    con.execute('CREATE TABLE s.obs AS SELECT * FROM df')
    con.execute('CREATE VIEW s.obs_alias AS SELECT * FROM s.obs')
    return con


def _answers(con, queries):
    return {k: con.execute(q).df() for k, q in queries.items()}


# ---- 1. parsing ------------------------------------------------------------------------
def test_fragment_parsing():
    f = _forms({
        1: 'SELECT station, depth FROM "s"."obs" o WHERE o.depth >= 10 AND depth < 20',
        2: "SELECT DISTINCT station FROM s.obs WHERE station IN ('S1','S2')",
        3: "SELECT * FROM s.obs_alias WHERE depth BETWEEN 5 AND 9",
        4: "SELECT station, count(*) FROM s.obs GROUP BY station",
        5: "SELECT station FROM s.obs WHERE depth > 1 OR depth < 0",
        6: "SELECT station FROM s.obs LIMIT 5",
        7: "SELECT a.station FROM s.obs a JOIN s.obs b ON a.station = b.station",
        8: "SELECT station AS st FROM s.obs WHERE 10 < depth",
    })
    assert f[1].ok and f[1].box["depth"] == O.Constraint(10, True, 20, False)
    assert f[2].ok and f[2].distinct and f[2].box["station"].values == frozenset({"S1", "S2"})
    assert f[3].ok and f[3].table == ("s", "obs") and [b for _, b in f[3].out] == COLS
    assert not f[4].ok and not f[5].ok and not f[6].ok and not f[7].ok
    assert f[8].ok and f[8].out == (("st", "station"),) and f[8].box["depth"] == O.Constraint(lo=10, lo_inc=False)


# ---- 2. guarantees hold on every instance --------------------------------------------------
def _random_query(rng):
    cols = rng.sample(COLS, rng.randint(1, 4))
    preds = []
    if rng.random() < 0.7:
        a = rng.randint(0, 35)
        preds.append(rng.choice([f"depth >= {a}", f"depth < {a}", f"depth BETWEEN {a} AND {a + rng.randint(1, 15)}",
                                 f"depth > {a}", f"depth <= {a}"]))
    if rng.random() < 0.5:
        vals = rng.sample(["S1", "S2", "S3"], rng.randint(1, 2))
        preds.append("station IN (" + ", ".join(f"'{v}'" for v in vals) + ")" if len(vals) > 1
                     else f"station = '{vals[0]}'")
    if rng.random() < 0.3:
        preds.append(f"kind = '{rng.choice(['a', 'b'])}'")
    pcols = [c for c in COLS if any(p.startswith(c) for p in preds)]
    if rng.random() < 0.5:                      # many sources keep their predicate columns
        cols = list(dict.fromkeys(cols + pcols))
    distinct = "DISTINCT " if rng.random() < 0.5 else ""
    where = (" WHERE " + " AND ".join(preds)) if preds else ""
    return f"SELECT {distinct}{', '.join(cols)} FROM s.obs{where}"


@pytest.mark.parametrize("seed", range(4))
def test_guaranteed_plans_hold_on_every_instance(seed):
    rng = random.Random(seed)
    queries = {k: _random_query(rng) for k in range(60)}
    forms = _forms(queries)
    assert all(f.ok for f in forms.values())
    structures = {t: O.target_structure(forms[t], forms) for t in forms}
    plans = []                                   # every single source, plus covers
    for t, st in structures.items():
        if not st:
            continue
        contrib, n, filters, kind = st
        for s, idx in contrib.items():
            if len(idx) == n:
                plans.append((t, [s]))
        if kind == "set" and contrib:
            prices = {s: 1.0 + rng.random() for s in contrib}
            cost, chosen = min_cost_row_cover(contrib, n, prices)
            if chosen:
                plans.append((t, chosen))
    assert len(plans) > 20                       # the random catalog yields many plans
    for inst in range(6):                        # schema-level: every plan, every instance
        con = _db(1000 * seed + inst)
        ans = _answers(con, queries)
        for t, plan in plans:
            st = structures[t]
            got = O.execute_plan(forms[t], plan, forms, st[2], O.PurchasedAnswers(ans, plan))
            assert frames_equal(got, ans[t].reset_index(drop=True)), (queries[t], [queries[s] for s in plan])
        con.close()


def test_instance_coincidence_is_not_a_guarantee():
    """On a database where depth >= 10 happens to equal depth >= 5, the answers agree,
    but the query texts do not guarantee it, so the operational buyer does not use it."""
    queries = {1: "SELECT station, depth FROM s.obs WHERE depth >= 10",
               2: "SELECT station FROM s.obs WHERE depth >= 5"}
    forms = _forms(queries)
    contrib, n, _, _ = O.target_structure(forms[2], forms)
    assert 1 not in contrib or len(contrib[1]) < n


def test_union_cover_needs_null_and_other_cells():
    queries = {"t": "SELECT DISTINCT station, depth FROM s.obs",
               "a": "SELECT station, depth FROM s.obs WHERE depth < 20",
               "b": "SELECT station, depth FROM s.obs WHERE depth >= 20"}
    forms = _forms(queries)
    contrib, n, _, kind = O.target_structure(forms["t"], forms)
    covered = np.zeros(n, bool)
    for v in contrib.values():
        covered[v] = True
    assert not covered.all()                     # rows with NULL depth are in no source
    queries["c"] = "SELECT station, depth FROM s.obs"
    forms = _forms(queries)
    contrib, n, _, _ = O.target_structure(forms["t"], forms)
    assert len(contrib["c"]) == n


# ---- 3. information boundary -----------------------------------------------------------------
def test_planner_never_receives_answers():
    sig = inspect.signature(O.plan_operational)
    assert list(sig.parameters) == ["prices", "structures", "cover_fn"]
    src = inspect.getsource(O.plan_operational) + inspect.getsource(O.target_structure) \
        + inspect.getsource(O.parse_query)
    assert "answers" not in src.split('"""')[-1]   # no answer is read outside docstrings


def test_execution_sees_only_purchased_answers():
    con = _db(7)
    queries = {"t": "SELECT DISTINCT station FROM s.obs WHERE station IN ('S1','S2')",
               "a": "SELECT station, depth FROM s.obs WHERE station = 'S1'",
               "b": "SELECT DISTINCT station FROM s.obs WHERE station = 'S2'"}
    ans = _answers(con, queries)
    forms = _forms(queries)
    st = O.target_structure(forms["t"], forms)
    plans = O.plan_operational({"t": 10.0, "a": 1.0, "b": 2.0}, {"t": st}, min_cost_row_cover)
    cost, plan = plans["t"]
    assert sorted(plan) == ["a", "b"] and cost == 3.0
    held = O.PurchasedAnswers(ans, plan)
    with pytest.raises(KeyError):
        held["t"]                                # the target's answer is not held
    got = O.execute_plan(forms["t"], plan, forms, st[2], held)
    assert frames_equal(got, ans["t"])           # the verifier compares afterwards
    con.close()


def test_semantic_pairs_are_single_source_guarantees():
    queries = {"t": "SELECT station FROM s.obs WHERE depth > 10",
               "s": "SELECT station, depth FROM s.obs WHERE depth > 5",
               "u": "SELECT station FROM s.obs WHERE depth > 5"}
    forms = _forms(queries)
    st = {k: O.target_structure(forms[k], forms) for k in forms}
    pairs = set(O.semantic_pairs(st))
    assert ("s", "t") in pairs                   # s keeps depth, so t's filter can be applied
    assert ("u", "t") not in pairs               # u drops depth: no guarantee


# ---- Part 5: answers as the pipeline stores them (mixed-case names, norm_frame) ----------
MIXED = ["Station", "Depth ", "TEMP", "Kind"]          # names as uploaded (one trailing space)


def _mixed_db(seed, n=60):
    rng = np.random.default_rng(seed)
    depth = rng.integers(0, 40, n).astype(object)
    depth[rng.random(n) < 0.1] = None                     # an integer column with NULLs
    df = pd.DataFrame({"Station": rng.choice(["S1", "S2", "S3", None], n, p=[0.35, 0.3, 0.25, 0.1]),
                       "Depth ": depth, "TEMP": np.round(rng.normal(10, 3, n), 1),
                       "Kind": rng.choice(["a", "b"], n)})
    con = duckdb.connect()
    con.execute("CREATE SCHEMA s")
    con.register("df", df)
    con.execute('CREATE TABLE s.obs AS SELECT "Station", CAST("Depth " AS INTEGER) AS "Depth ", '
                '"TEMP", "Kind" FROM df')
    return con


def _mixed_query(rng):
    cols = rng.sample(['"Station"', '"Depth "', 'temp', 'KIND'], rng.randint(1, 4))
    preds = []
    if rng.random() < 0.7:
        a = rng.randint(0, 35)
        preds.append(rng.choice([f'"Depth " >= {a}', f'"depth " < {a}', f'"DEPTH " BETWEEN {a} AND {a + 9}']))
    if rng.random() < 0.5:
        preds.append(f"station = '{rng.choice(['S1', 'S2', 'S3'])}'")
    if rng.random() < 0.5:
        cols = list(dict.fromkeys(cols + ['"Depth "', '"Station"']))
    distinct = "DISTINCT " if rng.random() < 0.5 else ""
    return f"SELECT {distinct}{', '.join(cols)} FROM s.obs" + (" WHERE " + " AND ".join(preds) if preds else "")


@pytest.mark.parametrize("seed", range(3))
def test_plans_verify_on_stored_answers_with_mixed_case_names(seed):
    """The pipeline stores answers with stripped, lowercased column names (norm_frame);
    the query texts keep the uploaded names. Every guaranteed plan must still rebuild its
    target, on several instances, and the verifier must say so."""
    from sqlshare_rerun.database import norm_frame
    rng = random.Random(seed)
    queries = {k: _mixed_query(rng) for k in range(50)}
    forms = {k: O.parse_query(k, q, _resolve, lambda b: MIXED) for k, q in queries.items()}
    assert all(f.ok for f in forms.values()), [f.reason for f in forms.values() if not f.ok]
    structures = {t: O.target_structure(forms[t], forms) for t in forms}
    plans = []
    for t, st in structures.items():
        if not st:
            continue
        contrib, n, filters, kind = st
        plans += [(t, [s]) for s, idx in contrib.items() if len(idx) == n]
        if kind == "set" and contrib:
            cost, chosen = min_cost_row_cover(contrib, n, {s: 1.0 + rng.random() for s in contrib})
            if chosen:
                plans.append((t, chosen))
    assert len(plans) > 10
    assert any(structures[t][2].get(s) for t, pl in plans for s in pl)        # some plans filter
    for inst in range(4):
        con = _mixed_db(100 * seed + inst)
        ans = {k: norm_frame(con.execute(q).df()) for k, q in queries.items()}
        for t, plan in plans:
            got = O.execute_plan(forms[t], plan, forms, structures[t][2], O.PurchasedAnswers(ans, plan))
            ok, note = O.compare_rebuilt(got, ans[t])
            assert ok, (note, queries[t], [queries[s] for s in plan])
        con.close()


def test_verifier_rejects_a_wrong_rebuild_and_says_why():
    want = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert O.compare_rebuilt(want.copy(), want) == (True, "ok")
    assert O.compare_rebuilt(want.iloc[:2], want)[1].startswith("shape")
    assert O.compare_rebuilt(want.rename(columns={"a": "A"}), want)[1].startswith("columns")
    bad = want.copy()
    bad.loc[0, "a"] = 9
    assert O.compare_rebuilt(bad, want) == (False, "values differ")
    obj = want.astype({"a": object})
    assert O.compare_rebuilt(obj, want)[0]

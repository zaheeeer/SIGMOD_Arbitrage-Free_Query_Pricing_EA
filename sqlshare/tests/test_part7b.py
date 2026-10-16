"""Revision 3, Part 7b: the revised protocol isolates query errors (a failing query counts
as changed) and makes every LIMIT deterministic (decision B1, cid 2610 finding)."""
import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun.config import make_settings                    # noqa: E402
from sqlshare_rerun.database import tiebreak_sql                   # noqa: E402
from sqlshare_rerun.extract import _stored_rule                     # noqa: E402
from sqlshare_rerun.support import SupportSampler, base_fingerprints, fp_sql  # noqa: E402


def _con(order):
    con = duckdb.connect()
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t (g VARCHAR, v INTEGER)")
    rows = [("a", 5), ("b", 5), ("c", 5), ("d", 1), ("e", 9)]
    con.executemany("INSERT INTO s.t VALUES (?, ?)", [rows[i] for i in order])
    con.execute("SET search_path='s'")
    return con


@pytest.mark.parametrize("sql, status", [
    ("SELECT g FROM t", "unchanged"),
    ("SELECT g, v FROM t ORDER BY v DESC LIMIT 2", "tiebreak"),
    ("SELECT * FROM t LIMIT 3", "tiebreak"),
    ("SELECT g FROM t UNION ALL SELECT g FROM t ORDER BY 1 LIMIT 2", "tiebreak"),
    ("SELECT g FROM (SELECT g, v FROM t ORDER BY v LIMIT 2) AS q", "tiebreak"),
    ("SELECT * FROM missing_table LIMIT 2", None),
])
def test_tiebreak_status(sql, status):
    con = _con(range(5))
    new, st = tiebreak_sql(con, "s", sql)
    if status is None:
        assert st.startswith("failed") and new == sql      # the original text is kept
        return
    assert st == status
    if status == "unchanged":
        assert new == sql
    else:
        assert "ORDER BY" in new.upper()
        assert len(con.execute(new).fetchall()) == len(con.execute(sql).fetchall())


def test_tiebreak_gives_one_answer_whatever_the_row_order():
    """Rows a, b, c tie on v = 5; with the tie-break the top-2 answer is fixed."""
    q = "SELECT g, v FROM t ORDER BY v DESC LIMIT 2"
    fps = set()
    for order in ([0, 1, 2, 3, 4], [2, 1, 0, 4, 3], [4, 3, 2, 1, 0], [1, 4, 0, 2, 3]):
        con = _con(order)
        new, st = tiebreak_sql(con, "s", q)
        assert st == "tiebreak"
        fps.add(con.execute(fp_sql(new)).fetchone()[0])
        assert sorted(con.execute(new).fetchall()) == [("a", 5), ("e", 9)]
    assert len(fps) == 1


def test_revised_preset_has_the_part7b_flags():
    s = make_settings("revised")
    assert (s.isolate_errors, s.error_as_change, s.limit_tiebreak) == (True, True, True)
    assert s.fingerprint() != "77d30e728d2f"                  # the P3 run is not reused
    for proto, fp in (("paper", "11cf10d670fa"), ("corrected", "c9c9df996639")):
        p = make_settings(proto)
        assert (p.isolate_errors, p.error_as_change, p.limit_tiebreak) == (False, False, False)
        assert p.fingerprint() == fp


def test_answers_cache_is_separate_with_the_tiebreak(tmp_path):
    from sqlshare_rerun import stages as S
    a = make_settings("revised", tmp_path)
    b = make_settings("corrected", tmp_path)
    assert a.limit_tiebreak and not b.limit_tiebreak
    # same work folder, different file names
    assert a.work_dir == b.work_dir
    import inspect
    src = inspect.getsource(S._answers)
    assert '"_tb" if s.limit_tiebreak' in src


def _err_db():
    con = duckdb.connect()
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t (flag VARCHAR, x VARCHAR, y INTEGER)")
    rows = [("ok", str(i), i) for i in range(1, 9)] + [("bad", "NA", 100), ("bad", "NA", 101)]
    con.executemany("INSERT INTO s.t VALUES (?, ?, ?)", rows)
    assets = {1: ("s", "SELECT sum(CAST(x AS INTEGER)) AS v FROM t WHERE flag = 'ok'"),
              2: ("s", "SELECT * FROM t"),
              3: ("s", "SELECT DISTINCT flag FROM t")}
    b = ("s", "t")
    return con, assets, {b: [1, 2, 3]}, base_fingerprints(con, ["s"], assets), b


def test_errors_are_counted_per_support_database():
    con, assets, readers, base, b = _err_db()
    smp = SupportSampler(con, ["s"], assets, readers, base, {b: ("s", "t")}, {b: 10}, "any_row",
                         threads=1, neighbours="qirana", isolate_errors=True, error_as_change=True)
    res = smp.sample([b], [1.0], 60, 1, 2)
    assert res.errors, "some swaps move 'NA' into an ok row, so query 1 fails"
    assert all(0 <= i < res.n and n >= 1 for i, n in res.errors.items())
    for i in res.errors:                                  # the failing query is in the conflict set
        assert i in set(res.incidence.get(1, ()))
    plain = SupportSampler(con, ["s"], assets, readers, base, {b: ("s", "t")}, {b: 10}, "any_row",
                           threads=1, neighbours="qirana").sample([b], [1.0], 60, 1, 2)
    assert not plain.errors                               # nothing is counted without isolation


def test_stored_rule_follows_the_settings():
    class Smp:
        pass
    a = Smp()
    assert _stored_rule(a) == "stop_at_first_error"
    a.isolate_errors, a.error_as_change = True, False
    assert _stored_rule(a) == "isolated"
    a.error_as_change = True
    assert _stored_rule(a) == "isolated_errors_as_change"


def test_sample_cache_key_includes_the_query_texts_with_the_tiebreak(tmp_path):
    """A rewritten query text must never reuse a support set drawn with the old text."""
    from sqlshare_rerun import stages as S

    class Fake:
        neighbours, counts, targets = "delete", {"b": 3}, {"b": ("s", "t")}

        def __init__(self, sql):
            self.assets, self.calls = {1: ("s", sql)}, 0

        def sample(self, *a, **k):
            self.calls += 1
            return {"sql": self.assets[1][1]}

    s = make_settings("revised", tmp_path)
    args = ("prep", "main_pricing", {"b": [1]}, ["b"], [1.0], 5, 1, 2)
    f1, f2, f3 = Fake("SELECT * FROM t LIMIT 1"), Fake("SELECT * FROM t ORDER BY 1 LIMIT 1"), \
        Fake("SELECT * FROM t ORDER BY 1 LIMIT 1")
    assert S._sample_inner(s, f1, *args) == {"sql": "SELECT * FROM t LIMIT 1"}
    assert S._sample_inner(s, f2, *args) == {"sql": "SELECT * FROM t ORDER BY 1 LIMIT 1"}
    assert S._sample_inner(s, f3, *args) == {"sql": "SELECT * FROM t ORDER BY 1 LIMIT 1"}
    assert (f1.calls, f2.calls, f3.calls) == (1, 1, 0)      # the same text reuses the cache

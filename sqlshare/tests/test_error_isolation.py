"""Revision 3, Part 7: a query that fails on a support database aborts DuckDB's
transaction. Without isolation every later query of that table also failed and was counted
as unchanged (the cid 6969 finding); with isolation each later query is evaluated."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun.support import SupportSampler, base_fingerprints  # noqa: E402


def _db():
    con = duckdb.connect()
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t (flag VARCHAR, x VARCHAR, y INTEGER)")
    rows = [("ok", str(i), i) for i in range(1, 9)] + [("bad", "NA", 100), ("bad", "NA", 101)]
    con.executemany("INSERT INTO s.t VALUES (?, ?, ?)", rows)
    assets = {1: ("s", "SELECT sum(CAST(x AS INTEGER)) AS v FROM t WHERE flag = 'ok'"),  # fails if 'NA' moves to an ok row
              2: ("s", "SELECT * FROM t"),
              3: ("s", "SELECT DISTINCT flag FROM t")}
    b = ("s", "t")
    readers = {b: [1, 2, 3]}
    base = base_fingerprints(con, ["s"], assets)
    return con, assets, readers, base, b


def _sampler(con, assets, readers, base, b, **kw):
    return SupportSampler(con, ["s"], assets, readers, base, {b: ("s", "t")}, {b: 10}, "any_row",
                          threads=1, neighbours="qirana", **kw)


# swap the x values of row 0 ("ok", "1") and row 8 ("bad", "NA"): query 1 now fails
SPEC = ("swap", 0, 8, (1,))


def test_without_isolation_later_queries_are_lost():
    con, assets, readers, base, b = _db()
    smp = _sampler(con, assets, readers, base, b)
    smp.prepare_updates([b])
    got = smp._evaluate_update((b, SPEC, None, False))
    assert got == {}                           # query 1 fails, queries 2 and 3 never run


@pytest.mark.parametrize("as_change", [False, True])
def test_with_isolation_every_query_is_evaluated(as_change):
    con, assets, readers, base, b = _db()
    smp = _sampler(con, assets, readers, base, b, isolate_errors=True, error_as_change=as_change)
    smp.prepare_updates([b])
    got = smp._evaluate_update((b, SPEC, None, True))
    assert 2 in got                            # SELECT * changes
    assert 3 not in got                        # the set of flags does not change
    assert (1 in got) == as_change
    if as_change:
        assert got[1] == ("error", "ConversionException")


def test_diagnosis_reports_the_error_and_the_lost_queries():
    con, assets, readers, base, b = _db()
    smp = _sampler(con, assets, readers, base, b)
    det = smp.diagnose_neighbour(b, SPEC)
    status = {k: st for k, st, _ in det}
    assert status == {1: "error", 2: "ok", 3: "ok"}
    fp2 = next(v for k, st, v in det if k == 2)
    assert fp2 != base[2]
    # the database is unchanged afterwards
    assert base_fingerprints(con, ["s"], assets) == base


def test_isolated_sampling_keeps_the_accepted_draws():
    con, assets, readers, base, b = _db()
    plain = _sampler(con, assets, readers, base, b).sample([b], [1.0], 30, 1, 2)
    iso = _sampler(con, assets, readers, base, b, isolate_errors=True).sample([b], [1.0], 30, 1, 2)
    assert plain.accepted == iso.accepted      # same draws; only the conflict sets can grow
    for k in assets:
        assert set(plain.incidence.get(k, ())) <= set(iso.incidence.get(k, ()))
    traced = _sampler(con, assets, readers, base, b, trace_costs=True).sample([b], [1.0], 30, 1, 2)
    assert traced.accepted == plain.accepted and traced.incidence == plain.incidence
    assert len(traced.cost_trace) == traced.draws and all(nq <= 3 for _, _, nq in traced.cost_trace)


def test_frequency_donors_copy_values_from_rows():
    con, assets, readers, base, b = _db()
    smp = _sampler(con, assets, readers, base, b, donor_mode="frequency")
    res = smp.sample([b], [1.0], 40, 3, 4)
    assert res.n == 40
    rows = [sp for _, sp in res.accepted if sp[0] == "row"]
    assert rows and all(all(0 <= d < 10 for d in sp[3]) for sp in rows)

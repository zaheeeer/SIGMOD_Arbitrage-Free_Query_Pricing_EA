"""Revision 3, P6: the extraction reads finished support sets only and reports the three
relations and worked examples consistently (supervisor comments 3 and 4)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun.cli import _args, run                           # noqa: E402
from sqlshare_rerun.selftest import SELFTEST_SETTINGS, build_fixture  # noqa: E402
from sqlshare_rerun.util import read_json                           # noqa: E402

EXTRA = dict(SELFTEST_SETTINGS, demand_arms=("same",), train_sizes=())


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    root = tmp_path_factory.mktemp("ext")
    (root / "data").mkdir()
    build_fixture(root / "data" / "sqlshare_data_release1.zip")
    assert run(_args(["all", "--protocol", "revised", "--root", str(root)]), **EXTRA) == 0
    cache = root / "work" / "sample_cache_revised"
    before = sorted(p.name for p in cache.glob("*.pkl"))
    assert run(_args(["extract", "--protocol", "revised", "--root", str(root)]), **EXTRA) == 0
    after = sorted(p.name for p in cache.glob("*.pkl"))
    return root, before, after


def test_extraction_draws_nothing_new(extracted):
    _, before, after = extracted
    assert before == after


def test_relations_are_named_and_evaluated_separately(extracted):
    root, _, _ = extracted
    res = root / "results" / "revised"
    rel = pd.read_csv(res / "extract" / "relations.csv")
    lat = pd.read_csv(res / "lattice" / "pairs.csv")
    summ = read_json(res / "extract" / "extract_summary.json")
    cids = {m["cid"] for m in summ["markets"]}
    assert int(rel.lattice.sum()) == len(lat[lat.cid.isin(cids)])
    main = read_json(res / "main_market" / "summary.json")
    first = summ["markets"][0]
    assert first["query_text_pairs"] == main["operational"]["semantic_pairs"]
    # a pair the query texts guarantee holds on every instance, so no support database can
    # change the target without changing the source
    assert (rel[rel.query_text].refuting_pricing == 0).all()
    assert {"refuting_curve", "refuting_readership"} <= set(rel.columns)


def test_assets_carry_query_texts_and_scores(extracted):
    root, _, _ = extracted
    a = pd.read_csv(root / "results" / "revised" / "extract" / "assets.csv")
    assert a.sql.notna().all() and "score_qirana_weighted_coverage" in a
    assert "score_compute_metered" not in a
    assert (a.score_qirana_weighted_coverage == a.conflict_size_pricing).all()


def test_worked_examples_show_answers_that_match_the_conflict_sets(extracted):
    root, _, _ = extracted
    files = sorted((root / "results" / "revised" / "extract" / "examples").glob("example_*.json"))
    assert files, "the synthetic run has free-source plans to explain"
    for f in files:
        ex = json.loads(f.read_text(encoding="utf-8"))
        tgt = next(k for k, q in ex["queries"].items() if q["role"] == "target")
        for db in ex["support_databases"]:
            for k, q in ex["queries"].items():
                changed = db["answer_changed"][k]
                assert changed == (db["support_index"] in q["conflict_set"]), (f.name, k, db["support_index"])
            assert db["in_target_conflict_set"] == (db["support_index"] in ex["queries"][tgt]["conflict_set"])
            assert db["changes"]


def test_error_check_reproduces_the_stored_conflict_sets(extracted):
    root, _, _ = extracted
    before = sorted(p.name for p in (root / "work" / "sample_cache_revised").glob("*.pkl"))
    assert run(_args(["diagnose", "--protocol", "revised", "--root", str(root)]), **EXTRA) == 0
    after = sorted(p.name for p in (root / "work" / "sample_cache_revised").glob("*.pkl"))
    assert before == after
    df = pd.read_csv(root / "results" / "revised" / "extract" / "error_check.csv")
    assert len(df) > 0 and df.old_matches_cached.all()
    assert (df.changed_isolated >= df.changed_old).all()
    # Part 7b: the revised protocol stores isolated sets with errors counted as changed,
    # and with the LIMIT tie-break every stored set is reproduced
    assert set(df.stored_rule) == {"isolated_errors_as_change"}
    assert df.stored_matches_recomputed.all()


def test_summaries_report_query_errors_and_tiebreaks(extracted):
    root, _, _ = extracted
    res = root / "results" / "revised"
    for f in (res / "main_market" / "summary.json", res / "markets" / "summary.json"):
        summ = read_json(f)
        assert "query_errors" in summ and "tiebreak_queries" in summ
        for v in summ["query_errors"].values():
            assert v["with_query_errors"] <= v["support_databases"]
    main = read_json(res / "main_market" / "summary.json")
    mk = read_json(res / "markets" / "summary.json")
    tb = list(main["tiebreak_queries"].values()) + [x for d in mk["tiebreak_queries"].values() for x in d.values()]
    assert tb and all(v == "tiebreak" for v in tb)      # the fixture's TOP queries
    assert list((root / "work" / "answers").glob("*_tb.pkl"))

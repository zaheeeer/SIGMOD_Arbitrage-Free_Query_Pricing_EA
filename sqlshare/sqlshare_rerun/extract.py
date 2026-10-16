"""Revision 3, phase P6: extraction for the analysis of supervisor comments 3 and 4.

    python -m sqlshare_rerun extract --protocol revised

No support set is drawn: every support set is read from work/sample_cache_revised, which
the revised main-market and markets stages filled (a missing set is an error). Nothing in
results/revised/main_market or results/revised/markets is changed. Outputs go to
results/revised/extract/:

  assets.csv       one row per asset of every market with the full protocol: the query
                   text (DuckDB SQL), its owner, whether it is in the operational
                   fragment (and why not), answer size, conflict-set size on the pricing
                   support, and the price scores of the deterministic mechanisms
                   (compute-metered prices depend on timings and are left out).
  relations.csv    every ordered pair (source, target) that is (a) in the verified
                   lattice (Definition 3.2, instance-level derivability, with the rewrite
                   family) or (b) a query-text pair (one source guaranteed by the query
                   texts to rebuild the target on every instance). For each pair, the
                   number of support databases that change the target but not the source,
                   on the Qirana pricing support (all markets) and on the deletion curve
                   and readership supports (main market). A pair with such a database is
                   not a determinacy pair (comment 3: the relations are named and
                   evaluated separately).
  support_<cid>.json  the pricing support of each market: neighbour specs and incidence.
  examples/example_NN.json  worked examples (comment 4): target and source queries with
                   their SQL, answers on D, conflict sets, and, for a few support
                   databases, the changed rows and the answers on that database.
  extract_summary.json  counts and the settings fingerprint.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .util import LOG, read_json, write_csv, write_json

MECH_ORDER = ("qirana_weighted_coverage", "qirana_shannon", "querymarket_viewcover", "chawla_lpip",
              "chawla_cip", "chawla_xos", "chawla_uip", "chawla_layering", "qirana_uniform_entropy_gain",
              "querymarket_viewcover_instance", "monotone_constrained", "monopoly_per_asset")


def _frame_json(df: pd.DataFrame, max_rows: int) -> dict:
    # Part 7c: rows sorted before truncation, so the shown rows do not depend on the order
    # in which the database returned an unordered answer
    df = df.reset_index(drop=True)
    try:
        order = pd.DataFrame({i: df.iloc[:, i] for i in range(df.shape[1])})
        try:
            idx = order.sort_values(list(order.columns), kind="mergesort", na_position="last").index
        except TypeError:                       # mixed types in a column: sort by text
            idx = order.astype(str).sort_values(list(order.columns), kind="mergesort").index
        df = df.loc[idx].reset_index(drop=True)
    except Exception:                           # never fail an extraction over display order
        pass
    head = df.head(max_rows)
    return {"rows": int(len(df)), "columns": [str(c) for c in df.columns],
            "data": json.loads(head.to_json(orient="values", date_format="iso", default_handler=str)),
            "truncated": bool(len(df) > max_rows)}


def _market_inputs(s: Settings, S, con, cat, closure, assets, rank, cid, main: bool, seconds=None):
    """Answers, pairs and support sets of one market, exactly as the revised stages built
    them (support sets come from the cache). ``seconds`` (Part 7c) replaces the stored
    single timings in the compute-metered score (the main market's median of three)."""
    from .support import SupportSampler, base_fingerprints
    data = S._answers(s, con, assets, cid)
    answers, meta = data["answers"], data["meta"]
    keys = sorted(answers)
    ki = {k: i for i, k in enumerate(keys)}
    L = S._lattice_pairs(s, cid)
    fam = {(a, b): f for a, b, f in L["verified"] if a in ki and b in ki}
    pairs = [(ki[a], ki[b]) for (a, b) in fam]
    prep_key = read_json(s.work_dir / "prepare.done")["key"]
    asset_sql = {k: meta[k] for k in keys}
    base_fp = base_fingerprints(con, cat["schemas"], asset_sql)
    sup = S._support_setup(s, con, cat, keys, meta, closure, "size" if main else "readership")
    if not sup["tables"]:
        return None                     # no base table to sample from (no full protocol either)
    qsampler = S._market_sampler(s, con, cat["schemas"], asset_sql, sup, base_fp)
    supports = {}
    if main:
        dsampler = SupportSampler(con, cat["schemas"], asset_sql, sup["readers"], base_fp, sup["targets"],
                                  sup["counts"], s.row_sampling, s.n_threads(), s.pool_stall_s,
                                  isolate_errors=s.isolate_errors, error_as_change=s.error_as_change)
        n_curve = max(s.curve_checkpoints)
        supports["curve"] = (dsampler, S._sample(s, dsampler, prep_key, "curve", sup["readers"], sup["tables"],
                                                 sup["size_w"], n_curve, s.curve_seed, S._row_seed(s.curve_seed),
                                                 checkpoints=s.curve_checkpoints, label="curve"))
        supports["readership"] = (dsampler, S._sample(
            s, dsampler, prep_key, "readership", sup["readers"], sup["tables"], sup["read_w"], s.table6_n,
            s.table6_seed, S._row_seed(s.table6_seed + 1), record_fp=(s.entropy == "exact"),
            label="readership-weighted"))
        price = S._sample(s, qsampler, prep_key, f"qirana_{s.qirana_table_weights}", sup["readers"], sup["tables"],
                          S._qirana_weights(s, sup), s.table6_n, s.table6_seed, S._row_seed(s.table6_seed + 2),
                          record_fp=(s.entropy == "exact"), label=f"qirana-{s.qirana_table_weights}")
        sizes_curve = supports["curve"][1].counts(keys)
        clean = [p for p in pairs if not sizes_curve[p[1]] > sizes_curve[p[0]] + 1e-9]
        seconds = data["seconds"] if seconds is None else seconds
    else:
        wts = S._qirana_weights(s, sup) if s.neighbours == "qirana" else sup["read_w"]
        price = S._sample(s, qsampler, prep_key, f"full_market_{cid}", sup["readers"], sup["tables"], wts,
                          s.table6_n, s.table6_seed + rank, S._row_seed(s.table6_seed + rank),
                          record_fp=(s.entropy == "exact"), label=f"full market {rank}")
        sizes = price.counts(keys)
        clean = [p for p in pairs if not sizes[p[1]] > sizes[p[0]] + 1e-9]
        seconds = data["seconds"] if seconds is None else seconds
    supports["pricing"] = (qsampler, price)
    from .lattice import cover_structure
    contrib, nrows = cover_structure(answers, keys, s.cover_target_cap)
    op = S._op_setup(s, con, cat, keys, meta)
    prov = S._provenance(con, keys, closure)
    scores, _, rowsz = S._scores(s, keys, answers, seconds, prov, price.edges(keys), price.changed_fp,
                                 s.table6_n, clean, contrib, nrows, prov, op=op)
    return {"keys": keys, "ki": ki, "answers": answers, "meta": meta, "fam": fam, "op": op,
            "supports": supports, "scores": scores, "sup": sup, "clean": clean, "contrib": contrib,
            "nrows": nrows, "prov": prov, "rowsz": rowsz, "base_fp": base_fp, "asset_sql": asset_sql,
            "prep_key": prep_key}


def _relations(cid, m) -> list:
    keys, op = m["keys"], m["op"]
    qt = set()
    for t, st in op["structures"].items():
        if not st:
            continue
        contrib, n_cells, _, _ = st
        for sk, idx in contrib.items():
            if n_cells and len(idx) == n_cells:
                qt.add((sk, t))
    inc = {name: res.incidence for name, (_, res) in m["supports"].items()}
    rows = []
    for (a, b) in sorted(set(m["fam"]) | qt, key=lambda x: (str(x[0]), str(x[1]))):
        r = {"cid": cid, "src": a, "tgt": b, "lattice": (a, b) in m["fam"],
             "lattice_family": m["fam"].get((a, b), ""), "query_text": (a, b) in qt}
        for name, ic in inc.items():
            ct, cs = set(ic.get(b, ())), set(ic.get(a, ()))
            r[f"refuting_{name}"] = len(ct - cs)
        rows.append(r)
    return rows


def _pick_examples(plans: pd.DataFrame, max_examples: int) -> list:
    p = plans[(plans.verified) & (plans.gain >= 0.999)].copy()
    if p.empty:
        return []
    p["prio"] = p.mechanism.map({m: i for i, m in enumerate(MECH_ORDER)}).fillna(len(MECH_ORDER))
    out, seen = [], set()
    for (att, t, src), g in p.sort_values(["prio", "attacker"]).groupby(["attacker", "target", "sources"], sort=False):
        k = (str(t), str(src))
        if k in seen:
            continue
        seen.add(k)
        out.append({"target": t, "sources": [x for x in str(src).split(";")], "attacker": att,
                    "mechanisms": sorted(set(g.mechanism)), "plans": int(len(g)),
                    "prio": float(g.prio.min())})
    out.sort(key=lambda e: e["prio"])
    return out[:max_examples]


def _key_of(keys, raw):
    """Plan logs store keys as text; map back to the market's key objects."""
    by_text = {str(k): k for k in keys}
    return by_text[str(raw)]


def _example(cid, m, ex, max_rows, n_show=(3, 2, 1)) -> dict:
    keys = m["keys"]
    t = _key_of(keys, ex["target"])
    srcs = [_key_of(keys, x) for x in ex["sources"]]
    sampler, res = m["supports"]["pricing"]
    inc = res.incidence
    ct = set(inc.get(t, ()))
    cu = set().union(*[set(inc.get(x, ())) for x in srcs]) if srcs else set()
    show = (sorted(ct - cu)[:n_show[0]] + sorted(ct & cu)[:n_show[1]] + sorted(cu - ct)[:n_show[2]])
    qkeys = [t] + srcs
    sc = m["scores"]
    ki = m["ki"]
    out = {"cid": cid, "attacker": ex.get("attacker"), "mechanisms_with_gain_1": ex.get("mechanisms"),
           "plans_logged": ex.get("plans"),
           "queries": {str(k): {"role": "target" if k == t else "source", "owner": m["meta"][k][0],
                                "sql": m["meta"][k][1],
                                "in_fragment": bool(m["op"]["forms"][k].ok),
                                "fragment_reason": m["op"]["forms"][k].reason,
                                "conflict_set": sorted(map(int, inc.get(k, ()))),
                                "scores": {name: float(v[ki[k]]) for name, v in sc.items()
                                           if name != "compute_metered"}} for k in qkeys},
           "pairs": [{"src": str(x), "lattice_family": m["fam"].get((x, t), ""),
                      "refuting_pricing": len(ct - set(inc.get(x, ())))} for x in srcs],
           "support_n": int(res.n),
           "target_not_covered_by_sources": sorted(map(int, ct - cu))}
    # answers on D (the sampler's search path, as for the conflict sets)
    any_b = res.accepted[0][0] if res.accepted else None
    base = None
    if any_b is not None:
        base = sampler.neighbour_answers(any_b, None, qkeys, base=True)
        out["answers_on_D"] = {str(k): _frame_json(v, max_rows) for k, v in base["answers"].items()}
    from .lattice import frames_equal
    nb = []
    for i in show:
        b, spec = res.accepted[i]
        got = sampler.neighbour_answers(b, spec, qkeys)
        nb.append({"support_index": int(i), "table": list(got["table"]), "spec": json.loads(json.dumps(spec, default=str)),
                   "in_target_conflict_set": i in ct, "in_source_conflict_sets": [i in set(inc.get(x, ())) for x in srcs],
                   "changes": json.loads(json.dumps(got["changes"], default=str)),
                   # full answers compared in any row order (the shown rows may be truncated)
                   "answer_changed": {str(k): (not frames_equal(got["answers"][k].reset_index(drop=True),
                                                                base["answers"][k].reset_index(drop=True)))
                                      for k in qkeys} if base is not None else {},
                   "answers": {str(k): _frame_json(v, max_rows) for k, v in got["answers"].items()}})
    out["support_databases"] = nb
    return out


def stage_extract(s: Settings, examples: list | None = None, max_examples: int = 12, max_rows: int = 40) -> dict:
    from . import stages as S
    if s.protocol != "revised":
        raise ValueError("the extraction reads the revised protocol's support sets; use --protocol revised")
    t0 = time.perf_counter()
    out = s.results_dir / "extract"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    mm = s.results_dir / "main_market"
    if not (mm / "main.done").exists():
        raise RuntimeError("run the revised main market first (python -m sqlshare_rerun main --protocol revised)")
    s.extra["sample_cache_only"] = True
    corpus, sel, assets, cat = S._load_prepared(s)
    closure = S._Closure(corpus)
    ranks = S._market_ranks(assets, sel.selected)
    con = S._connect(s)
    asset_rows, rel_rows, summary = [], [], {"fingerprint": s.fingerprint(), "markets": []}
    main_m = None
    try:
        for rank, cid, surv in ranks:
            data = S._answers(s, con, assets, cid)
            if len(data["answers"]) < s.market_min_assets:
                continue
            if rank > 1 and not (s.full_protocol_markets or cid in set(s.full_markets)):
                continue
            m = _market_inputs(s, S, con, cat, closure, assets, rank, cid, main=(rank == 1))
            if m is None:
                continue
            if rank == 1:
                main_m = (cid, m)
            _, price = m["supports"]["pricing"]
            for k in m["keys"]:
                f = m["op"]["forms"][k]
                row = {"rank": rank, "cid": cid, "key": k, "owner": m["meta"][k][0], "sql": m["meta"][k][1],
                       "in_fragment": bool(f.ok), "fragment_reason": f.reason,
                       "answer_rows": int(len(m["answers"][k])), "answer_cols": int(m["answers"][k].shape[1]),
                       "conflict_size_pricing": len(price.incidence.get(k, ()))}
                for name, v in m["scores"].items():
                    if name != "compute_metered":
                        row[f"score_{name}"] = float(v[m["ki"][k]])
                asset_rows.append(row)
            rel = _relations(cid, m)
            rel_rows += rel
            write_json({"n": price.n, "accepted": [[list(b), list(sp) if isinstance(sp, tuple) else sp]
                                                   for b, sp in price.accepted],
                        "incidence": {str(k): list(map(int, v)) for k, v in price.incidence.items()}},
                       out / f"support_{cid}.json")
            r = pd.DataFrame(rel)
            summary["markets"].append({
                "rank": rank, "cid": cid, "assets": len(m["keys"]), "lattice_pairs": int(r.lattice.sum()) if len(r) else 0,
                "query_text_pairs": int(r.query_text.sum()) if len(r) else 0,
                "both": int((r.lattice & r.query_text).sum()) if len(r) else 0,
                "lattice_refuted_by_pricing": int((r.lattice & (r.refuting_pricing > 0)).sum()) if len(r) else 0,
                "query_text_refuted_by_pricing": int((r.query_text & (r.refuting_pricing > 0)).sum()) if len(r) else 0})
            LOG.info("extract: market rank %d (cid %d): %d assets, %d relation rows", rank, cid, len(m["keys"]), len(rel))
        write_csv(pd.DataFrame(asset_rows), out / "assets.csv")
        write_csv(pd.DataFrame(rel_rows), out / "relations.csv")
        if main_m is not None:
            cid, m = main_m
            if examples is None:
                plans = pd.read_csv(mm / "plans_log.csv")
                plans = plans[plans.arm == s.main_arm] if "arm" in plans else plans
                examples = _pick_examples(plans, max_examples)
            for i, ex in enumerate(examples, 1):
                write_json(_example(cid, m, ex, max_rows), out / "examples" / f"example_{i:02d}.json")
            summary["examples"] = [{k: v for k, v in e.items() if k != "prio"} for e in examples]
    finally:
        con.close()
        s.extra.pop("sample_cache_only", None)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(summary, out / "extract_summary.json")
    LOG.info("extract: %s (%.0f s)", out, summary["seconds"])
    return summary


# ---------------------------------------------------------------------------------------
# Part 7 check: query errors on support databases (cid 6969 finding)
# ---------------------------------------------------------------------------------------
def _stored_rule(sampler) -> str:
    """Which recomputed set the stored conflict sets should equal, given the sampler settings."""
    if getattr(sampler, "isolate_errors", False):
        return "isolated_errors_as_change" if getattr(sampler, "error_as_change", False) else "isolated"
    return "stop_at_first_error"


def _diagnose_support(cid, name, sampler, res, picks, keys_with_base):
    rows = []
    rule = _stored_rule(sampler)
    inc = res.incidence
    for i in picks:
        b, spec = res.accepted[i]
        det = sampler.diagnose_neighbour(b, spec)
        first_err = next((j for j, (_, st, _) in enumerate(det) if st == "error"), None)
        old, iso, err = set(), set(), set()
        for j, (k, st, v) in enumerate(det):
            if st == "error":
                err.add(k)
                continue
            if st != "ok":
                continue
            changed = v != sampler.base_fp.get(k)
            if changed:
                iso.add(k)
                if first_err is None or j < first_err:
                    old.add(k)
        cached = {k for k in keys_with_base if i in set(inc.get(k, ()))}
        expect = {"stop_at_first_error": old, "isolated": iso,
                  "isolated_errors_as_change": iso | (err & set(keys_with_base))}[rule]
        later_ok = sum(1 for j, (_, st, _) in enumerate(det) if first_err is not None and j > first_err and st == "ok")
        rows.append({"cid": cid, "support": name, "index": int(i), "table": "/".join(map(str, b)),
                     "readers": len(det), "errors": len(err),
                     "first_error_position": -1 if first_err is None else int(first_err),
                     "queries_lost_old": int(later_ok),
                     "changed_old": len(old), "changed_isolated": len(iso),
                     "changed_isolated_errors_as_change": len(iso | err),
                     "old_matches_cached": old == cached,
                     "stored_rule": rule,
                     "stored_matches_recomputed": expect == cached,
                     "new_changes_found": len(iso - old),
                     "error_examples": " | ".join(sorted({v for _, st, v in det if st == "error"})[:3])})
    return rows


def stage_diagnose(s: Settings, per_market: int = 40, seed: int = 0) -> dict:
    """Re-evaluate support databases with every query error isolated, and compare with the
    stored conflict sets. Before Part 7b the stored sets stop at the first error; with the
    Part 7b settings (isolate_errors, error_as_change) they should equal the isolated sets
    with failing queries counted as changed. Column stored_matches_recomputed checks the
    rule that matches the settings; a mismatch means a query answer is not reproducible.
    Draws no new support set. Writes results/<protocol>/extract/error_check.csv."""
    from . import stages as S
    t0 = time.perf_counter()
    out = s.results_dir / "extract"
    out.mkdir(parents=True, exist_ok=True)
    s.extra["sample_cache_only"] = True
    corpus, sel, assets, cat = S._load_prepared(s)
    closure = S._Closure(corpus)
    ranks = S._market_ranks(assets, sel.selected)
    con = S._connect(s)
    rows = []
    rng = np.random.default_rng(seed)
    try:
        for rank, cid, surv in ranks:
            data = S._answers(s, con, assets, cid)
            if len(data["answers"]) < s.market_min_assets:
                continue
            if rank > 1 and not (s.full_protocol_markets or cid in set(s.full_markets)):
                continue
            m = _market_inputs(s, S, con, cat, closure, assets, rank, cid, main=(rank == 1))
            if m is None:
                continue
            refuting = set()
            qsampler, price = m["supports"]["pricing"]
            for r in _relations(cid, m):
                if r["query_text"] and r["refuting_pricing"] > 0:
                    ct = set(price.incidence.get(r["tgt"], ()))
                    cs = set(price.incidence.get(r["src"], ()))
                    refuting |= ct - cs
            keys_with_base = [k for k in m["keys"] if qsampler.base_fp.get(k) is not None]
            for name, (smp, res) in m["supports"].items():
                if res.n == 0 or name == "curve":
                    continue
                k_ = min(per_market, res.n)
                picks = sorted(set(rng.choice(res.n, size=k_, replace=False).tolist())
                               | (refuting if name == "pricing" else set()))
                rows += _diagnose_support(cid, name, smp, res, picks,
                                          [k for k in m["keys"] if smp.base_fp.get(k) is not None])
            LOG.info("error check: market rank %d (cid %d) done", rank, cid)
    finally:
        con.close()
        s.extra.pop("sample_cache_only", None)
    df = pd.DataFrame(rows)
    write_csv(df, out / "error_check.csv")
    summ = {"seconds": round(time.perf_counter() - t0, 1), "neighbours_checked": int(len(df))}
    if len(df):
        g = df.groupby(["cid", "support"]).agg(checked=("index", "size"), with_error=("errors", lambda x: int((x > 0).sum())),
                                               queries_lost=("queries_lost_old", "sum"),
                                               changed_sets_differ=("new_changes_found", lambda x: int((x > 0).sum())),
                                               old_matches_cached=("old_matches_cached", "mean"),
                                               stored_matches=("stored_matches_recomputed", "mean")).reset_index()
        write_csv(g, out / "error_check_summary.csv")
        summ.update({"with_error": int((df.errors > 0).sum()),
                     "changed_sets_differ": int((df.new_changes_found > 0).sum()),
                     "old_matches_cached_share": float(df.old_matches_cached.mean()),
                     "stored_rule": sorted(df.stored_rule.unique().tolist()),
                     "stored_matches_share": float(df.stored_matches_recomputed.mean()),
                     "stored_mismatch_markets": sorted({int(c) for c in df.loc[~df.stored_matches_recomputed, "cid"]})})
    write_json(summ, out / "error_check.json")
    LOG.info("error check: %s", summ)
    return summ

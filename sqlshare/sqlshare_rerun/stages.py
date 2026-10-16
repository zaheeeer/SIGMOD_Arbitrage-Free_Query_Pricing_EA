"""Pipeline stages. Each stage writes its outputs plus a ``.done`` marker keyed by the
settings that affect it; rerunning a command skips stages that are already complete."""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import corpus as C
from . import database as D
from . import RESULTS_VERSION
from .config import PILOT_OWNERS, Settings
from .lattice import cover_structure, market_lattice, query_predicate
from .planner import min_cost_row_cover, run_protocol
from .pricing import (audit_monotone, chawla_all, draw_buyers, exact_entropy, info_families,
                      qirana_prices,
                      monopoly_prices, qirana_from_edges, querymarket_cover,
                      querymarket_single_predecessor, scaled_prices, score)
from .support import (SupportSampler, base_fingerprints, resolve_table, table_catalog,
                      table_counts)
from .util import (LOG, environment_info, load_pickle, mark_done, quote_ident, read_csv, read_json, save_pickle,
                   stage_done, timed, write_csv, write_json)

FAMILIES = ("answer_cells", "support_rows", "rows_x_log_support", "flat")


def _row_seed(seed: int) -> int:
    return int(np.random.SeedSequence([int(seed), 0x5EED]).generate_state(1)[0])


def _connect(s: Settings):
    tmp = s.work_dir / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return D.connect(s.db_path, s.duckdb_threads, s.memory_limit, tmp)


# ========================================================================================
# 1. download
# ========================================================================================
def stage_download(s: Settings) -> dict:
    info_path = s.data_dir / "release_info.json"
    if s.zip_file.exists() and info_path.exists():
        info = read_json(info_path)
        if info.get("bytes") == s.zip_file.stat().st_size and info.get("path") == str(s.zip_file):
            LOG.info("release present and checked earlier (%s)", s.zip_file)
            return info
    user_zip = s.zip_path is not None
    if not s.zip_file.exists():
        if user_zip:
            raise RuntimeError(f"the release zip you gave (-Zip) does not exist: {s.zip_file}")
        C.download(s.zip_url, s.zip_file)
    try:
        info = {"path": str(s.zip_file), "bytes": s.zip_file.stat().st_size, **C.verify_zip(s.zip_file)}
    except RuntimeError:
        if not user_zip:
            bad = s.zip_file.with_suffix(".zip.broken")
            if bad.exists():
                bad.unlink()
            s.zip_file.rename(bad)
            LOG.error("the downloaded file is damaged; it was renamed to %s. Run the same command "
                      "again to download it fresh.", bad.name)
        raise
    s.data_dir.mkdir(parents=True, exist_ok=True)
    write_json(info, info_path)
    LOG.info("release SHA-256 %s", info["sha256"])
    return info


def _prepare_key(s: Settings, info: dict) -> str:
    return (f"{info.get('sha256', info.get('bytes'))}|{s.min_component_queries}|{s.max_component_gb}"
            f"|{s.row_cap}|code {RESULTS_VERSION}")


# ========================================================================================
# 2. prepare: corpus, extraction, database, execution (shared by both protocols)
# ========================================================================================
def stage_prepare(s: Settings, force: bool = False) -> dict:
    info = stage_download(s)
    marker = s.work_dir / "prepare.done"
    key = _prepare_key(s, info)
    if not force and stage_done(marker, key) and s.db_path.exists():
        LOG.info("prepare: already complete")
        return read_json(s.work_dir / "prepare_summary.json")
    s.work_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():                 # the database is about to be rebuilt: the old marker is void
        marker.unlink()
    timings = {}
    with timed("parse release", timings):
        corpus = C.parse_release(s.zip_file)
    rows = corpus.rows
    resolvable = sum(1 for r in rows if r["status"] == "resolvable")
    LOG.info("queries %d | resolvable %d | views %d", len(rows), resolvable, len(corpus.views))
    with timed("select markets", timings):
        sel = C.select_markets(corpus, s.min_component_queries, s.max_component_gb)
        need_v = C.needed_views(corpus, sel.sel_q)
    LOG.info("components %d | selected %d with %d queries | views needed %d",
             sel.n_components, len(sel.selected), len(sel.sel_q), len(need_v))
    members = C.files_to_extract(corpus, sel, need_v, PILOT_OWNERS)
    with timed(f"extract {len(members)} files", timings):
        manifest = C.extract(s.zip_file, members, s.work_dir / "files")
    zip_order = {name: i for i, (name, _) in enumerate(corpus.infos)}
    disk = D.windows_disk_view(manifest, zip_order)
    collapsed = sum(1 for m in disk if m.get("collapsed"))
    if collapsed:
        LOG.info("%d file names collided on a case-insensitive disk (kept as the historical run did)", collapsed)

    for p in (s.db_path, s.db_path.with_suffix(".duckdb.wal")):
        if p.exists():
            p.unlink()
    shutil.rmtree(s.work_dir / "answers", ignore_errors=True)   # answers depend on this database
    con = _connect(s)
    with timed("load tables", timings):
        load_df = D.load_all(con, disk)
    n_ok = int(load_df["mode"].notna().sum())
    LOG.info("loaded %d/%d files, %s rows", n_ok, len(load_df), f"{int(load_df.rows.sum()):,}")
    with timed("create aliases", timings):
        n_alias, alias_map = D.create_aliases(con, load_df)
    schemas = D.schema_list(con)
    with timed("create views", timings):
        made, failed, msg = D.build_views(con, corpus, need_v, schemas)
    LOG.info("views %d/%d created", len(made), len(need_v))
    with timed("bind and execute selected queries", timings):
        stages_df, execs_df = D.bind_and_execute(con, corpus, sel.sel_q, schemas, s.query_timeout_s)
    con.execute("CHECKPOINT")
    extensions = sorted(r[0] for r in con.execute(
        "SELECT extension_name FROM duckdb_extensions() WHERE loaded").fetchall())
    con.close()

    if len(execs_df):
        ok = execs_df[(execs_df.ok) & (execs_df.rows > 0)].copy()
    else:
        ok = pd.DataFrame(columns=["k", "own", "duck", "rows", "ok"])
    ok["comp"] = [sel.comp_of_query[k] for k in ok.k]
    assets = ok[["k", "own", "duck", "rows", "comp"]].reset_index(drop=True)
    if len(stages_df) == 0:
        stages_df = pd.DataFrame(columns=["k", "stage"])

    transpiled = int((~stages_df.stage.isin(["transpile_error", "unsupported_patindex"])).sum())
    funnel = pd.DataFrame([
        {"stage": "queries in release", "n": len(rows)},
        {"stage": "all relations resolvable", "n": resolvable},
        {"stage": "in a selected market component", "n": len(sel.sel_q)},
        {"stage": "transpiled to DuckDB", "n": transpiled},
        {"stage": "binds against loaded schema", "n": int((stages_df.stage == "bind_ok").sum())},
        {"stage": "executes successfully", "n": int(execs_df.ok.sum()) if len(execs_df) else 0},
        {"stage": "returns a non-empty answer", "n": len(assets)},
    ])
    funnel["pct_of_corpus"] = (funnel.n / max(len(rows), 1)).round(4)
    funnel["expected_paper"] = [11121, 6398, 2660, 2506, 1135, 1051, 975]

    s.work_dir.mkdir(parents=True, exist_ok=True)
    save_pickle({"rows": rows, "ent": corpus.ent, "views": corpus.views, "tables": corpus.tables},
                s.work_dir / "corpus.pkl")
    save_pickle(sel, s.work_dir / "selection.pkl")
    save_pickle(assets, s.work_dir / "assets.pkl")
    save_pickle({"alias_map": alias_map, "schemas": schemas}, s.work_dir / "catalog.pkl")
    write_csv(load_df, s.work_dir / "load_log.csv")
    write_csv(stages_df, s.work_dir / "bind_stages.csv")
    write_csv(execs_df.drop(columns=["duck"], errors="ignore"), s.work_dir / "executions.csv")
    write_csv(pd.DataFrame([{"owner": o[0], "view": o[1], "reason": v, "message": msg.get(o, "")}
                            for o, v in sorted(failed.items())],
                           columns=["owner", "view", "reason", "message"]), s.work_dir / "views_failed.csv")
    write_json(manifest, s.work_dir / "manifest.json")
    write_csv(funnel, s.work_dir / "funnel.csv")
    summary = {
        "release": info, "components_total": sel.n_components,
        "components_selected": len(sel.selected), "queries_selected": len(sel.sel_q),
        "files_extracted": len(members), "files_loaded": n_ok, "rows_loaded": int(load_df.rows.sum()),
        "aliases": n_alias, "views_needed": len(need_v), "views_created": len(made),
        "survivors": len(assets), "timings_s": timings, "environment": environment_info(),
        "duckdb_extensions_loaded": extensions,
    }
    write_json(summary, s.work_dir / "prepare_summary.json")
    mark_done(marker, key)
    LOG.info("prepare: %d non-empty answers (paper: 975)", len(assets))
    return summary


# ========================================================================================
# shared helpers
# ========================================================================================
def _load_prepared(s: Settings):
    corpus = load_pickle(s.work_dir / "corpus.pkl")
    sel = load_pickle(s.work_dir / "selection.pkl")
    assets = load_pickle(s.work_dir / "assets.pkl")
    cat = load_pickle(s.work_dir / "catalog.pkl")
    return corpus, sel, assets, cat


class _Closure:
    """Base-table closure over the parsed corpus (same logic as corpus.Corpus)."""

    def __init__(self, corpus: dict):
        self.rows, self.ent, self.views = corpus["rows"], corpus["ent"], corpus["views"]
        self._cache = {}

    def base(self, obj, seen=None):
        seen = seen or set()
        if obj in seen:
            return set()
        seen = seen | {obj}
        if obj in self.ent:
            return {obj}
        if obj in self.views:
            out = set()
            for d in self.views[obj]:
                out |= self.base(d, seen)
            return out
        return set()

    def query_tables(self, k):
        if k not in self._cache:
            out = set()
            for ref in self.rows[k]["refs"]:
                out |= self.base(ref)
            self._cache[k] = out
        return self._cache[k]


def _market_ranks(assets: pd.DataFrame, selected: list | None = None) -> list:
    """Markets ordered by survivors; ties keep the historical component order."""
    cc = assets.comp.value_counts()
    pos = {int(c): i for i, c in enumerate(selected or [])}
    order = sorted(cc.index, key=lambda c: (-int(cc[c]), pos.get(int(c), 10 ** 9), int(c)))
    return [(rank + 1, int(c), int(cc[c])) for rank, c in enumerate(order)]


def _answers(s: Settings, con, assets: pd.DataFrame, cid: int):
    tb = "_tb" if s.limit_tiebreak else ""      # revision 3 (B1): separate cache
    path = s.work_dir / "answers" / f"{cid}_cap{s.row_cap}{tb}.pkl"
    if path.exists():
        return load_pickle(path)
    sub = assets[(assets.comp == cid) & (assets.rows <= s.row_cap)]
    status = {}
    if s.limit_tiebreak:
        sub = sub.copy()
        new = []
        for r in sub.itertuples():
            q, st = D.tiebreak_sql(con, r.own, r.duck)
            new.append(q)
            if st != "unchanged":
                status[r.k] = st
        sub["duck"] = new
        if status:
            LOG.info("market %d: tie-break added to %d LIMIT queries (%d kept unchanged after a failure)",
                     cid, sum(1 for v in status.values() if v == "tiebreak"),
                     sum(1 for v in status.values() if v.startswith("failed")))
    answers, seconds = D.materialize(con, sub, s.query_timeout_s, repeats=1)
    meta = {r.k: (r.own, r.duck) for r in sub.itertuples() if r.k in answers}
    data = {"answers": answers, "seconds": seconds, "meta": meta, "tiebreak": status}
    save_pickle(data, path)
    return data


def _lattice_pairs(s: Settings, cid: int):
    return load_pickle(s.lattice_dir / "pairs" / f"{cid}.pkl")


# ========================================================================================
# 3. lattice
# ========================================================================================
def stage_lattice(s: Settings, force: bool = False) -> dict:
    stage_prepare(s)
    out = s.lattice_dir
    marker = out / "lattice.done"
    key = f"{s.fingerprint()}|{read_json(s.work_dir / 'prepare.done')['key']}"
    if not force and stage_done(marker, key):
        LOG.info("lattice (%s): already complete", s.protocol)
        return read_json(out / "summary.json")
    corpus, sel, assets, cat = _load_prepared(s)
    ranks = _market_ranks(assets, sel.selected)
    con = _connect(s)
    comp_rows, pair_rows = [], []
    t0 = time.perf_counter()
    for rank, cid, surv in ranks:
        data = _answers(s, con, assets, cid)
        answers = data["answers"]
        structural = surv >= s.min_lattice_survivors
        if not structural and len(answers) < s.market_min_assets:
            continue
        preds = None
        if s.selection == "predicate":
            preds = {k: query_predicate(data["meta"][k][1]) for k in answers}
        L = market_lattice(answers, s.selection, preds, s.n_workers(), s.work_dir / "tmp", s.pool_stall_s)
        save_pickle(L, out / "pairs" / f"{cid}.pkl")
        fam = L["by_family"]
        comp_rows.append({"rank": rank, "cid": cid, "survivors": surv, "materialized": len(answers),
                          "structural": structural, "pairs_attempted": L["attempted"],
                          "verified_pairs": len(L["verified"]),
                          "n_projection": fam.get("projection", 0), "n_rollup": fam.get("rollup", 0),
                          "n_selection": fam.get("selection", 0)})
        for (i, j, f) in L["verified"]:
            pair_rows.append({"rank": rank, "cid": cid, "src": i, "tgt": j, "family": f})
        LOG.info("lattice rank %d (cid %d): %d assets, %d attempted, %d verified %s (%.0f s)",
                 rank, cid, len(answers), L["attempted"], len(L["verified"]), fam,
                 time.perf_counter() - t0)
    con.close()
    if not comp_rows:
        raise RuntimeError("no market has enough surviving queries for a lattice; see work/funnel.csv")
    comps = pd.DataFrame(comp_rows)
    pairs = pd.DataFrame(pair_rows, columns=["rank", "cid", "src", "tgt", "family"])
    write_csv(comps, out / "components.csv")
    write_csv(pairs, out / "pairs.csv")
    st = comps[comps.structural]
    stp = pairs[pairs.cid.isin(st.cid)]
    tot = int(st.verified_pairs.sum())
    fam = stp.family.value_counts().to_dict()
    summary = {
        "markets_structural": int(len(st)), "survivors_covered": int(st.survivors.sum()),
        "materialized_assets": int(st.materialized.sum()), "pairs_attempted": int(st.pairs_attempted.sum()),
        "verified_pairs": tot, "pairs_per_asset": round(tot / max(int(st.materialized.sum()), 1), 2),
        "family_counts": fam,
        "family_share_pct": {k: round(100 * v / max(tot, 1), 1) for k, v in fam.items()},
        "markets_with_structure": int((st.verified_pairs > 0).sum()),
        "row_cap_exclusions": int((assets.rows > s.row_cap).sum()),
        "survivors_total": int(len(assets)),
    }
    write_json(summary, out / "summary.json")
    mark_done(marker, key)
    return summary


# ========================================================================================
# market inputs shared by the main market and the full protocol on other markets
# ========================================================================================
def _support_setup(s: Settings, con, cat, keys, meta, closure: _Closure, weights_kind: str):
    """Tables, readers, weights, deletable targets and counts for one market."""
    readers = collections.defaultdict(list)
    for k in keys:
        for b in sorted(closure.query_tables(k)):
            readers[b].append(k)
    counts = table_counts(con, sorted(readers))
    catalog = table_catalog(con)
    if s.alias_delete == "resolve":
        merged, mcount = collections.defaultdict(list), {}
        for b in sorted(readers):
            tgt = resolve_table(b, catalog, cat["alias_map"], "resolve")
            if tgt is None or b not in counts:
                continue
            tb = (tgt[0].lower(), tgt[1].lower())
            for k in readers[b]:
                if k not in merged[tb]:
                    merged[tb].append(k)
            mcount[tb] = counts[b]
        readers = {b: sorted(v) for b, v in merged.items()}
        counts = mcount
        targets = {b: resolve_table(b, catalog, cat["alias_map"], "resolve") for b in readers}
    else:
        targets = {b: resolve_table(b, catalog, cat["alias_map"], "skip") for b in readers}
    tables = sorted(b for b in readers if counts.get(b, 0) > 0)
    size_w = np.array([counts[b] for b in tables], float)
    read_w = np.array([len(readers[b]) for b in tables], float)
    # The historical per-market sampler drew from every table the market reads, empty or
    # unloaded ones included (a draw on those is rejected); same table stream when kept.
    all_tables = sorted(readers)
    all_read_w = np.array([len(readers[b]) for b in all_tables], float)
    return {"tables": tables, "readers": dict(readers), "counts": counts, "targets": targets,
            "size_w": size_w, "read_w": read_w, "all_tables": all_tables, "all_read_w": all_read_w}


def _qirana_weights(s: Settings, sup: dict, tables=None):
    """Table weights for the Qirana support set (revision 3, D8): uniform over the
    relations, as Qirana specifies, or by readership or size for comparison."""
    tabs = sup["tables"] if tables is None else tables
    if s.qirana_table_weights == "uniform":
        return np.ones(len(tabs), float)
    if s.qirana_table_weights == "size":
        return np.array([sup["counts"].get(b, 0) for b in tabs], float)
    if s.qirana_table_weights == "readership":
        return np.array([len(sup["readers"].get(b, ())) for b in tabs], float)
    raise ValueError(f"unknown qirana_table_weights {s.qirana_table_weights!r}")


def _market_sampler(s: Settings, con, schemas, asset_sql, sup, base_fp):
    """Sampler for a market's pricing support: deletions (2.x) or Qirana updates."""
    return SupportSampler(con, schemas, asset_sql, sup["readers"], base_fp, sup["targets"],
                          sup["counts"], s.row_sampling, s.n_threads(), s.pool_stall_s,
                          neighbours=s.neighbours, swap_ratio=s.swap_ratio, donor_mode=s.qirana_donors,
                          isolate_errors=s.isolate_errors, error_as_change=s.error_as_change)


def _op_setup(s: Settings, con, cat, keys, meta):
    """Revision 3 (D3): query-text structure for the operational buyer, built from the
    public catalog only (query texts and table column lists), or None when switched off."""
    if s.attackers not in ("oracle", "both"):
        raise ValueError(f"unknown attackers {s.attackers!r}")
    if s.attackers == "oracle" and s.querymarket != "cover_semantic":
        return None
    from .operational import parse_query, semantic_pairs, target_structure
    catalog = table_catalog(con)
    cols_cache = {}

    def resolve(schema, name):
        hit = catalog.get((schema.lower(), name.lower()))
        if hit is None:
            return None
        if hit[2] == "BASE TABLE":
            return (hit[0], hit[1])
        real = cat["alias_map"].get((schema.lower(), name.lower()))
        if real is not None:
            hit2 = catalog.get((real[0].lower(), real[1].lower()))
            if hit2 is not None and hit2[2] == "BASE TABLE":
                return (hit2[0], hit2[1])
        return None

    def columns(base):
        if base not in cols_cache:
            hit = catalog.get(base)
            q = f"{quote_ident(hit[0])}.{quote_ident(hit[1])}"
            cols_cache[base] = [d[0] for d in con.execute(f"SELECT * FROM {q} LIMIT 0").description]
        return cols_cache[base]

    forms = {k: parse_query(k, meta[k][1], resolve, columns, default_schema=meta[k][0]) for k in keys}
    structures = {t: target_structure(forms[t], forms) for t in keys}
    ki = {k: i for i, k in enumerate(keys)}
    sem = [(ki[a], ki[b]) for a, b in semantic_pairs(structures)]
    stats = {"in_fragment": sum(f.ok for f in forms.values()), "assets": len(keys),
             "targets_with_sources": sum(1 for v in structures.values() if v and v[0]),
             "semantic_pairs": len(sem),
             "outside_fragment_reasons": dict(collections.Counter(f.reason for f in forms.values() if not f.ok))}
    return {"forms": forms, "structures": structures, "sem_pairs": sem, "stats": stats}


def _provenance(con, keys, closure: _Closure) -> np.ndarray:
    tabs = sorted({b for k in keys for b in closure.query_tables(k)})
    cnt = table_counts(con, tabs)
    return np.array([sum(cnt.get(b, 0) for b in closure.query_tables(k)) for k in keys], float)


def _scores(s: Settings, keys, answers, seconds, prov, edges, changed_fp, support_n,
            pairs, contrib, nrows, qm_base, op=None):
    n = len(keys)
    rowsz = np.array([len(answers[k]) for k in keys], float)
    et = np.array([seconds.get(k, 0.0) for k in keys], float)
    pos = et[et > 0]
    et[et <= 0] = pos.min() if len(pos) else 1.0
    q = qirana_from_edges(edges, support_n)
    if s.entropy == "exact":
        q["qirana_shannon"] = np.array([exact_entropy(changed_fp.get(k, {}), support_n) for k in keys])
    qm_inst = None
    if s.querymarket == "single_predecessor":
        qm, n_cat = querymarket_single_predecessor(n, pairs, qm_base)
    elif s.querymarket in ("cover", "cover_semantic"):
        qm, n_cat = querymarket_cover(keys, pairs, prov, contrib, nrows,
                                      lambda c, r, p: min_cost_row_cover(c, r, p, s.ilp_time_limit))
        if s.querymarket == "cover_semantic":
            # Revision 3 (D10 = C): Koutris et al. price through INSTANCE-BASED determinacy.
            # Only derivations guaranteed by the query texts are used (they hold on every
            # database, so instance-based determinacy holds); the instance-level version
            # above is kept as the comparison "querymarket_viewcover_instance".
            if op is None:
                raise ValueError("querymarket='cover_semantic' needs the query-text structure")
            qm_inst = qm
            sem_contrib = {t: st[0] for t, st in op["structures"].items() if st and st[0] and st[3] == "set"}
            sem_n = {t: op["structures"][t][1] for t in sem_contrib}
            qm, n_cat = querymarket_cover(keys, op["sem_pairs"], prov, sem_contrib, sem_n,
                                          lambda c, r, p: min_cost_row_cover(c, r, p, s.ilp_time_limit))
    else:
        raise ValueError(f"unknown querymarket {s.querymarket!r}")
    scores = {"uniform": np.ones(n), "size_proportional": rowsz, "compute_metered": et,
              "qirana_weighted_coverage": q["qirana_weighted_coverage"],
              "qirana_uniform_gain": q["qirana_uniform_gain"],
              "qirana_shannon": np.maximum(q["qirana_shannon"], 0),
              "querymarket_viewcover": qm}
    if qm_inst is not None:
        scores["querymarket_viewcover_instance"] = qm_inst
    if s.qirana_gain == "entropy_gain":
        # Revision 3 (D2): the published uniform entropy gain log|C|/log|S| replaces the
        # historical |C|/|S|, which was weighted coverage over a constant.
        del scores["qirana_uniform_gain"]
        scores["qirana_uniform_entropy_gain"] = qirana_prices(edges, support_n)["qirana_uniform_entropy_gain"]
    elif s.qirana_gain != "fraction":
        raise ValueError(f"unknown qirana_gain {s.qirana_gain!r}")
    return scores, n_cat, rowsz


def _runs(s: Settings, label: str, keys, answers, pairs, scores, edges, support_n, info,
          contrib, nrows, tmp: Path, op=None, plans_out: Path | None = None) -> pd.DataFrame:
    need = set(contrib)
    for c_ in contrib.values():
        need |= set(c_)
    if op is not None:
        for t, st in op["structures"].items():
            if st and st[0]:
                need |= {t} | set(st[0])
    payload = {
        "keys": keys, "ki": {k: i for i, k in enumerate(keys)},
        "answers": {k: answers[k] for k in need}, "contrib": contrib, "nrows": nrows,
        "pairs": pairs, "scale_scores": scores, "edges": edges, "support_n": support_n,
        "info": info,
        "cfg": {"demand": s.demand, "fit": s.fit, "n_buyers": s.n_buyers, "max_lps": s.max_lps,
                "cip_eps": s.cip_eps, "ilp_time_limit": s.ilp_time_limit,
                "exact_formulation": s.exact_formulation(), "log_plans": bool(s.log_plans),
                "record_costs": bool(s.record_costs)},
        "op": None if op is None else {k: op[k] for k in ("forms", "structures", "sem_pairs")},
    }
    path = tmp / f"market_{label}.pkl"
    save_pickle(payload, path)
    arms = ["published"] if s.demand == "redraw" else (["stationary", "shift"] if s.shift_arm else ["stationary"])
    tasks = None
    if s.demand == "arms":                 # revision 3 (D7): main arm planned, study arms not
        tasks = [(s.main_arm, f, sd, s.n_buyers, True) for f in FAMILIES for sd in s.run_seeds]
        if label.startswith("main_"):
            tasks += [(a, f, sd, s.n_buyers, False) for a in s.demand_arms if a != s.main_arm
                      for f in FAMILIES for sd in s.run_seeds]
            tasks += [(f"{s.main_arm}_n{n}", f, sd, n, False) for n in s.train_sizes if n != s.n_buyers
                      for f in FAMILIES for sd in s.run_seeds]
    try:
        df = run_protocol(path, arms, FAMILIES, s.run_seeds, s.n_workers(), s.pool_stall_s,
                          with_plans=bool(s.log_plans), tasks=tasks)
        if s.log_plans:
            df, plans = df
            if plans_out is not None:
                plans_out.parent.mkdir(parents=True, exist_ok=True)
                write_csv(plans, plans_out)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    return df


def _sample_cache_dir(s: Settings):
    """Ablation runs share draws through ``extra["sample_cache"]``. Revision 3 (Part 5): the
    revised protocol also keeps every finished support set in work/sample_cache_revised, so
    a stopped run (for example during the many-hour markets stage) continues from the last
    finished support set instead of drawing everything again. A reused draw is the one a
    fresh draw gives (the key holds every input of the sampler). The 2.x protocols are
    unchanged."""
    cdir = s.extra.get("sample_cache")
    if cdir:
        return cdir
    if s.protocol == "revised":
        return str(s.work_dir / "sample_cache_revised")
    return None


def _sample(s: Settings, sampler, prep_key: str, kind: str, readers: dict, tables, weights, n: int,
            table_seed: int, row_seed: int, **kw):
    """``sampler.sample``, reused from disk in ablation runs.

    Only ablation runs set ``extra["sample_cache"]``. The cache key holds every input the
    sampler reads (database, row rule, tables, weights, readers, counts, seeds, size,
    checkpoints, tracked assets, fingerprint recording), so a reused result is the one a
    fresh draw would give; ablation switches that do not touch sampling then cost minutes.
    """
    res = _sample_inner(s, sampler, prep_key, kind, readers, tables, weights, n, table_seed, row_seed, **kw)
    errs = getattr(res, "errors", {}) or {}
    s.extra.setdefault("query_errors", {})[kind] = {
        "support_databases": int(res.n), "with_query_errors": int(len(errs)),
        "query_errors": int(sum(errs.values()))}
    return res


def _sample_inner(s: Settings, sampler, prep_key: str, kind: str, readers: dict, tables, weights, n: int,
                  table_seed: int, row_seed: int, **kw):
    cdir = _sample_cache_dir(s)
    if not cdir:
        return sampler.sample(tables, weights, n, table_seed, row_seed, **kw)
    Path(cdir).mkdir(parents=True, exist_ok=True)
    track = kw.get("track")
    blob = json.dumps({
        "results_version": RESULTS_VERSION, "prepare": prep_key, "row_sampling": s.row_sampling,
        "alias_delete": s.alias_delete, "kind": kind, "n": int(n), "table_seed": int(table_seed),
        **({"neighbours": sampler.neighbours, "swap_ratio": sampler.swap_ratio}
           if getattr(sampler, "neighbours", "delete") != "delete" else {}),
        **({"donors": sampler.donor_mode} if getattr(sampler, "donor_mode", "distinct") != "distinct" else {}),
        **({"isolate_errors": True, "error_as_change": bool(sampler.error_as_change)}
           if getattr(sampler, "isolate_errors", False) else {}),
        **({"trace_costs": True} if getattr(sampler, "trace_costs", False) else {}),
        # the prepare key fixes the query texts, except for the LIMIT tie-break (Part 7b):
        # then the texts the sampler runs enter the key
        **({"sql": hashlib.sha1(json.dumps(sorted((str(k), str(v)) for k, v in sampler.assets.items()))
                                .encode()).hexdigest()} if s.limit_tiebreak else {}),
        "row_seed": int(row_seed), "checkpoints": [int(c) for c in kw.get("checkpoints", ())],
        "track": sorted(map(str, track)) if track is not None else None,
        "record_fp": bool(kw.get("record_fp", False)),
        "tables": [str(b) for b in tables], "weights": [float(w) for w in weights],
        "readers": {str(b): sorted(map(str, v)) for b, v in sorted(readers.items(), key=lambda x: str(x[0]))},
        "counts": {str(b): int(c) for b, c in sorted(sampler.counts.items(), key=lambda x: str(x[0]))},
        "targets": {str(b): str(v) for b, v in sorted(sampler.targets.items(), key=lambda x: str(x[0]))},
    }, sort_keys=True).encode()
    path = Path(cdir) / f"{kind}_{hashlib.sha1(blob).hexdigest()[:20]}.pkl"
    if s.extra.get("sample_cache_only") and not path.exists():
        raise RuntimeError(f"support set {kind!r} is not in the sample cache {cdir}; the extraction "
                           "reads finished support sets only (run the revised protocol first)")
    if path.exists():
        try:
            res = load_pickle(path)
            LOG.info("%s: reused the identical draw from the sample cache", kind)
            return res
        except Exception:                  # a damaged cache file is simply drawn again
            pass
    res = sampler.sample(tables, weights, n, table_seed, row_seed, **kw)
    save_pickle(res, path)
    return res


def _metered_seconds(s: Settings, con, sub: pd.DataFrame, cid: int) -> dict:
    """Median of ``metered_repeats`` warm timings per asset. Ablation runs measure once and
    share the timings, so a switch never shows timing noise as an effect."""
    cdir = s.extra.get("sample_cache")
    path = None
    if cdir:
        tag = hashlib.sha1(read_json(s.work_dir / "prepare.done")["key"].encode()).hexdigest()[:12]
        path = Path(cdir) / f"metered_{cid}_x{s.metered_repeats}_{tag}.pkl"
        if path.exists():
            try:
                seconds = load_pickle(path)
                LOG.info("compute-metered timings: reused from the ablation sample cache")
                return seconds
            except Exception:
                pass
    _, seconds = D.materialize(con, sub, s.query_timeout_s, repeats=s.metered_repeats)
    if path is not None:
        save_pickle(seconds, path)
    return seconds


def _summaries(runs: pd.DataFrame, by=("arm", "mechanism")):
    num = [c for c in runs.columns if c not in ("arm", "family", "seed", "mechanism", "market", "rank", "cid")
           and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby(list(by))[num]
    mean = g.mean().add_suffix("_mean")
    sd = g.std(ddof=1).add_suffix("_sd")
    mx = runs.groupby(list(by))["comb_max_gain"].max().rename("comb_max_gain_global")
    return pd.concat([mean, sd, mx], axis=1).reset_index()


# ========================================================================================
# 4. main market (largest component): Table 6, falsification, Tables 3-5 SQLShare
# ========================================================================================
def stage_main_market(s: Settings, force: bool = False) -> dict:
    stage_prepare(s)
    out = s.results_dir / "main_market"
    marker = out / "main.done"
    key = f"{s.fingerprint()}|{read_json(s.work_dir / 'prepare.done')['key']}"
    if not force and stage_done(marker, key):
        LOG.info("main market (%s): already complete", s.protocol)
        return read_json(out / "summary.json")
    if not s.extra.get("lattice_ready"):       # ablation runs prepare their lattice first
        stage_lattice(s)
    corpus, sel, assets, cat = _load_prepared(s)
    closure = _Closure(corpus)
    ranks = _market_ranks(assets, sel.selected)
    if not ranks:
        raise RuntimeError("no query survived preparation, so there is no market to price; see work/funnel.csv")
    rank, cid, surv = ranks[0]
    con = _connect(s)
    data = _answers(s, con, assets, cid)
    answers, meta = data["answers"], data["meta"]
    keys = sorted(answers)
    ki = {k: i for i, k in enumerate(keys)}
    seconds = data["seconds"]
    if s.metered_repeats > 1:
        sub = assets[assets.k.isin(keys)]
        if s.limit_tiebreak:                   # time the queries as they are priced
            sub = sub.assign(duck=sub.k.map(lambda k: meta[k][1]))
        seconds = _metered_seconds(s, con, sub, cid)
    L = _lattice_pairs(s, cid)
    pairs = [(ki[a], ki[b]) for a, b, _ in L["verified"] if a in ki and b in ki]
    LOG.info("main market: cid %d, %d assets, %d verified pairs", cid, len(keys), len(pairs))
    prov = _provenance(con, keys, closure)
    ncols = np.array([answers[k].shape[1] for k in keys], float)

    sup = _support_setup(s, con, cat, keys, meta, closure, "size")
    tables, readers = sup["tables"], sup["readers"]
    if not tables:
        raise RuntimeError("the largest market has no loaded, non-empty base table to sample from; "
                           "see work/load_log.csv")
    with np.errstate(invalid="ignore", divide="ignore"):
        r_size_read = float(np.corrcoef(sup["size_w"], sup["read_w"])[0, 1]) if len(tables) > 1 else float("nan")
    schemas = cat["schemas"]
    asset_sql = {k: meta[k] for k in keys}
    with timed("base fingerprints"):
        base_fp = base_fingerprints(con, schemas, asset_sql)
    sampler = SupportSampler(con, schemas, asset_sql, readers, base_fp, sup["targets"], sup["counts"],
                             s.row_sampling, s.n_threads(), s.pool_stall_s,
                             isolate_errors=s.isolate_errors, error_as_change=s.error_as_change)
    timings = {}
    prep_key = read_json(s.work_dir / "prepare.done")["key"]

    # ---- resolution curve and falsification --------------------------------------
    n_curve = max(s.curve_checkpoints)
    with timed(f"support set: size-proportional, {n_curve} neighbours (curve)", timings):
        curve_res = _sample(s, sampler, prep_key, "curve", readers, tables, sup["size_w"], n_curve,
                            s.curve_seed, _row_seed(s.curve_seed),
                            checkpoints=s.curve_checkpoints, label="curve")
    curve = []
    for c in s.curve_checkpoints:
        v = curve_res.counts(keys, upto=c)
        curve.append({"support_rows": c, "zero_price": int((v == 0).sum()),
                      "distinct_prices": len(set(v.tolist())), "median": float(np.median(v)),
                      "max": float(v.max()), "elapsed_s": curve_res.checkpoint_seconds.get(c)})
    dc = curve_res.counts(keys)
    viol = [(a, b) for a, b in pairs if dc[b] > dc[a] + 1e-9]
    involved = sorted({keys[i] for p in viol for i in p})
    with timed(f"witness search, {s.witness_n} neighbours", timings):
        wit_res = _sample(s, sampler, prep_key, "witness", readers, tables, sup["size_w"], s.witness_n,
                          s.witness_seed, _row_seed(s.witness_seed + 7919), track=involved, label="witness")
    inc = {k: set(wit_res.incidence.get(k, ())) for k in involved}
    fals_rows = [{"src": keys[a], "tgt": keys[b], "dc_src": int(dc[a]), "dc_tgt": int(dc[b]),
                  "witnesses_found": len(inc.get(keys[b], set()) - inc.get(keys[a], set()))} for a, b in viol]

    # ---- Table 6: two samplers --------------------------------------------------
    with timed(f"support set: size-proportional, {s.table6_n} neighbours", timings):
        size_res = _sample(s, sampler, prep_key, "size", readers, tables, sup["size_w"], s.table6_n,
                           s.table6_seed, _row_seed(s.table6_seed), label="size-proportional")
    with timed(f"support set: readership-weighted, {s.table6_n} neighbours", timings):
        read_res = _sample(s, sampler, prep_key, "readership", readers, tables, sup["read_w"], s.table6_n,
                           s.table6_seed, _row_seed(s.table6_seed + 1), record_fp=(s.entropy == "exact"),
                           label="readership-weighted")

    price_res = read_res
    if s.neighbours == "qirana":
        qsampler = _market_sampler(s, con, schemas, asset_sql, sup, base_fp)
        with timed(f"support set: Qirana updates ({s.qirana_table_weights} tables), {s.table6_n} neighbours",
                   timings):
            price_res = _sample(s, qsampler, prep_key, f"qirana_{s.qirana_table_weights}", readers, tables,
                                _qirana_weights(s, sup), s.table6_n, s.table6_seed,
                                _row_seed(s.table6_seed + 2), record_fp=(s.entropy == "exact"),
                                label=f"qirana-{s.qirana_table_weights}")

    def t6(res, name):
        sizes = res.counts(keys)
        return {"sampling": name, "support_rows": res.n,
                "assets_priceable": int((sizes > 0).sum()), "assets": len(keys),
                "median_conflict_set": float(np.median(sizes)),
                "distinct_price_levels": len(set(sizes.tolist())), "build_seconds": res.seconds}
    t6_rows = [t6(size_res, "size_proportional (as specified)"), t6(read_res, "readership_weighted")]
    if price_res is not read_res:
        t6_rows.append(t6(price_res, f"qirana_updates_{s.qirana_table_weights}"))
    table6 = pd.DataFrame(t6_rows)

    fals_set = set(viol)
    clean1 = [p for p in pairs if p not in fals_set]
    dc2 = price_res.counts(keys)
    new_viol = [(a, b) for a, b in clean1 if dc2[b] > dc2[a] + 1e-9]
    clean = clean1 if s.remove_falsified == "curve" else [p for p in clean1 if p not in set(new_viol)]

    # ---- the full protocol ---------------------------------------------------------
    with timed("cover structure", timings):
        contrib, nrows = cover_structure(answers, keys, s.cover_target_cap)
    with timed("operational buyer structure (query texts only)", timings):
        op = _op_setup(s, con, cat, keys, meta)
    edges = price_res.edges(keys)
    scores, n_cat, rowsz = _scores(s, keys, answers, seconds, prov, edges, price_res.changed_fp,
                                   s.table6_n, clean, contrib, nrows, prov, op=op)
    info = info_families(rowsz, prov, ncols if s.answer_cells == "rows_x_cols" else np.ones(len(keys)))
    con.close()
    with timed("fit, freeze, and evaluate runs", timings):
        runs = _runs(s, f"main_{cid}", keys, answers, clean, scores, edges, s.table6_n, info,
                     contrib, nrows, s.work_dir / "tmp", op=op if s.attackers == "both" else None,
                     plans_out=out / "plans_log.csv")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(runs, out / "runs.csv")
    summ = _summaries(runs)
    write_csv(summ, out / "summary_by_mechanism.csv")
    write_csv(_summaries(runs, by=("arm", "family", "mechanism")), out / "summary_by_family.csv")
    write_csv(pd.DataFrame(curve), out / "resolution_curve.csv")
    write_csv(pd.DataFrame(fals_rows), out / "falsified_edges.csv")
    write_csv(table6, out / "table6_support_sampling.csv")
    if s.neighbours == "qirana" or s.log_plans:     # revision 3: kept for the worked example
        write_json({"kind": ("qirana_updates_" + s.qirana_table_weights) if price_res is not read_res
                    else "readership_deletions", "n": price_res.n,
                    "accepted": [[list(b), list(sp) if isinstance(sp, tuple) else sp]
                                 for b, sp in price_res.accepted],
                    "incidence": {str(k): list(map(int, v)) for k, v in price_res.incidence.items()}},
                   out / "pricing_support.json")
    pv = runs.groupby(["arm", "mechanism"])[["plans_verified", "plans_failed"]].sum().reset_index()
    write_csv(pv, out / "plan_verification.csv")
    main_arm = (s.main_arm if s.demand == "arms" else
                "published" if "published" in set(runs.arm) else "stationary")
    main = runs[runs.arm == main_arm]
    summary = {
        "cid": cid, "rank": rank, "survivors": surv, "assets": len(keys),
        "claimed_edges": len(pairs), "falsified_curve": len(viol),
        "witnessed": sum(1 for r in fals_rows if r["witnesses_found"] > 0),
        "falsified_by_pricing_support": len(new_viol),
        "false_edges_total": len(viol) + len(new_viol),
        "false_edge_rate_pct": round(100 * (len(viol) + len(new_viol)) / max(len(pairs), 1), 2),
        "clean_pairs_used": len(clean), "corr_size_vs_readership": round(r_size_read, 3),
        "zero_priced_readership": int((read_res.counts(keys) == 0).sum()),
        "zero_priced_pricing_support": int((dc2 == 0).sum()),
        "pricing_support": ("readership_deletions" if price_res is read_res
                            else f"qirana_updates_{s.qirana_table_weights}"),
        "cover_targets": len(contrib),
        "querymarket_catalogue": n_cat,
        "build_cost_ratio": round(read_res.seconds / max(size_res.seconds, 1e-9), 1),
        "main_arm": main_arm,
        "plans_verified": int(main.plans_verified.sum()), "plans_failed": int(main.plans_failed.sum()),
        "plans_failed_by_mechanism": {m: int(v) for m, v in main.groupby("mechanism").plans_failed.sum().items() if v},
        "plans_verified_all_arms": int(runs.plans_verified.sum()),
        "alpha_at_grid_max_share": float(runs.alpha_at_grid_max.mean()) if "alpha_at_grid_max" in runs else None,
        "neighbour_draws": {"curve": curve_res.draws, "witness": wit_res.draws,
                            "size": size_res.draws, "readership": read_res.draws,
                            **({"qirana": price_res.draws} if price_res is not read_res else {})},
        "timings_s": timings,
        **({"operational": op["stats"]} if op is not None else {}),
        "query_errors": {k: v for k, v in s.extra.get("query_errors", {}).items()},
        "tiebreak_queries": {str(k): v for k, v in data.get("tiebreak", {}).items()},
    }
    write_json(summary, out / "summary.json")
    mark_done(marker, key)
    return summary


# ========================================================================================
# 5. all markets: Table 7, Figure 2(b), cross-market falsification
# ========================================================================================
def _reduced_market(s: Settings, con, cat, closure, rank, cid, data, pairs_all):
    answers, meta = data["answers"], data["meta"]
    keys = sorted(answers)
    base = {"rank": rank, "cid": cid, "assets": len(keys)}
    if len(keys) < s.market_min_assets:
        return [{**base, "status": "too_small"}], None
    ki = {k: i for i, k in enumerate(keys)}
    pairs = [(ki[a], ki[b]) for a, b, _ in pairs_all if a in ki and b in ki]
    sup = _support_setup(s, con, cat, keys, meta, closure, "readership")
    tabs, wts = ((sup["all_tables"], sup["all_read_w"]) if s.market_tables == "all_readers"
                 else (sup["tables"], sup["read_w"]))
    if not tabs:
        return [{**base, "status": "no_base_tables"}], None
    asset_sql = {k: meta[k] for k in keys}
    base_fp = base_fingerprints(con, cat["schemas"], asset_sql)
    sampler = _market_sampler(s, con, cat["schemas"], asset_sql, sup, base_fp)
    if s.neighbours == "qirana":
        tabs = sup["tables"]
        wts = _qirana_weights(s, sup, tabs)
    res = _sample(s, sampler, read_json(s.work_dir / "prepare.done")["key"], f"market_{cid}", sup["readers"],
                  tabs, wts, s.market_support_n, s.market_seed, _row_seed(s.market_seed), label=f"market {rank}")
    edges = res.edges(keys)
    rowsz = np.array([len(answers[k]) for k in keys], float)
    prov = _provenance(con, keys, closure)
    ncols = np.array([answers[k].shape[1] for k in keys], float)
    contrib = nrows = None
    if s.querymarket != "single_predecessor":
        contrib, nrows = cover_structure(answers, keys, s.cover_target_cap)
    op = _op_setup(s, con, cat, keys, meta) if s.querymarket == "cover_semantic" else None
    rows = reduced_pricing(s, keys, pairs, edges, rowsz, prov, ncols, contrib, nrows, base, op=op)
    return rows, {"keys": keys, "pairs": pairs, "prov": prov, "ncols": ncols, "rowsz": rowsz}


def reduced_pricing(s: Settings, keys, pairs, edges, rowsz, prov, ncols, contrib, nrows, base, op=None):
    """The reduced per-market protocol of legacy/market_run.py (no cover planner)."""
    sizes = np.array([len(e) for e in edges], float)
    qs = qirana_from_edges(edges, s.market_support_n)
    if s.querymarket == "single_predecessor":
        qm, _ = querymarket_single_predecessor(len(keys), pairs, rowsz)
    elif s.querymarket == "cover_semantic":
        sem_contrib = {t: st[0] for t, st in op["structures"].items() if st and st[0] and st[3] == "set"}
        sem_n = {t: op["structures"][t][1] for t in sem_contrib}
        qm, _ = querymarket_cover(keys, op["sem_pairs"], prov, sem_contrib, sem_n,
                                  lambda c, r, p: min_cost_row_cover(c, r, p, s.ilp_time_limit))
    else:
        qm, _ = querymarket_cover(keys, pairs, prov, contrib, nrows,
                                  lambda c, r, p: min_cost_row_cover(c, r, p, s.ilp_time_limit))
    falsified = sum(1 for a, b in pairs if sizes[b] > sizes[a] + 1e-9)
    if s.market_support_family == "rows":
        info = info_families(rowsz, rowsz, np.ones(len(keys)))
    else:
        info = info_families(rowsz, prov, ncols if s.answer_cells == "rows_x_cols" else np.ones(len(keys)))
    rows = []
    for fam in s.market_families:
        for sd in s.market_seeds:
            (tt, tv), (ht, hv) = draw_buyers(s.demand, s.main_arm if s.demand == "arms" else "stationary",
                                             info[fam], 100 * sd + 3,
                                             s.market_buyers, s.market_buyers)
            pr, alphas = {}, {}
            pr["uniform"], alphas["uniform"] = scaled_prices(s.fit, tt, tv, np.ones(len(keys)))
            pr["qirana_weighted_coverage"], alphas["qirana_weighted_coverage"] = scaled_prices(
                s.fit, tt, tv, qs["qirana_weighted_coverage"])
            pr["querymarket_viewcover"], alphas["querymarket_viewcover"] = scaled_prices(s.fit, tt, tv, qm)
            pr["monopoly_per_asset"] = monopoly_prices(tt, tv, len(keys))
            pr.update(chawla_all(edges, tt, tv, s.market_support_n, max_lps=s.market_max_lps,
                                 eps=s.cip_eps, which=("chawla_ubp", "chawla_lpip")))
            ref = score(tt, tv, pr["monopoly_per_asset"])["revenue"]
            for nm, p in pr.items():
                sc_ = score(ht, hv, p)
                aud = audit_monotone(p, pairs)
                rows.append({**base, "pairs": len(pairs), "falsified": falsified,
                             "zero_priced": int((sizes == 0).sum()), "family": fam, "seed": sd,
                             "mechanism": nm, "revenue_norm": sc_["revenue"] / max(ref, 1e-12),
                             "welfare_norm": sc_["welfare"] / max(ref, 1e-12), "served": sc_["served"],
                             "mono_violation_rate": aud["violation_rate"],
                             "alpha": alphas.get(nm, np.nan), "status": "ok"})
    return rows


def _full_market(s: Settings, con, cat, closure, rank, cid, data, pairs_all, tmp):
    """Corrected protocol, C15: the main-market protocol on another market."""
    answers, meta = data["answers"], data["meta"]
    keys = sorted(answers)
    ki = {k: i for i, k in enumerate(keys)}
    pairs = [(ki[a], ki[b]) for a, b, _ in pairs_all if a in ki and b in ki]
    sup = _support_setup(s, con, cat, keys, meta, closure, "readership")
    if not sup["tables"]:
        return None
    asset_sql = {k: meta[k] for k in keys}
    base_fp = base_fingerprints(con, cat["schemas"], asset_sql)
    sampler = _market_sampler(s, con, cat["schemas"], asset_sql, sup, base_fp)
    wts = _qirana_weights(s, sup) if s.neighbours == "qirana" else sup["read_w"]
    res = _sample(s, sampler, read_json(s.work_dir / "prepare.done")["key"], f"full_market_{cid}", sup["readers"],
                  sup["tables"], wts, s.table6_n, s.table6_seed + rank, _row_seed(s.table6_seed + rank),
                  record_fp=(s.entropy == "exact"), label=f"full market {rank}")
    sizes = res.counts(keys)
    fals = {(a, b) for a, b in pairs if sizes[b] > sizes[a] + 1e-9}
    clean = [p for p in pairs if p not in fals]
    prov = _provenance(con, keys, closure)
    ncols = np.array([answers[k].shape[1] for k in keys], float)
    contrib, nrows = cover_structure(answers, keys, s.cover_target_cap)
    op = _op_setup(s, con, cat, keys, meta)
    edges = res.edges(keys)
    scores, _, rowsz = _scores(s, keys, answers, data["seconds"], prov, edges, res.changed_fp,
                               s.table6_n, clean, contrib, nrows, prov, op=op)
    info = info_families(rowsz, prov, ncols if s.answer_cells == "rows_x_cols" else np.ones(len(keys)))
    runs = _runs(s, f"market_{cid}", keys, answers, clean, scores, edges, s.table6_n, info,
                 contrib, nrows, tmp, op=op if s.attackers == "both" else None)
    runs.insert(0, "rank", rank)
    runs.insert(1, "cid", cid)
    runs["falsified_edges"] = len(fals)
    runs["pairs"] = len(pairs)
    return runs


def stage_markets(s: Settings, force: bool = False) -> dict:
    stage_prepare(s)
    out = s.results_dir / "markets"
    marker = out / "markets.done"
    key = f"{s.fingerprint()}|{read_json(s.work_dir / 'prepare.done')['key']}"
    if not force and stage_done(marker, key):
        LOG.info("markets (%s): already complete", s.protocol)
        return read_json(out / "summary.json")
    stage_lattice(s)
    main_runs = None
    if s.full_protocol_markets or s.full_markets:
        stage_main_market(s)
        main_runs = read_csv(s.results_dir / "main_market" / "runs.csv")
    corpus, sel, assets, cat = _load_prepared(s)
    closure = _Closure(corpus)
    ranks = _market_ranks(assets, sel.selected)
    con = _connect(s)
    all_rows, full_runs = [], []
    t0 = time.perf_counter()
    tb_all = {}
    for rank, cid, surv in ranks:
        data = _answers(s, con, assets, cid)
        if data.get("tiebreak"):
            tb_all[str(cid)] = {str(k): v for k, v in data["tiebreak"].items()}
        pairs_path = s.lattice_dir / "pairs" / f"{cid}.pkl"
        pairs_all = load_pickle(pairs_path)["verified"] if pairs_path.exists() else []
        rows, info = _reduced_market(s, con, cat, closure, rank, cid, data, pairs_all)
        all_rows += rows
        status = rows[0]["status"]
        LOG.info("market rank %d (cid %d, %d assets): %s (%.0f s)", rank, cid, rows[0]["assets"],
                 status, time.perf_counter() - t0)
        if (s.full_protocol_markets or (rank == 1 and s.full_markets) or cid in set(s.full_markets)) \
                and status == "ok":
            if rank == 1 and main_runs is not None:
                fr = main_runs.copy()          # the main market already ran the full protocol
                fr.insert(0, "rank", rank)
                fr.insert(1, "cid", cid)
            else:
                fr = _full_market(s, con, cat, closure, rank, cid, data, pairs_all, s.work_dir / "tmp")
            if fr is not None:
                full_runs.append(fr)
    con.close()
    out.mkdir(parents=True, exist_ok=True)
    am = pd.DataFrame(all_rows)
    if "status" not in am:
        am["status"] = pd.Series(dtype=str)
    write_csv(am, out / "runs.csv")
    ok = am[am.status == "ok"].copy()
    metric_cols = ["revenue_norm", "welfare_norm", "served", "mono_violation_rate"]
    if ok.empty:
        LOG.warning("no market had enough assets for the per-market protocol")
        ok = pd.DataFrame({c: pd.Series(dtype=float) for c in
                           ["rank", "cid", "assets", "pairs", "falsified", "zero_priced", *metric_cols]})
        ok["mechanism"] = pd.Series(dtype=object)
    struct = ok.groupby(["rank", "cid"])[["assets", "pairs", "falsified", "zero_priced"]].first().reset_index()
    struct["falsified_rate"] = struct.falsified / struct.pairs.replace(0, np.nan)
    struct["zero_priced_frac"] = struct.zero_priced / struct.assets
    write_csv(struct, out / "structure.csv")
    per = ok.groupby(["rank", "mechanism"])[metric_cols].mean().reset_index()
    write_csv(per, out / "per_market_means.csv")
    piv_r = per.pivot(index="rank", columns="mechanism", values="revenue_norm")
    piv_w = per.pivot(index="rank", columns="mechanism", values="welfare_norm")
    info_m, flat_m = ["qirana_weighted_coverage", "querymarket_viewcover"], ["uniform", "chawla_ubp"]
    for m in info_m + flat_m:
        for piv in (piv_r, piv_w):
            if m not in piv:
                piv[m] = np.nan
    fig = pd.DataFrame({"best_flat_welfare": piv_w[flat_m].max(axis=1),
                        "best_published_welfare": piv_w[info_m].max(axis=1),
                        "best_flat_revenue": piv_r[flat_m].max(axis=1),
                        "best_published_revenue": piv_r[info_m].max(axis=1)}).reset_index()
    write_csv(fig, out / "figure2b_markets.csv")
    t7 = []
    for m in ["uniform", "monopoly_per_asset", "qirana_weighted_coverage", "querymarket_viewcover",
              "chawla_ubp", "chawla_lpip"]:
        sub = ok[ok.mechanism == m]
        pm = per[per.mechanism == m]
        t7.append({"mechanism": m,
                   "revenue_mean": sub.revenue_norm.mean(), "revenue_sd_pooled": sub.revenue_norm.std(ddof=1),
                   "revenue_sd_across_markets": pm.revenue_norm.std(ddof=1),
                   "welfare_mean": sub.welfare_norm.mean(), "welfare_sd_pooled": sub.welfare_norm.std(ddof=1),
                   "welfare_sd_across_markets": pm.welfare_norm.std(ddof=1)})
    write_csv(pd.DataFrame(t7), out / "table7_markets.csv")
    n = len(piv_r)
    welfare_wins = int((fig.best_published_welfare > fig.best_flat_welfare).sum())
    revenue_wins = int((fig.best_flat_revenue > fig.best_published_revenue).sum())
    summary = {
        "markets_ok": int(n), "markets_too_small": int((am.status == "too_small").sum()),
        "assets": int(struct.assets.sum()), "pairs": int(struct.pairs.sum()),
        "falsified": int(struct.falsified.sum()), "markets_with_falsified": int((struct.falsified > 0).sum()),
        "welfare_wins_published": welfare_wins, "revenue_wins_flat": revenue_wins,
        "zero_priced_median_pct": round(100 * float(struct.zero_priced_frac.median()), 1) if n else None,
        "markets_all_zero_priced": int((struct.zero_priced_frac >= 1.0).sum()),
    }
    if full_runs:
        fr = pd.concat(full_runs, ignore_index=True)
        write_csv(fr, out / "full_protocol_runs.csv")
        write_csv(_summaries(fr, by=("arm", "rank", "mechanism")), out / "full_protocol_by_market.csv")
        metrics = ["revenue_strategic_norm", "welfare_strategic_norm", "served_strategic",
                   "generalization_gap", "mono_violation_rate", "comb_violation_rate"]
        pm = fr.groupby(["arm", "rank", "mechanism"])[metrics].mean().reset_index()
        across = pm.groupby(["arm", "mechanism"])[metrics].agg(["mean", "std"])
        across.columns = ["_".join(c) for c in across.columns]
        write_csv(across.reset_index(), out / "full_protocol_across_markets.csv")
        summary["full_protocol_markets"] = int(fr["rank"].nunique())
    summary["query_errors"] = {k: v for k, v in s.extra.get("query_errors", {}).items()}
    summary["tiebreak_queries"] = tb_all
    write_json(summary, out / "summary.json")
    mark_done(marker, key)
    return summary

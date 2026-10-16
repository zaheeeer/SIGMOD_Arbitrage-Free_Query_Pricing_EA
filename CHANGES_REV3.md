# Revision 3 changes (supervisor comments, October 2026)

## Part 1: Qirana fidelity (code version 3.0.0.dev1)
Comment 1; decisions D2 and D8. Work items S1.a, S1.b, S1.c.

- `pricing.qirana_prices`: the three Qirana pricing functions exactly as published
  (Deep and Koutris, SIGMOD 2017, Eq. 1 to 3): weighted coverage, uniform entropy gain
  log|C|/log|S| (new), and Shannon entropy over the partition of the support set by
  query answer.
- `qirana_gain="entropy_gain"` replaces the 2.x "uniform gain" (|C|/|S|, which was weighted
  coverage over a constant) by the published uniform entropy gain, under the key
  `qirana_uniform_entropy_gain`.
- `support.SupportSampler(neighbours="qirana")`: Qirana neighbours. The relation is drawn
  by `qirana_table_weights` ("uniform" as Qirana specifies), then a row update or a swap
  update (ratio `swap_ratio`, 1:1), each non-key attribute chosen with probability 0.5,
  new values from the active domain, primary-key and unique columns never changed,
  identity draws rejected, deterministic for any thread count.
- `stages`: when `neighbours="qirana"`, the main market and every market price with a
  Qirana support set; Table 7 of the paper gains a "qirana_updates_<weights>" row; the
  falsification test keeps deletion neighbours (Lemma 3.4 holds for any support set).
- `FIDELITY.md`: published definition -> code -> notes -> property -> test, per mechanism.
- New settings default to their 2.x values (`NEUTRAL_DEFAULTS`), so the paper and
  corrected protocols keep their checkpoint fingerprints (11cf10d670fa, c9c9df996639)
  and reproduce the published results unchanged. RESULTS_VERSION stays 2.0.0.
- Tests: 97 (68 existing + 24 in `tests/test_qirana_fidelity.py` + 5 in
  `tests/test_revision_pipeline.py`). The new tests check hand-computed prices, the
  arbitrage properties Deep and Koutris state for each function (including the bundle
  arbitrage of uniform entropy gain), and every update neighbour against an independent
  re-execution with plain SQL.

## Part 2: buyer information boundary (code version 3.0.0.dev2)
Comment 3 (and 4a); decision D3. Work items S3.b, S3.c (code side), S4.a, coverage 3e.

- New module `operational.py`: the operational buyer.
  - `parse_query`: the query fragment (one base table or alias view; plain columns or `*`,
    optional renames, optional DISTINCT; WHERE = conjunction of single-column comparisons
    with constants: =, <, <=, >, >=, IN, BETWEEN). Everything else is outside the fragment
    and is not attacked by this buyer.
  - `target_structure`: from query texts only, the sources that can supply a target (same
    base table, all target columns, and either inside the target's region or carrying the
    target's predicate columns so the buyer can filter), and the region cells each covers.
    Regions are compared exactly by atoms per column (points, open intervals, NULL,
    "other values"). DISTINCT targets may be rebuilt by a union of sources; bag targets
    only by one bag source that contains the target's region.
  - `semantic_pairs`: the query-text (schema-level) single-source relation, kept apart
    from the instance-level relation of Definition 3.2 (comment 3, last paragraph).
  - `plan_operational(prices, structures, cover_fn)`: cheapest guaranteed plan. It takes
    no answers.
  - `execute_plan(..., PurchasedAnswers)`: rebuilds the target from the purchased answers
    only; `PurchasedAnswers` refuses any other key, the target included. The verifier
    compares the result with the target answer afterwards.
- Settings: `attackers="both"` adds the operational buyer to every run (2.x default
  "oracle"); `log_plans=True` writes every accepted plan of both attackers to
  `main_market/plans_log.csv` (arm, family, seed, mechanism, attacker, target, price,
  cost, gain, sources, verified).
- New run columns: `revenue_strategic_op_norm`, `surplus_strategic_op_norm`,
  `welfare_strategic_op_norm`, `served_strategic_op`, `comb_violation_rate_op`,
  `comb_mean_gain_op`, `comb_max_gain_op`, `plans_verified_op`, `plans_failed_op`, and
  `mono_violation_rate_sem` (information-arbitrage audit over the query-text relation).
- `main_market/summary.json` gains `operational` (assets in the fragment, targets with
  sources, semantic pairs, reasons for exclusion). `main_market/pricing_support.json`
  stores the accepted support set and every conflict set, for the worked example of
  comment 4.
- The 2.x oracle planner is unchanged and still runs; it is reported as an upper bound.
- Tests: 108 (97 + 10 in `tests/test_operational.py` + 1 pipeline test). The key test
  builds plans once from query texts and checks that each one rebuilds its target on
  24 different random databases (schema-level guarantee, not instance coincidence);
  two tests check the information boundary (the planner's signature and source contain
  no answers; execution can read only purchased answers).

## Part 3: revised protocol (code version 3.0.0.dev3)
Decisions D1, D6, D7; comments 2, 5 (LP limit), 6, 8 (costs); FIDELITY for comment 1.

- `--protocol revised` (`REVISED_OVERRIDES`): the eight corrections of D1 (alias
  re-execution, answer-class entropy, exact uncapped fit, predicate selection, QueryMarket
  cover rule, any row deletable, answer cells = rows x columns, median of three timings),
  Qirana neighbours with relations drawn uniformly, the published uniform entropy gain,
  both attackers with plan logging, LPIP without a cap (`max_lps=0`), and the D7 study.
  `--protocol both` still means paper + corrected, which reproduce the published results.
- D7 test arms (`pricing.buyer_arms`): same demand, controlled shifts that permute the
  popularity ranks of 10%, 25% or 50% of the assets, and a fully redrawn ranking. Every arm
  shares the same training buyers; each run records Kendall's tau and the total-variation
  distance between training and test demand. Training sizes 100, 200, 400, 800, 1600 with
  800 test buyers. Only the main arm (same demand, 800 buyers) runs the attackers.
- D6: `lpip(max_lps=0)` solves one LP per buyer edge, as Chawla et al. specify.
- S8.c (`record_costs=True`): per run and mechanism, price-fitting seconds and planning
  seconds for each attacker (support-building seconds were already recorded).
- `revised_report`: summaries by arm, training size, family and mechanism; the 2.x
  comparison with the printed paper does not apply to this protocol.
- FIDELITY.md: Chawla et al. algorithms verified against PVLDB 13(1) Sections 3 and 5;
  QueryMarket checked against Koutris et al. PODS 2012 (instance-based determinacy),
  marked PARTIAL pending decision D10.
- New settings are neutral for paper and corrected, whose runs.csv stay byte-identical
  (the ablation end-point check passes).
- D10 = C (`querymarket="cover_semantic"`): QueryMarket prices through determinacy that
  the query texts guarantee (faithful to the instance-based determinacy of Koutris et al.);
  the instance-level version is kept as mechanism `querymarket_viewcover_instance` for the
  appendix comparison. Applies to the main market, the full-protocol markets and the
  reduced per-market runs.
- D4 (revised by Jack on 2026-10-12): the full protocol (cover planner, both attackers,
  all mechanisms) runs on all 15 markets with at least 8 assets
  (`full_protocol_markets=True`). The `full_markets` setting remains for selecting markets.
- Tests: 117 (108 + 9 in `tests/test_revised_protocol.py`, including a full
  `all --protocol revised` run on the synthetic release).

## Part 4: TPC-H under the revised protocol (code version 3.0.0.dev4)
Decisions D1, D2, D3, D6, D7, D8 applied to the second workload (comment 2f, S2.b).

- `python tpch/run_tpch.py --protocol revised [--workers N]` writes `tpch/out/revised/`.
  The default (`--protocol paper`) is the 2.x run, unchanged; the six 2.x modules in
  `tpch/modules` are byte-identical to the published artifact (checked by a test).
- New module `tpch/modules/revised_tpch.py`. It uses the verified SQLShare code
  (`sqlshare_rerun.pricing`) for every mechanism, the fit, the buyers and the Chawla et
  al. algorithms, so both workloads share one implementation.
- Qirana support (D8): row and swap updates of lineitem (the only relation the catalog
  reads; the relation is drawn uniformly among the relations the catalog reads, as for
  SQLShare), TPC-H keys (l_orderkey, l_linenumber) never changed, values from the active
  domain, unchanged draws rejected, |S| = 400. Row deletions remain as
  `--neighbours delete` (written to `tpch/out/revised_delete/`).
- Exact conflict sets and answer fingerprints (comment 1, S1.b): every asset is an
  additive aggregate over lineitem, so its answer on a neighbour is the true answer plus
  the change of the replaced rows, computed in whole cents. A group that appears or
  disappears counts as a change. Tests check every conflict set and every fingerprint
  against a re-execution of the asset's SQL on the changed database.
- Qirana prices (D2): weighted coverage, uniform entropy gain log|C|/log|S|, Shannon
  entropy over the answer partition. The 2.x `qirana_uniform_gain` (|C|/|S|) is gone.
- Exact uncapped fit, LPIP with one LP per hyperedge, answer cells = rows x columns,
  median of three timings for compute-metered prices (D1, D6).
- D7: the same arms and training sizes as SQLShare; only the main arm runs the attacker.
- D3, operational buyer on TPC-H: plans from the asset definitions and prices only;
  sources of the same template with grouping at least as fine; a source that exposes
  shipmonth may be restricted to the months it supplies, other sources are used whole and
  may not overlap; every month of the target comes from exactly one source (an integer
  program). Every plan is executed on the purchased answers only and verified against
  the target. The 2.x TPC-H planner allowed no restriction and re-executed at most 8
  plans per run. Single-source plans equal the 1,046 schema pairs (test).
- S8.c and S4: support, timing, score, fit and planning seconds; every plan logged in
  `plans_log.csv` with its sources and which ones were restricted.
- Resumable: `build.done` keeps the catalog and support; every task writes its own file
  in `tasks/`; the same command continues a stopped run. A folder that holds a run with
  other settings is refused.
- Fix (both workloads): with a training size different from the 800 test buyers, test
  revenue was divided by the training reference without scaling it to the test size
  (off by 800 / n_train in the D7 training-size runs of Part 3; no reported results were
  affected, because no revised run has been made). The reference is now scaled by
  n_test / n_train; with equal sizes the factor is exactly 1, so the paper and corrected
  protocols are unchanged. Raw revenues and the reference are also recorded.
- Tests: 140 (117 + 23 in `tests/test_tpch_revised.py`, with 1 new check in
  `tests/test_revised_protocol.py`).

## Part 5: fixes found in the P2 smoke runs (code version 3.0.0.dev5)

- Operational buyer, SQLShare (bug): on the real data every operational plan failed its
  check (467 of 467 in the main market), so the operational attack found no arbitrage. Cause:
  the pipeline stores answers with stripped, lowercased column names
  (`database.norm_frame`), while the plan used the names as written in the query texts
  (for example "Station"); the projection raised an error, which counted as a failed plan.
  The synthetic tests used lowercase names only. Fix: `operational.answer_name` maps
  output names to the stored form. Markets whose names were lowercase already had
  verified operational plans in the smoke run (cids 33, 2941, 6969, 8126).
- Plan execution keeps the purchased rows themselves: DuckDB evaluates the target's
  predicate and returns row ids only, so values and column types are not changed by a
  round trip through SQL.
- Verifier: `compare_rebuilt` gives the reason when a plan fails (logged as
  `verify_note` in plans_log.csv). When a union of sources changes a column's type (an
  integer column with NULLs in one source only), numbers are compared as numbers; such
  plans are logged as "ok (numeric types aligned)", so they can be counted separately.
- Common reference for the D7 study: `ref_monopoly_test`, the per-asset optimum on the
  test buyers (monopoly prices fitted and scored on the test sample). It does not depend
  on the training sample, so test arms and training sizes can be compared on one scale.
  (In the TPC-H smoke run the "redrawn" arm exceeded 1 against the training reference.)
- Resume (revised protocol only): every finished support set is kept in
  `work/sample_cache_revised`, including the per-market support sets, which take most of
  the markets stage (up to several hours for one market). A stopped run continues from
  the last finished support set. A reused draw is the draw a fresh run gives (the cache
  key holds every input of the sampler); paper and corrected are unchanged.
- Tests: 145 (140 + 4 in tests/test_operational.py with mixed-case names, NULLs in integer
  columns and the stored-answer form + 1 in tests/test_revised_protocol.py for resume).

## Part 6: extraction for the analysis of comments 3 and 4 (code version 3.0.0.dev6)

- New command `python -m sqlshare_rerun extract --protocol revised` (module
  `sqlshare_rerun/extract.py`), run after `all --protocol revised`. It draws no support set:
  every support set comes from `work/sample_cache_revised` (a missing one is an error;
  `_sample` gained a cache-only guard). It does not change the P3 results. Outputs in
  `results/revised/extract/`:
  - `assets.csv`: query text, owner, operational-fragment status, answer size, conflict-set
    size on the pricing support, and the scores of the deterministic mechanisms, for every
    asset of the 15 full-protocol markets;
  - `relations.csv`: every pair in the verified lattice (Definition 3.2, with its rewrite
    family) or guaranteed by the query texts, with the number of support databases that
    change the target but not the source (Qirana pricing support for every market; deletion
    curve and readership supports for the main market) (comment 3d);
  - `support_<cid>.json`: the pricing support of each market (specs and incidence);
  - `examples/example_NN.json`: worked examples for gain-1.00 plans of the main market
    (comment 4c): SQL texts, answers on D, conflict sets, scores, and for a few support
    databases the changed rows, the answers on that database and whether each answer changed.
- `SupportSampler.neighbour_answers`: applies one accepted neighbour in a transaction, runs
  the given queries, rolls back.
- Tests: 149 (145 + 4 in tests/test_extract.py: nothing new is drawn; lattice and query-text
  counts match the P3 outputs; query-text pairs are never refuted on the support; every shown
  answer change agrees with the conflict sets).

## Part 7a: query errors on support databases (code version 3.0.0.dev7)

- Finding (P6, cid 6969): one query-text pair (source SELECT *, target SELECT DISTINCT of
  two columns of the same table) had 3 Qirana neighbours that changed the target but not
  the source. Cause found in the sampler: DuckDB aborts a transaction after any error, so
  when one query that reads the changed table failed on a neighbour, every later query of
  that table also failed and was counted as unchanged. This can shrink conflict sets in
  every protocol (2.x included). A unit test reproduces it.
- Fix, off by default (paper, corrected and revised results unchanged until approved):
  `isolate_errors` applies the neighbour again in a fresh transaction after an error, so
  the remaining queries are evaluated; `error_as_change` counts a query that fails on the
  neighbour (but not on D) as changed. Both enter the support-set cache key when on.
- New command `python -m sqlshare_rerun diagnose --protocol revised`: for each of the 15
  markets, re-evaluates 40 random support databases of each support set (plus every
  database that refutes a query-text pair) with errors isolated, and compares with the
  stored conflict sets. Draws nothing new. Writes extract/error_check.csv, _summary.csv, .json.
- Also added for the P4 studies (off by default): `qirana_donors="frequency"` (study D),
  cost tracing per candidate (study B).
- Tests: 156 (149 + 6 in tests/test_error_isolation.py + 1 in tests/test_extract.py).

## Part 7b: revised protocol with error isolation and a LIMIT tie-break (code version 3.0.0.dev8)

Approved by Jack on 2026-10-14 ("1. Okay", "2. okay B1", "okay full run it is then").

- The revised preset now sets `isolate_errors=True` and `error_as_change=True` (Part 7a
  options). A query that fails on a support database no longer hides the later queries of
  that table, and a query that fails on the neighbour but not on D counts as changed.
  Found by the P7a diagnose run: errors on 3 cid 6969 databases only (7 failing queries
  each, 4 later queries lost).
- New setting `limit_tiebreak` (decision B1), on in the revised preset. Finding (P7a
  diagnose): the stored conflict sets of cid 2610 were reproduced for only 17 of 40
  databases, because ORDER BY ... LIMIT keeps an unspecified subset of tied rows, so the
  same query gave different answers on the same database. `database.tiebreak_sql` appends
  the positions of all output columns to the ORDER BY of every SELECT or set operation
  with a LIMIT (rows that tie on every output column are identical, so the answer is one
  fixed multiset). It checks that the row count is unchanged and keeps the original text
  when the rewrite fails. Queries without LIMIT are unchanged. LIMIT queries per market
  in P3: 2610: 19, 459: 18, 270: 3, 458: 2, 474: none.
- The tie-break text is used everywhere a query is evaluated: the materialized answers
  (separate cache file `<cid>_cap<rows>_tb.pkl`), the lattice, the pricing supports, the
  metered timings and the operational buyer.
- Summaries: main_market/summary.json and markets/summary.json gain `query_errors` (per
  support set: databases, databases with a query error, failing queries) and
  `tiebreak_queries` (status per rewritten query). SampleResult gains `errors`.
- `diagnose` now compares the stored sets with the rule that matches the settings
  (`stored_rule`, `stored_matches_recomputed`); after the rerun every stored set should
  be reproduced.
- The paper and corrected fingerprints are unchanged (11cf10d670fa, c9c9df996639). The
  revised fingerprint changes (it was 77d30e728d2f in P3), so a revised run recomputes the
  answers, lattice, main market and 15 markets. The P3 results stay as the comparison.
  TPC-H is not affected (exact delta engine, no SQL errors or ties) and its P3 results
  stay final.
- The support-set cache key includes a hash of the query texts when the tie-break is on,
  so a rewritten text never reuses a set drawn with the old text.
- Tests: 169 (156 + 12 in tests/test_part7b.py + 1 in tests/test_extract.py).

## Part 7c: studies A-D (code version 3.0.0.dev9)

Designs approved by Jack on 2026-10-14; run after P3b (`python -m sqlshare_rerun study
--protocol revised [--which A,B,C,D]` and `python tpch/run_tpch.py --study A,C,D --out <P3
tpch_out>`). Nothing in the main results changes; every study reads the finished runs.

- New modules: `sqlshare_rerun/study_core.py` (workload-independent pieces),
  `sqlshare_rerun/studies.py` (SQLShare driver), `tpch/modules/studies_tpch.py` (TPC-H).
  Each unit of work writes its own file, so a stopped run continues.
- Markets are rebuilt from the P3b caches (`extract._market_inputs`, cache only; it now
  also returns the clean pairs, cover structure and provenance). Study C checks the
  rebuild: its multiplier-1 rows equal the main results (all mechanisms; on the main
  market compute_metered differs, see the next point).
- Compute-metered timings: the other 14 markets and TPC-H reuse the stored P3b/P3 timings
  exactly. The main market's median-of-three timings were not stored by P3b, so the first
  study run measures them once (work/studies/metered_<cid>.pkl) and every study reuses them.
- Study A (comment 5c): six Chawla constructions; protocol in_sample / held_out; valuations
  ours (4 families) / Uniform[1,k] k in {10,100} / Zipf a in {1.5,2.5}; buyers per_query
  (one buyer per query) / popularity (800, the main protocol); LP cap none / 60; support
  Qirana updates (all 15 markets, TPC-H) and row deletions (main market readership
  deletions, TPC-H deletions); 5 seeds. Revenue normalized by the sum of valuations
  (Chawla et al.) and by the per-asset monopoly reference; rank per cell; hypergraph
  structure per workload. The (ours, popularity, no cap, held_out) cell equals the main runs.
- Study B (comment 8c): markets 474, 4930, 7635, 8126; samplers size-proportional deletions,
  readership-weighted deletions, Qirana updates; 3 sampler seeds; equal size (1,000) and
  equal budget in evaluation seconds and in query executions (each sampler at the cost of
  1,000 neighbours of each other sampler; prefixes of one traced draw, at most 20,000).
  Support quality (priceable assets, price levels, refuted lattice pairs) and downstream
  main-arm runs (4 families). LPIP in study B solves its LPs over the items used
  (`compact=True`; same optimum, tested; the main runs are unchanged).
- Study C (comment 8e): main-arm runs refitted with a price-multiplier sweep 0.10-3.00
  (step 0.01): welfare at matched revenue (30/40/50% of the reference; smallest multiplier
  reaching it) and matched coverage (25/50/75% served; largest multiplier still serving
  it); naive, oracle and operational views (attacker effective prices scale with the
  multiplier); matches at the grid edge are flagged and left out of the means.
- Study D: Qirana support with frequency-weighted donors (value copied from a random row),
  same tables, seeds and size; main market and TPC-H (new TpchSettings.qirana_donors; the
  P3 TPC-H fingerprints ec79407bc3e0 and dd366a9cb107 are unchanged).
- Worked-example files show their rows sorted (display only).
- Tests: test_studies.py (7), test_studies_tpch.py (4).

## Part 7d: the analysis of the revised paper (code version 3.0.0.dev10)

Approved by Jack on 2026-10-15 (one laptop run for everything that remains).

- New command `python -m sqlshare_rerun analyze --protocol revised [--tpch <TPC-H out>]
  [--tpch-studies <dir>]` (module `analysis.py`). Reads finished results only and writes
  results/revised/analysis/: funnel 975 -> 806 -> 812 with exclusions and the lattice
  family shares (comment 7); main results per workload; paired bootstrap 95% intervals
  (10,000 resamples, seed 20261015; units (family, seed) on one market and markets across
  markets) for every mechanism against uniform pricing, LPIP against UBP, and the oracle
  against the operational attacker, with the number of settings in which each comparison
  holds (comment 8a/8b); per-family, test-arm and training-size tables (comment 6); costs
  of support building, price fitting and planning (comment 8d); plans with gain 1.00 by
  cause (comment 4b); the three relations per market (comment 3d); study A-D tables;
  paper_numbers.json. Workloads: sqlshare_main, tpch, sqlshare_15, sqlshare_14 (without
  cid 2610).
- Tests: test_studies.py (+2), test_studies_tpch.py (+1).

## Part 7e: fixes after the P7c laptop run (code version 3.0.0.dev11)

- `analyze` stopped with "No module named 'probe_tpch_lattice'": it runs from the sqlshare
  folder, and unpickling the TPC-H market.pkl needs tpch/modules on the path (the tests
  had that path set, so they missed it). Fixed in analysis.causes; a new test runs the
  step in a fresh process from the sqlshare folder and fails without the fix.
- Study A normalization: the held-out `revenue_over_monopoly` divided by the monopoly
  revenue on the evaluation buyers, which is close to 0 with Zipf(1.5) values (ratios up to
  -234 in the LPIP minus UBP contrast). The approved choice was "our usual reference": the
  new `revenue_norm_main` divides by the monopoly revenue on the training buyers of the same
  cell, as the main protocol does (study_core.add_main_norm). The analysis reports LPIP
  minus UBP on `revenue_over_values` (Chawla et al.) and on `revenue_norm_main`; the study
  A summary ranks by `revenue_norm_main`. No study result changes (the raw revenues and
  references are in study_A_runs.csv).
- Observed, no code change: in one TPC-H run (rows_x_log_support, seed 2) the LPIP, CIP and
  XOS prices of studies A and C differ slightly from the P3 run (normalized test revenue
  0.4617 vs 0.4622 for LPIP); the inputs are identical, so the LP solver returned another
  optimal solution of a degenerate LP. All other TPC-H runs and all SQLShare runs match.

## Part 8: tests, paper tables, and the revision-3 artifact (2026-10-16)

- `sqlshare/tests/test_chawla_hand.py` (6 tests): prices computed by hand on a four-buyer,
  three-item hypergraph for UBP, UIP, LPIP, and layering; CIP by its dual properties; XOS as
  the maximum of LPIP and CIP. No pricing code changed. 190 tests pass on the run machine
  (Windows 11, Python 3.12.10; 10.1 minutes) with every tracked file matching git.
- `reproduce/make_paper_tables.py` writes every table with measured values of the revised
  paper and appendix from `results/revised/` and compares each one with
  `reproduce/expected_tables/` (21 of 21 match).
- `results/revised/`: the result records of the final runs (SQLShare protocol, TPC-H with
  both supports, extract, studies A-D, analysis), as returned from the run machine;
  `sqlshare_studies/C/study_C_curves.csv` is stored gzip-compressed.
- README.md rewritten for revision 3; `sqlshare/FIDELITY.md` lists the test of every
  revenue-maximizing port.

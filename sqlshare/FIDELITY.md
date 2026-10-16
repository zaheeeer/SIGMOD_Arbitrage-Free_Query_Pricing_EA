# Fidelity: published pricing definitions mapped to this code

Revision 3 (supervisor comment 1, decisions D2 and D8; Part 8 added hand-computed tests for
the Chawla et al. ports, `tests/test_chawla_hand.py`). Each row gives the published
definition, the function that implements it, how the implementation differs, and the test
that checks it. Status "VERIFIED" means the formula was checked against the source paper
and a hand-computed test passes. "PENDING" rows are verified in a later part of revision 3.

Sources:
- [DK17] Deep and Koutris, "QIRANA: A Framework for Scalable Query Pricing", SIGMOD 2017,
  Section 2.3 (pricing functions), Table 1 (arbitrage properties), Section 4 (support sets).
  doi:10.1145/3035918.3064017
- [CDKS19] Chawla, Deep, Koutris, Teng, "Revenue Maximization for Query Pricing",
  PVLDB 13(1), Sections 5 and 6. doi:10.14778/3357377.3357378
- [KUBHS15] Koutris, Upadhyaya, Balazinska, Howe, Suciu, "Query-Based Data Pricing",
  J. ACM 62(5). doi:10.1145/2770870

Notation: S is the support set, C(Q) = {D' in S : Q(D') != Q(D)} the conflict set,
P_Q the partition of S by query answer, w_i the support weights.

## Qirana [DK17]

| Mechanism key | Published definition | Code | Implementation notes | Arbitrage property [DK17] | Test | Status |
|---|---|---|---|---|---|---|
| `qirana_weighted_coverage` | p_wc = sum over i in C(Q) of w_i (Eq. 1); default w_i = P/\|S\| | `pricing.qirana_prices` (and the 2.x `qirana_from_edges`, same values) | unit weights; the seller's scale P is fitted afterwards, so a common factor does not change prices after the fit | no information or bundle arbitrage | `test_hand_example_prices`, `test_weights_are_used_by_weighted_coverage`, `test_weighted_coverage_has_no_information_or_bundle_arbitrage` | VERIFIED |
| `qirana_uniform_entropy_gain` (new in revision 3) | p_ueg = log\|C(Q) ∩ S\| / log\|S\| (Eq. 2) | `pricing.qirana_prices` | \|C\| = 0 and \|C\| = 1 give price 0, as the formula implies | no information arbitrage; **does exhibit bundle arbitrage** | `test_hand_example_prices`, `test_uniform_entropy_gain_has_no_information_arbitrage`, `test_uniform_entropy_gain_exhibits_bundle_arbitrage` | VERIFIED |
| `qirana_uniform_gain` (2.x only, retired) | none: the 2.x code computed \|C\|/\|S\| | `pricing.qirana_from_edges` | weighted coverage divided by a constant; after the scale fit it priced exactly like weighted coverage. Kept only to reproduce 2.x results (`qirana_gain="fraction"`) | n/a | `test_historical_uniform_gain_was_coverage_over_a_constant` | RETIRED |
| `qirana_shannon` | p_H = -sum over B in P_Q of w_B log w_B (Eq. 3), sum of w_i = 1 | `pricing.exact_entropy` via `pricing.qirana_prices` (`entropy="exact"`) | w_i = 1/\|S\|; log base 2 (a common factor, removed by the scale fit). Blocks are formed from the actual answer fingerprints, so neighbours with equal answers share a block (comment 1). The 2.x default (`entropy="singleton"`) made every changed neighbour its own block | text: information and bundle arbitrage-free; Table 1: **weakly** information arbitrage-free | `test_hand_example_prices`, `test_exact_entropy_merges_equal_answers`, `test_shannon_entropy_is_monotone_and_subadditive_on_answer_partitions` | VERIFIED |
| q-entropy (Eq. 4) | p_T = sum over B of w_B (1 - w_B) | not implemented | not among the evaluated constructions | | | NOT EVALUATED |

### Support set [DK17, Section 4] (decision D8)

| Element | Published definition | Code (`support.SupportSampler`, `neighbours="qirana"`) | Notes | Test |
|---|---|---|---|---|
| Relation choice | chosen uniformly at random | `qirana_table_weights="uniform"` (also "readership" and "size" for comparison) | | `test_uniform_table_draw_follows_the_weights` |
| Row update | one tuple; each attribute not in the primary key chosen with probability 0.5; new value from the attribute's domain, else its active domain | `draw_update` ("row"), `_evaluate_update` | SQLShare declares no domains, so the active domain is used, each distinct value equally likely; at least one attribute is chosen; PRIMARY KEY and UNIQUE columns are excluded (SQLShare uploads declare none, so every column is eligible there) | `test_every_neighbour_changes_the_database_and_keys_are_never_changed` |
| Swap update | two tuples of one relation exchange the values of the chosen attributes | `draw_update` ("swap"), `_evaluate_update` | | same |
| Row : swap ratio | 1:1 in the experiments | `swap_ratio=0.5` | | same (both kinds occur) |
| Neighbour equal to D | (not discussed) | rejected and redrawn | a draw that changes no value is not a neighbour | same (every neighbour changes SELECT *) |
| Conflict set | {D' in S : Q(D') != Q(D)} | `_changed_answers`: every query reading the relation is re-run inside the transaction, compared by an order-insensitive answer fingerprint, then rolled back | alias views resolve to their base table (`alias_delete="resolve"`) | `test_update_conflict_sets_match_an_independent_reexecution` |
| Reproducibility | | every draw made in the calling thread; accepted set independent of thread count | | `test_update_neighbours_are_deterministic_and_rolled_back` |

2.x deviation now removed from the Qirana prices: neighbours were single-row deletions.
Deletions remain available (`neighbours="delete"`) and are used for the falsification test,
where Lemma 3.4 holds for any support set.

## QueryMarket [KUBHS15]

Checked against Koutris et al., "Query-Based Data Pricing", PODS 2012 (the conference
version of [KUBHS15]): the arbitrage-price of Q is the price of the cheapest set of priced
views C that determines Q (Definition 2.13), and determinacy is INSTANCE-BASED
(Definition 2.2): D ⊢ V ↠ Q if for every D', V(D) = V(D') implies Q(D) = Q(D'). This is
stronger than the instance-level derivability of our Definition 3.2 (a rewrite that
matches on D alone) and weaker than query determinacy over all instances.

| Mechanism key | Published definition | Code | Notes | Status |
|---|---|---|---|---|
| `querymarket_viewcover` | p(Q) = min over C ⊆ priced views with D ⊢ C ↠ Q of the sum of view prices | `pricing.querymarket_cover` fed with the query-text relation (`querymarket="cover_semantic"`, revision 3, D10 = C) | TPC-H: per-template, per-month views, determinacy by a schema-level rule (exact). SQLShare: catalog = maximal assets (our substitution, marked †); determinacy only through single sources and union covers guaranteed by the query texts (`operational.semantic_pairs`, `target_structure`), which hold on every database and therefore satisfy instance-based determinacy | VERIFIED (catalog †) |
| `querymarket_viewcover_instance` (comparison, appendix) | same formula | `pricing.querymarket_cover` fed with the instance-level relation of Definition 3.2 and answer-row covers (the 2.x "cover" rule) | weaker than instance-based determinacy; kept to show how much the looser rule changes the results | COMPARISON |

## Revenue-maximizing algorithms [CDKS19]

Checked against Chawla et al., PVLDB 13(1), Sections 3 and 5 (2026-10-12). Hypergraph:
items = support databases, one hyperedge per buyer e_i = C_S(Q_i), single-minded buyers,
unlimited supply, revenue = sum of p(e_i) over buyers with p(e_i) <= v_i. Identical
hyperedges are not merged (each buyer is its own edge), as in our code.

| Mechanism key | Published definition | Code | Notes | Test | Status |
|---|---|---|---|---|---|
| `chawla_ubp` | for each edge, price every bundle at P = v_e; keep the best revenue | `pricing.ubp` | | `test_chawla_hand.py::test_ubp_by_hand`, `test_units.py::test_flat_fit_equals_uniform_bundle_price` | VERIFIED |
| `chawla_uip` | for each edge, every item weight w_j = v_e/\|e\|; keep the best revenue | `pricing.uip` | | `test_chawla_hand.py::test_uip_by_hand`, `test_units.py::test_uip_fast_path_equals_legacy` | VERIFIED |
| `chawla_lpip` | for each edge e, LP(e) over F_e = {e' : v_e' >= v_e}: maximize revenue s.t. sum_{j in e'} w_j <= v_e' for e' in F_e, w >= 0; keep the best | `pricing.lpip` | objective = sum over e' in F_e of p(e'), which is the revenue when every edge of F_e sells, as the constraints force. 2.x solved at most 60 LPs (25 per market) at evenly spaced thresholds; revision 3 solves one LP per edge (`max_lps=0`, D6) | `test_chawla_hand.py::test_lpip_by_hand`, `test_revised_protocol.py::test_lpip_uncapped_solves_one_lp_per_buyer_and_never_loses_revenue` | VERIFIED (uncapped) |
| `chawla_cip` | welfare LP with uniform item capacity k; item prices from its dual; k = 1, (1+ε), (1+ε)^2, ...; keep the best | `pricing.cip` | ε = 0.5 (the paper used 0.2 to 4 per workload to limit runtime; a smaller ε searches more capacities) | `test_chawla_hand.py::test_cip_prices_are_dual_feasible_by_hand` (property), `test_units.py::test_cip_vectorized_equals_legacy` | VERIFIED |
| `chawla_layering` | Algorithm 1: repeatedly take a minimal set cover of the remaining edges' items as a layer; keep the layer with the largest total value; price one unique item of each of its edges at the edge's value | `pricing.layering` | greedy cover then pruning gives a minimal cover; the unique item chosen is the smallest index | `test_chawla_hand.py::test_layering_by_hand` | VERIFIED |
| `chawla_xos` | p(e) = max of the LPIP and CIP bundle prices (k = 2 additive functions) | `pricing.chawla_all` | | `test_chawla_hand.py::test_xos_takes_the_larger_bundle_price_by_hand`, `test_units.py::test_xos_reuses_lpip_and_cip` | VERIFIED |

## TPC-H workload (revision 3, Part 4)

`tpch/run_tpch.py --protocol revised` (module `tpch/modules/revised_tpch.py`) prices the
368 TPC-H assets with the same functions as the rows above (`sqlshare_rerun.pricing`).
Only the inputs that depend on the workload are computed differently:

| Element | Published definition | Code | Notes | Test (`tests/test_tpch_revised.py`) |
|---|---|---|---|---|
| Relation choice [DK17] | uniform over relations | `CATALOG_RELATIONS = ("lineitem",)` | the 368 assets read only lineitem; as for SQLShare, the draw is over the relations the catalog reads | `test_qirana_neighbours_follow_the_published_rules` |
| Row / swap update [DK17] | as above | `LineitemNeighbours.draw`, `.changes` | same drawing rule as `support.SupportSampler`; TPC-H primary key (l_orderkey, l_linenumber) excluded; active domain of each column | same |
| Conflict set [DK17] | {D' in S : Q(D') != Q(D)} | `DeltaEngine.neighbour` | exact: answer change computed from the replaced rows in whole cents (assets are sums and counts over lineitem) | `test_conflict_sets_and_fingerprints_match_reexecution` |
| Answer partition P_Q (Shannon) [DK17] | blocks of S by answer | `DeltaEngine.asset_change` fingerprints | equal fingerprints exactly when equal answers | same, `test_answer_partition_gives_shannon_prices` |
| QueryMarket [KUBHS15] | cheapest determining set of priced views | `ports.querymarket_prices` (unchanged 2.x code) | views = per-month views at full grouping, priced by base rows read; determinacy by the schema rule (month union of additive aggregates), which is instance-based | |
| Operational buyer (D3) | | `op_structures`, `plan_target`, `execute_plan` | plans from asset definitions and prices; restriction on a visible shipmonth key; executed on purchased answers only | `test_operational_plans_hold_on_other_instances_and_are_cheapest`, `test_planner_never_sees_answers`, `test_single_source_plans_are_the_schema_pairs` |

"""Run settings for the two protocols.

``paper`` is the setting of the main results (the middle column of Table 2 of the paper).
``corrected`` switches on every alternative of Table 2 (the right column). The labels
C04 to C15 in the comments and switch names identify these choices. Every difference
between the two is a field below, so a reader can see exactly what changes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import RESULTS_VERSION

LEGACY_PROTOCOLS = ("paper", "corrected")
PROTOCOLS = LEGACY_PROTOCOLS + ("revised",)

#: Owners whose whole folder the historical run extracted before selecting markets.
PILOT_OWNERS = ("1059", "1314howe", "446")


@dataclass
class Settings:
    protocol: str = "paper"

    # ---- locations ------------------------------------------------------------------
    root: Path = Path(".")
    zip_url: str = "https://shrquerylogs.s3.amazonaws.com/public/sqlshare_data_release1.zip"
    zip_path: Path | None = None          # default: <root>/data/sqlshare_data_release1.zip

    # ---- corpus and market selection (identical in both protocols) ---------------------
    min_component_queries: int = 50       # component kept if it has at least this many queries
    max_component_gb: float = 2.0         # ... and its base tables fit under this footprint
    min_lattice_survivors: int = 20       # structural lattice uses components with >= this
    row_cap: int = 50_000                 # answers above this many rows are not materialized
    cover_target_cap: int = 2_000         # cover planner targets at most this many rows
    query_timeout_s: float = 3_600.0      # watchdog for a single bind/execute call

    # ---- main market (largest component) ----------------------------------------------
    curve_checkpoints: tuple = (400, 2_000, 10_000, 30_000)
    curve_seed: int = 11
    witness_n: int = 4_000
    witness_seed: int = 11
    table6_n: int = 1_000
    table6_seed: int = 101
    run_seeds: tuple = (0, 1, 2, 3, 4)
    n_buyers: int = 800
    max_lps: int = 60
    cip_eps: float = 0.5
    ilp_time_limit: float = 5.0

    # ---- all markets (Table 7, Figure 2b) ----------------------------------------------
    market_support_n: int = 300
    market_seed: int = 4242
    market_buyers: int = 400
    market_seeds: tuple = (0, 1, 2)
    market_families: tuple = ("support_rows", "answer_cells")
    market_max_lps: int = 25
    market_min_assets: int = 8

    # ---- protocol switches (paper value first, corrected value in comments) -----------
    demand: str = "redraw"                # C04: "shared" keeps one popularity ranking
    shift_arm: bool = False               # C04: also report a demand-shift arm
    fit: str = "grid60"                   # C05: "breakpoints" = exact, uncapped scale fit
    entropy: str = "singleton"            # C06: "exact" = entropy over answer classes
    selection: str = "value_set"          # C07: "predicate" = buyer-side WHERE rewrite
    querymarket: str = "single_predecessor"  # C09: "cover" = min-cost view cover
    answer_cells: str = "rows"            # C11: "rows_x_cols" uses the answer width
    market_support_family: str = "rows"   # C11: "provenance" = base rows read
    row_sampling: str = "unique_only"     # C12: "any_row" deletes any physical row
    remove_falsified: str = "curve"       # "all" also drops edges the pricing support refutes
    alias_delete: str = "skip"            # "resolve" maps an alias view to its base table
    market_tables: str = "all_readers"    # "nonempty": market samplers skip empty or unloaded tables
    metered_repeats: int = 1              # "3" = median of three warm timings
    report_sd: str = "pooled"             # C14: "across_markets" for Table 7
    full_protocol_markets: bool = False   # C15: full protocol on every market
    ilp_formulation: str = ""             # "" = as the protocol ("all_sources" in paper,
                                          # "cheaper_sources" in corrected); ablation switch

    # ---- revision 3 (supervisor comments, 2026-10) : Qirana fidelity -----------------
    neighbours: str = "delete"            # "qirana": row and swap updates as in Deep and
                                          # Koutris (SIGMOD 2017), relation drawn per
                                          # qirana_table_weights, primary keys never changed
    qirana_table_weights: str = "readership"  # table draw for the Qirana support set when
                                          # neighbours == "qirana": "uniform" (as Qirana
                                          # specifies), "readership", or "size"
    swap_ratio: float = 0.5               # share of swap updates among Qirana neighbours (1:1)
    qirana_gain: str = "fraction"         # "entropy_gain": Qirana uniform entropy gain
                                          # log|C|/log|S|; "fraction" = historical |C|/|S|
    attackers: str = "oracle"             # "both": also the operational buyer (D3), which
                                          # plans from query texts and prices only
    log_plans: bool = False               # write every accepted plan to plans_log.csv
    demand_arms: tuple = ()               # demand == "arms": test arms of the D7 study
    main_arm: str = "same"                # the arm the main tables report
    train_sizes: tuple = ()               # extra training sizes for the D7 study (main arm)
    record_costs: bool = False            # revision 3 (S8.c): fit and planning seconds per run
    full_markets: tuple = ()              # revision 3 (D4): cids that get the full protocol
    isolate_errors: bool = False          # revision 3 (Part 7 fix): after a query error on a
                                          # neighbour, apply the neighbour again in a fresh
                                          # transaction so the remaining queries are evaluated
                                          # (DuckDB aborts the transaction on any error)
    error_as_change: bool = False         # with isolate_errors: a query that fails on the
                                          # neighbour but not on D counts as changed
    qirana_donors: str = "distinct"       # study D: "frequency" draws new values in proportion
                                          # to their frequency in the column
    limit_tiebreak: bool = False          # revision 3 (Part 7b, B1): append the output columns
                                          # to ORDER BY in every SELECT with LIMIT, so ties at
                                          # the cut-off no longer make the answer random

    # ---- resources --------------------------------------------------------------------
    workers: int = 0                      # processes for lattice and runs (0 = cores - 1)
    threads: int = 0                      # sampling threads (0 = cores)
    duckdb_threads: int = 0               # DuckDB worker threads (0 = DuckDB default)
    memory_limit: str = ""                # e.g. "8GB"; empty = DuckDB default
    pool_stall_s: float = 1800.0          # a worker pool with no finished task for this long is
                                          # replaced by this process (same results, slower)
    quick: bool = False                   # scaled-down run for a fast end-to-end check

    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def results_dir(self) -> Path:
        if self.extra.get("results_dir"):          # ablation runs write elsewhere
            return Path(self.extra["results_dir"])
        return self.root / "results" / self.protocol

    @property
    def lattice_dir(self) -> Path:
        if self.extra.get("lattice_dir"):          # ablation runs share one lattice per mode
            return Path(self.extra["lattice_dir"])
        return self.results_dir / "lattice"

    def exact_formulation(self) -> bool:
        if self.ilp_formulation:
            return self.ilp_formulation == "all_sources"
        return self.protocol == "paper"

    @property
    def zip_file(self) -> Path:
        return Path(self.zip_path) if self.zip_path else self.data_dir / "sqlshare_data_release1.zip"

    @property
    def db_path(self) -> Path:
        return self.work_dir / "sqlshare2.duckdb"   # the historical file name: same catalog name

    def n_workers(self) -> int:
        if self.workers and self.workers > 0:
            return self.workers
        return max(1, (os.cpu_count() or 2) - 1)

    def n_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        return max(1, os.cpu_count() or 2)

    def fingerprint(self) -> str:
        """Hash of every setting that changes results, for checkpoint validity."""
        skip = {"root", "zip_path", "workers", "threads", "duckdb_threads", "memory_limit",
                "query_timeout_s", "pool_stall_s", "extra"}
        d = {k: v for k, v in dataclasses.asdict(self).items() if k not in skip}
        for k, neutral in NEUTRAL_DEFAULTS.items():   # fields added later: at their neutral
            if d.get(k) == neutral:                   # value they leave earlier keys unchanged
                d.pop(k, None)
        d["code_version"] = RESULTS_VERSION
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["root"] = str(self.root)
        d["zip_path"] = str(self.zip_file)
        return d


#: Settings added after 2.0.0, with the value that reproduces 2.0.0 behavior exactly.
NEUTRAL_DEFAULTS = {"ilp_formulation": "", "neighbours": "delete", "qirana_table_weights": "readership",
                    "swap_ratio": 0.5, "qirana_gain": "fraction", "attackers": "oracle",
                    "log_plans": False, "demand_arms": (), "main_arm": "same", "train_sizes": (),
                    "record_costs": False, "full_markets": (), "isolate_errors": False,
                    "error_as_change": False, "qirana_donors": "distinct",
                    "limit_tiebreak": False}

#: Revision 3 main protocol (decisions D1, D2, D3, D6, D7, D8). The eight corrections of D1
#: (alias re-execution, answer-class entropy, exact uncapped fit, predicate selection,
#: QueryMarket cover rule, any row deletable, answer cells = rows x columns, median of three
#: timings), Qirana neighbours with relations drawn uniformly, the published uniform
#: entropy gain, both attackers with plan logging, LPIP without a cap, and the three test
#: arms with the training-size study. Refuted edges stay in the relation (removing them is
#: an analysis option, D1).
REVISED_OVERRIDES = dict(
    alias_delete="resolve",
    entropy="exact",
    fit="breakpoints",
    selection="predicate",
    querymarket="cover_semantic",     # D10 = C: query-text determinacy; instance-level as comparison
    row_sampling="any_row",
    answer_cells="rows_x_cols",
    market_support_family="provenance",
    metered_repeats=3,
    market_tables="nonempty",
    report_sd="across_markets",
    witness_seed=12,
    remove_falsified="curve",
    neighbours="qirana",
    qirana_table_weights="uniform",
    qirana_gain="entropy_gain",
    attackers="both",
    log_plans=True,
    record_costs=True,
    max_lps=0,
    market_max_lps=0,
    demand="arms",
    demand_arms=("same", "shift10", "shift25", "shift50", "redrawn"),
    main_arm="same",
    train_sizes=(100, 200, 400, 800, 1600),
    shift_arm=False,
    full_protocol_markets=True,       # D4 (revised 2026-10-12): full protocol on all 15 markets
    isolate_errors=True,              # Part 7b: a failing query no longer hides later ones
    error_as_change=True,             # a query that fails on a neighbour (not on D) changed
    limit_tiebreak=True,              # B1: one answer per database for top-k queries
)

CORRECTED_OVERRIDES = dict(
    demand="shared",
    shift_arm=True,
    fit="breakpoints",
    entropy="exact",
    selection="predicate",
    querymarket="cover",
    answer_cells="rows_x_cols",
    market_support_family="provenance",
    row_sampling="any_row",
    remove_falsified="all",
    alias_delete="resolve",
    market_tables="nonempty",
    metered_repeats=3,
    report_sd="across_markets",
    full_protocol_markets=True,
    witness_seed=12,          # the paper calls the witness sample independent
)

QUICK_OVERRIDES = dict(
    curve_checkpoints=(400, 2_000),
    witness_n=1_000,
    table6_n=300,
    run_seeds=(0, 1),
    market_support_n=100,
    market_seeds=(0,),
)


def make_settings(protocol: str, root: Path | str = ".", quick: bool = False, **kw) -> Settings:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    s = Settings(protocol=protocol, root=Path(root), quick=quick)
    if protocol == "corrected":
        for k, v in CORRECTED_OVERRIDES.items():
            setattr(s, k, v)
    if protocol == "revised":
        for k, v in REVISED_OVERRIDES.items():
            setattr(s, k, v)
    if quick:
        for k, v in QUICK_OVERRIDES.items():
            setattr(s, k, v)
    for k, v in kw.items():
        if v is None:
            continue
        if not hasattr(s, k):
            raise ValueError(f"unknown setting {k}")
        setattr(s, k, v)
    return s

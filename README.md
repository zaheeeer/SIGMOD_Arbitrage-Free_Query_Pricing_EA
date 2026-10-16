# Arbitrage-Free Query Pricing under Recomposition: artifact (revision 3)

Code, tests, result records, and reproduction scripts for the paper *Arbitrage-Free Query
Pricing under Recomposition: An Experimental Evaluation of Published Constructions:
[Experiments & Analysis]* (SIGMOD 2027 submission, revision 3).

The paper evaluates ten published query-pricing constructions from three systems
(QueryMarket view-cover pricing, the three Qirana conflict-set functions, and the six
revenue-maximizing algorithms of Chawla et al.) and four baselines, on TPC-H and on 15
markets of SQLShare queries. It separates three relations between assets, two attackers
(an operational buyer that sees only query texts, prices, and purchased answers, and a
privileged attacker that reads the target answer), and three test arms, and it reports
four studies (A: the Chawla et al. factorial; B: support samplers at equal size and equal
cost; C: welfare at matched revenue and coverage; D: the value rule of Qirana updates).

## Contents

| folder | what it holds |
|---|---|
| `results/revised/` | every result record behind the paper and its appendix (revision 3), as returned from the run machine; see "Records" below |
| `reproduce/make_paper_tables.py` | writes every table with measured values from `results/revised/` and checks each one against `reproduce/expected_tables/`, the copies the paper and appendix print |
| `sqlshare/` | the SQLShare pipeline (`python -m sqlshare_rerun`), its 190 tests, and `FIDELITY.md` (published definition -> code -> test for every construction) |
| `tpch/` | the TPC-H campaign (`run_tpch.py`) and its modules |
| `results/sqlshare/`, `results/tpch/`, `reproduce/make_tables.py`, `reproduce/make_appendix_tables.py` | records and scripts of the earlier version 2.x evaluation; the paper uses them only for the failure cases of appendix Table A12 (`results/sqlshare/ablation/`) |
| `CHANGES_REV3.md` | every change of revision 3, part by part |

## Quick check: rebuild the paper's tables from the records (about a minute)

```bash
pip install -r requirements.txt
python reproduce/make_paper_tables.py
```

The script writes the table files to `reproduce/out/tables/` and ends with
`Tables: 21 of 21 match the paper and appendix`. It exits with status 1 if any table
differs. Tables 1, 3, A1, and A12 hold no values computed from these records (notation, the
fidelity mapping, the commands, and the version 2.x failure cases).

## Records (`results/revised/`)

| folder | written by | paper |
|---|---|---|
| `sqlshare_revised/` (`lattice/`, `main_market/`, `markets/`, `settings.json`) | `python -m sqlshare_rerun all --protocol revised` | runs behind Tables 5-9 and 12; main-market plans (`plans_log.csv`) and pricing support (`pricing_support.json`) |
| `tpch_revised/`, `tpch_revised_delete/` | `python tpch/run_tpch.py --protocol revised [--neighbours delete]` | TPC-H runs (Qirana updates; row deletions for comparison) |
| `sqlshare_extract/` | `python -m sqlshare_rerun extract --protocol revised` | Table 4 (relations), worked examples (`examples/`), conflict-set error check |
| `sqlshare_studies/` (`A/`-`D/`), `tpch_studies/` (`A/`, `C/`, `D/`) | `python -m sqlshare_rerun study --protocol revised --which A,B,C,D`; `python tpch/run_tpch.py --study A,C,D` | Tables 10, 11, 13, A7-A11; `C/study_C_curves.csv.gz` is the full price sweep (gzip) |
| `analysis/` | `python -m sqlshare_rerun analyze --protocol revised --tpch <tpch> --tpch-studies <tpch studies>` | Tables 2, 4-9, 12, A3-A6; paired bootstrap intervals (`comparisons.csv`); `paper_numbers.json` |

The run machine wrote the SQLShare records under `<root>/results/revised/` (lattice,
main_market, markets, extract, studies, analysis); this folder groups them by workload.

## Rerunning the experiments

Python 3.12 with the versions in `requirements.txt` (the runs used Python 3.12.10, DuckDB
1.5.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 with HiGHS, sqlglot 30.18.0) on Windows 11,
an Intel Core i7-9850H (6 cores, 12 threads), and 31.7 GB of memory.

```bash
# tests (about 10 minutes): 190 tests, including hand-computed prices for the Qirana
# functions, UBP, UIP, LPIP, and layering
cd sqlshare && python -m pytest -q

# SQLShare: downloads the public release (3.6 GB), needs about 15 GB of disk
python -m sqlshare_rerun all     --protocol revised --root <root> --zip <release zip> --workers 11
python -m sqlshare_rerun extract --protocol revised --root <root> --zip <release zip>
python -m sqlshare_rerun study   --protocol revised --which A,B,C,D --root <root> --zip <release zip> --workers 11

# TPC-H (generated at scale factor 1 by DuckDB's tpch extension)
cd ../tpch
python run_tpch.py --protocol revised --workers 6 --out <tpch>
python run_tpch.py --protocol revised --neighbours delete --workers 6 --out <tpch>
python run_tpch.py --study A,C,D --out <tpch> --study-out <tpch studies>

# analysis: intervals, funnel, causes, costs, study tables
cd ../sqlshare
python -m sqlshare_rerun analyze --protocol revised --root <root> --zip <release zip> --tpch <tpch> --tpch-studies <tpch studies>
```

Run times on the machine above: SQLShare protocol 24.2 hours, TPC-H 30.7 minutes with both
supports, studies 18.9 hours (SQLShare) and 15 minutes (TPC-H). Every stage resumes where it
stopped when the same command is given again. Every random draw is seeded; the tests check
that results do not depend on the number of threads or worker processes. Compute-metered
prices come from measured query times, so that mechanism's numbers depend on the machine
(a repeat on the main market moved its normalized revenue by up to 0.017).

## Fidelity and the information boundary

- `sqlshare/FIDELITY.md` maps every construction to its published definition, the function
  that implements it, and the tests (`tests/test_qirana_fidelity.py`,
  `tests/test_chawla_hand.py`, `tests/test_units.py`, `tests/test_tpch_revised.py`).
- The operational buyer (`sqlshare_rerun/operational.py`): `plan_operational` receives
  prices and parsed query texts only; `execute_plan` reads answers through
  `PurchasedAnswers`, which refuses any asset that was not bought.
  `tests/test_operational.py` fails if either boundary is crossed.

## Data

- TPC-H is generated by DuckDB's `tpch` extension at scale factor 1; the specification and
  generator are available from https://www.tpc.org.
- SQLShare: Jain et al., SIGMOD 2016 (doi:10.1145/2882903.2882957). The release
  (https://uwescience.github.io/sqlshare/data_release.html) is downloaded by the pipeline
  and is not redistributed here.

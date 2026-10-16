"""Write the data tables of the appendix from the result records in results/.

    python reproduce/make_appendix_tables.py [results_dir]

Writes reproduce/out/tab_app_sens_setting.tex, tab_app_sens_alt.tex, tab_app_markets.tex,
tab_app_family_revenue.tex and tab_app_family_welfare.tex, and prints the facts the
appendix text quotes. No value is typed in by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

MECH = [("flat fee (= UBP)", "uniform"), ("size proportional", "size_proportional"),
        ("compute metered", "compute_metered"), ("per-asset monopoly", "monopoly_per_asset"),
        ("monotone fitted", "monotone_constrained"),
        (r"\qirana{} weighted coverage", "qirana_weighted_coverage"),
        (r"\qirana{} uniform gain", "qirana_uniform_gain"), (r"\qirana{} entropy", "qirana_shannon"),
        (r"\querymarket{} view cover$^{\dagger}$", "querymarket_viewcover"),
        ("uniform bundle (UBP)", "chawla_ubp"), ("uniform item (UIP)", "chawla_uip"),
        ("LP item (LPIP)", "chawla_lpip"), ("capacity item (CIP)", "chawla_cip"),
        ("layering", "chawla_layering"), ("XOS", "chawla_xos")]
TEN = [m for _, m in MECH[5:]]
ALL = [m for _, m in MECH]
CHOICE = {"C04_demand": "test demand", "C05_fit": "scale fit", "C06_entropy": "entropy classes",
          "C07_selection": "selection rewrite", "C09_querymarket": r"\querymarket{} rule",
          "C11_answer_cells": "answer cells", "C12_row_sampling": "deletable rows",
          "alias_resolve": "re-executed queries", "remove_all_falsified": "edges removed",
          "metered_x3": "metered time", "witness_seed": "witness seed (check)",
          "ilp_formulation": "cover formulation (check)"}
FAMILIES = [("answer_cells", "cells"), ("support_rows", "support"),
            ("rows_x_log_support", r"rows$\cdot$log"), ("flat", "flat")]


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def fmt(x, d):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


# ---------------------------------------------------------------- sensitivity, 30 configurations
long = pd.read_csv(os.path.join(RES, "sqlshare", "ablation", "ablation_long.csv"))
ab = long.pivot_table(index="configuration", columns="quantity", values="value", aggfunc="first")
ab = ab.apply(pd.to_numeric, errors="coerce")
rev = lambda c, m: ab.loc[c, "revenue:" + m]   # noqa: E731
wel = lambda c, m: ab.loc[c, "welfare:" + m]   # noqa: E731


def spearman(c):
    r = pd.Series([rev(c, m) for m in ALL]).rank().to_numpy()
    w = pd.Series([wel(c, m) for m in ALL]).rank().to_numpy()
    return float(np.corrcoef(r, w)[0, 1])


COLS = [("claimed", "claimed_edges", 0), ("refuted", "false_edges_total", 0),
        ("zero", "zero_priced_readership", 0), ("priced", "t6_priceable_read", 0),
        ("cap", "alpha_at_grid_max_share", 3), ("neg", "gaps_negative", 0), ("gap", "gap_mean", 3),
        ("UBP R", "revenue:chawla_ubp", 3), ("UBP W", "welfare:chawla_ubp", 3),
        ("LPIP W", "welfare:chawla_lpip", 3), ("QW W", "welfare:qirana_weighted_coverage", 3),
        ("QM R", "qm_revenue", 3), ("QM W", "qm_welfare", 3)]


def sens_table(base, label, caption):
    seeds = [base + "_seedset1", base + "_seedset2"]
    order = [base] + [("add_" if base == "paper" else "drop_") + s for s in CHOICE] + seeds
    head = " & ".join(["configuration"] + [c for c, _, _ in COLS] + [r"$\rho$"]) + r" \\"
    rows = []
    for c in order:
        if c == base:
            name = "setting" if base == "paper" else "every alternative on"
        elif c in seeds:
            name = "same, sampling seed set " + c[-1]
        else:
            sw = c.split("_", 1)[1]
            name = ("+ " if base == "paper" else r"$-$ ") + CHOICE[sw]
        cells = [fmt(ab.loc[c, q], d) for _, q, d in COLS] + [fmt(spearman(c), 2)]
        rows.append(" & ".join([name] + cells) + r" \\")
    body = "\n    ".join(rows)
    return rf"""\begin{{table*}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{{caption}}}
  \label{{{label}}}
  \begin{{tabular}}{{@{{}}l{'r' * (len(COLS) + 1)}@{{}}}}
    \toprule
    {head}
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


LEG = ("claimed and refuted edges; zero: zero-priced assets and priced: priceable assets "
       "under readership-weighted sampling (of 222); cap: share of fits at the grid cap; "
       r"neg: negative generalization gaps (of 15); gap: mean gap; R: revenue, W: welfare; "
       r"QW: \qirana{} weighted coverage; QM: \querymarket{} view cover; $\rho$: Spearman "
       "correlation of revenue and welfare over the 15 mechanisms.")
write("tab_app_sens_setting.tex", sens_table(
    "paper", "tab:app-sens-setting",
    "Sensitivity analysis on the largest \\sqlshare{} market, starting from the setting of "
    "Table 2 (each row switches one choice to its alternative). Columns: " + LEG))
write("tab_app_sens_alt.tex", sens_table(
    "corrected", "tab:app-sens-alt",
    "Sensitivity analysis on the largest \\sqlshare{} market, starting from every "
    "alternative of Table 2 on (each row switches one choice back). Columns as in "
    "Table~\\ref{tab:app-sens-setting}."))

# ---------------------------------------------------------------- per market
def market_frame(prot):
    st = pd.read_csv(os.path.join(RES, "sqlshare", prot, "markets", "structure.csv")).set_index("rank")
    fb = pd.read_csv(os.path.join(RES, "sqlshare", prot, "markets", "figure2b_markets.csv")).set_index("rank")
    return st.join(fb)


ms, mc = market_frame("paper"), market_frame("corrected")
rows = []
for rank in ms.index:
    a, b = ms.loc[rank], mc.loc[rank]
    rows.append(" & ".join([
        str(rank), f"{int(a.assets)}", f"{int(a.pairs):,}", f"{int(a.falsified)}",
        f"{100 * a.zero_priced_frac:.1f}", fmt(a.best_published_welfare - a.best_flat_welfare, 3),
        fmt(a.best_published_revenue - a.best_flat_revenue, 3),
        f"{int(b.pairs):,}", f"{int(b.falsified)}", f"{100 * b.zero_priced_frac:.1f}",
        fmt(b.best_published_welfare - b.best_flat_welfare, 3),
        fmt(b.best_published_revenue - b.best_flat_revenue, 3)]) + r" \\")
write("tab_app_markets.tex", r"""\begin{table*}[t]
  \centering
  \footnotesize
  \caption{The 15 \sqlshare{} markets of Table 8 (reduced protocol, Section 4.4), at the setting and with every alternative on. Refuted: refuted pairs; zero \%: share of zero-priced assets; $\Delta W$ and $\Delta R$: welfare and revenue of the better of \qirana{} weighted coverage and \querymarket{} view cover minus those of the better of the flat fee and UBP.}
  \label{tab:app-markets}
  \begin{tabular}{@{}rrrrrrrrrrrr@{}}
    \toprule
    & & \multicolumn{5}{c}{setting} & \multicolumn{5}{c}{every alternative on} \\
    \cmidrule(lr){3-7}\cmidrule(lr){8-12}
    rank & assets & pairs & refuted & zero \% & $\Delta W$ & $\Delta R$ & pairs & refuted & zero \% & $\Delta W$ & $\Delta R$ \\
    \midrule
    """ + "\n    ".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- per valuation family
tr = pd.read_csv(os.path.join(RES, "tpch", "full_tpch_runs.csv"))
tf = tr.groupby(["family", "mechanism"]).mean(numeric_only=True)
sf = pd.read_csv(os.path.join(RES, "sqlshare", "paper", "main_market", "summary_by_family.csv"))
sf = sf[sf.arm == "published"].set_index(["family", "mechanism"])


def family_table(kind, label, caption):
    col_t = {"revenue": "revenue_strategic_norm", "welfare": "welfare_strategic_norm"}[kind]
    col_s = col_t + "_mean"
    best = {}
    for fam, _ in FAMILIES:
        best[("t", fam)] = max(TEN, key=lambda m: tf.loc[(fam, m), col_t])
        best[("s", fam)] = max(TEN, key=lambda m: sf.loc[(fam, m), col_s])
    rows = []
    for name, m in MECH:
        cells = []
        for w, src, col in (("t", tf, col_t), ("s", sf, col_s)):
            for fam, _ in FAMILIES:
                v = f"{src.loc[(fam, m), col]:.3f}"
                cells.append(r"\textbf{" + v + "}" if best[(w, fam)] == m else v)
        rows.append(" & ".join([name] + cells) + r" \\")
    fams = " & ".join([f for _, f in FAMILIES] * 2)
    return rf"""\begin{{table*}}[t]
  \centering
  \footnotesize
  \caption{{{caption}}}
  \label{{{label}}}
  \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
    \toprule
    & \multicolumn{{4}}{{c}}{{\tpch{{}}}} & \multicolumn{{4}}{{c}}{{\sqlshare{{}}}} \\
    \cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
    mechanism & {fams} \\
    \midrule
    {chr(10).join("    " + r for r in rows).strip()}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""


FAM_LEG = (r"cells: answer cells; support: support rows read; rows$\cdot$log: rows times "
           r"$\log(1 + \text{support rows})$; flat: one value index for every asset. Mean over five "
           r"seeds per family; Table 4 is the mean of the four columns of each workload. "
           r"Bold: the highest among the ten constructions. $\dagger$: catalog substituted on \sqlshare{}.")
write("tab_app_family_revenue.tex", family_table(
    "revenue", "tab:app-family-revenue",
    "Revenue against recomposing buyers per valuation family. " + FAM_LEG))
write("tab_app_family_welfare.tex", family_table(
    "welfare", "tab:app-family-welfare",
    "Welfare against recomposing buyers per valuation family. " + FAM_LEG))

# ---------------------------------------------------------------- facts for the text
print("UBP top revenue of the ten, per family:")
for fam, _ in FAMILIES:
    print("  TPC-H", fam, max(TEN, key=lambda m: tf.loc[(fam, m), "revenue_strategic_norm"]),
          "| SQLShare", fam, max(TEN, key=lambda m: sf.loc[(fam, m), "revenue_strategic_norm_mean"]))
lp = [(w, fam) for w, src, col_r, col_w in (("t", tf, "revenue_strategic_norm", "welfare_strategic_norm"),
                                             ("s", sf, "revenue_strategic_norm_mean", "welfare_strategic_norm_mean"))
      for fam, _ in FAMILIES
      if src.loc[(fam, "chawla_lpip"), col_w] > src.loc[(fam, "chawla_ubp"), col_w]
      and src.loc[(fam, "chawla_lpip"), col_r] < src.loc[(fam, "chawla_ubp"), col_r]]
print("LPIP more welfare and less revenue than UBP in", len(lp), "of 8 workload-family cells")
qw = [(w, fam) for w, src, col in (("t", tf, "welfare_strategic_norm"), ("s", sf, "welfare_strategic_norm_mean"))
      for fam, _ in FAMILIES if src.loc[(fam, "qirana_weighted_coverage"), col] > src.loc[(fam, "chawla_ubp"), col]]
print("Qirana w.c. welfare above UBP in", len(qw), "of 8 cells:", qw)
print("markets: setting dW>0", int((ms.best_published_welfare > ms.best_flat_welfare).sum()),
      "dR<0", int((ms.best_published_revenue < ms.best_flat_revenue).sum()),
      "| all alternatives dW>0", int((mc.best_published_welfare > mc.best_flat_welfare).sum()),
      "dR<0", int((mc.best_published_revenue < mc.best_flat_revenue).sum()))
print("markets with refuted pairs: setting", int((ms.falsified > 0).sum()), "alt", int((mc.falsified > 0).sum()))

# Writes the LaTeX table files of the revision-3 paper and appendix from the result records
# in results/revised/ and checks each one against reproduce/expected_tables/ (the copies the
# paper and appendix print). Usage: python reproduce/make_paper_tables.py [records folder]
"""Write every data table of the paper and the appendix from the laptop result records.

    python tools/make_tables.py <results folder>

The results folder holds the returned laptop records of revision 3 (code 3.0.0.dev11):

  analysis/              results/revised/analysis of `python -m sqlshare_rerun analyze`
  sqlshare_studies/      results/revised/studies (studies A-D on SQLShare)
  tpch_studies/          runs/R11_P7c_all/tpch_studies (studies A, C, D on TPC-H)
  sqlshare_extract/      results/revised/extract (relations, worked examples)
  sqlshare_revised/      results/revised (main_market, markets, lattice) of the P3b run
  tpch_revised/          runs/R06_P3_full/tpch_out/revised (TPC-H, Qirana updates)
  tpch_revised_delete/   runs/R06_P3_full/tpch_out/revised_delete (TPC-H, deletions)

No number is typed by hand: every cell is read from those files. Means over runs are
taken here only where the analysis did not already write them (marked in the code).
Output: floats/tables/tab_*.tex (overwritten).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reproduce", "out", "tables")
os.makedirs(OUT, exist_ok=True)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "revised")
EXPECTED = os.path.join(ROOT, "reproduce", "expected_tables")
WRITTEN = []

NAMES = {
    "size_proportional": "size proportional",
    "compute_metered": "compute metered",
    "monopoly_per_asset": "per-asset monopoly",
    "monotone_constrained": "monotone fitted",
    "qirana_weighted_coverage": r"\qirana{} weighted coverage",
    "qirana_shannon": r"\qirana{} Shannon entropy",
    "qirana_uniform_entropy_gain": r"\qirana{} unif.\ entropy gain$^{\ddagger}$",
    "querymarket_viewcover": r"\querymarket{} view cover$^{\dagger}$",
    "querymarket_viewcover_instance": r"\querymarket{}, instance level$^{\S}$",
    "chawla_ubp": "uniform bundle (UBP)",
    "chawla_uip": "uniform item (UIP)",
    "chawla_lpip": "LP item (LPIP)",
    "chawla_cip": "capacity item (CIP)",
    "chawla_layering": "layering",
    "chawla_xos": "XOS",
}
ORDER = ["size_proportional", "compute_metered", "monopoly_per_asset", "monotone_constrained",
         "qirana_weighted_coverage", "qirana_shannon", "qirana_uniform_entropy_gain",
         "querymarket_viewcover", "querymarket_viewcover_instance",
         "chawla_ubp", "chawla_uip", "chawla_lpip", "chawla_cip", "chawla_layering", "chawla_xos"]
CONSTRUCTIONS = ORDER[4:]
GROUP_BREAKS = {"qirana_weighted_coverage", "chawla_ubp"}      # \midrule before these rows


def P(*parts):
    return os.path.join(DATA, *parts)


def csv(*parts):
    return pd.read_csv(P(*parts))


def f3(x, plus=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.3f}" if plus else f"{x:.3f}"
    return s.replace("-", "$-$")


def f2(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.2f}".replace("-", "$-$")


def ci(lo, hi):
    return f"[{f3(lo)}, {f3(hi)}]"


def intc(x):
    return f"{int(round(x)):,}"


def write(name, text):
    if name.startswith("tab_app_"):      # appendix floats may also go on a float page
        text = text.replace("\\begin{table}[t]", "\\begin{table}[tbp]").replace("\\begin{table*}[t]", "\\begin{table*}[tp]")
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    WRITTEN.append(name)
    print("wrote", name)


def rows_in_order(mechs, skip=()):
    out = []
    for m in ORDER:
        if m in mechs and m not in skip:
            out.append(m)
    return out


# ---------------------------------------------------------------------------------------
def tab_funnel():
    f = csv("analysis", "funnel.csv")
    lines = []
    for r in f.itertuples():
        if r.step == "survivors (non-empty answers)":
            continue
        step = {"in the structural markets (lattice)": "in the 15 structural markets (lattice)",
                "materialized assets in structural markets": "materialized assets, structural markets",
                "assets in markets with the full protocol": "assets, 15 markets with the full protocol",
                "binds against loaded schema": "names resolve against loaded schema"}.get(r.step, r.step)
        lines.append(f"    {step} & {intc(r.count)} \\\\")
    fam = csv("analysis", "lattice_families.csv").set_index("family")
    body = "\n".join(lines)
    write("tab_funnel.tex", rf"""
\begin{{table}}[t]
  \centering
  \small
  \caption{{\sqlshare{{}} funnel, from the query log to the assets of the 15 markets with the full protocol; Section~\ref{{sec:workloads}} explains the last step.}}
  \label{{tab:funnel}}
  \begin{{tabular}}{{@{{}}p{{0.70\columnwidth}}r@{{}}}}
    \toprule
    step & queries \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    return fam


def tab_main():
    m = csv("analysis", "main_results.csv")
    W = {"sqlshare_main": "main", "sqlshare_15": "m15", "tpch": "tpch"}
    get = {}
    for r in m.itertuples():
        get[(r.workload, r.mechanism)] = r
    lines = []
    for mech in ORDER:
        if mech in GROUP_BREAKS:
            lines.append("    \\midrule")
        cells = [NAMES[mech]]
        for w in ("sqlshare_main", "sqlshare_15", "tpch"):
            r = get.get((w, mech))
            if r is None:
                cells += ["--", "--", "--"]
                continue
            sd = "_sd_across_markets" if w == "sqlshare_15" else "_sd"
            cells.append(f"{f3(r.revenue_strategic_op_norm_mean)} $\\pm$ {f3(getattr(r, 'revenue_strategic_op_norm' + sd))}")
            cells.append(f3(r.welfare_strategic_op_norm_mean))
            cells.append(f2(r.served_strategic_op_mean))
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_main.tex", rf"""
\begin{{table*}}[t]
  \centering
  \small
  \setlength{{\tabcolsep}}{{4pt}}
  \caption{{Revenue (mean $\pm$ standard deviation), welfare, and share of buyers served against the operational buyer, normalized by the training monopoly revenue; main arm, 20 runs per market (Section~\ref{{sec:metrics}}). $\dagger$: substituted catalog; $\ddagger$: information-arbitrage-free only; $\S$: comparison only (Section~\ref{{sec:mechanisms}}).}}
  \label{{tab:main}}
  \begin{{tabular}}{{@{{}}lccccccccc@{{}}}}
    \toprule
    & \multicolumn{{3}}{{c}}{{\sqlshare{{}} main market}} & \multicolumn{{3}}{{c}}{{\sqlshare{{}}, 15 markets}} & \multicolumn{{3}}{{c}}{{\tpch{{}}}} \\
    \cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
    mechanism & revenue & welfare & served & revenue & welfare & served & revenue & welfare & served \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_comparisons():
    c = csv("analysis", "comparisons.csv")
    c = c[c.comparison == "vs_uniform"]
    get = {(r.workload, r.metric, r.a): r for r in c.itertuples()}
    lines = []
    for mech in ORDER:
        if mech == "chawla_ubp":
            continue
        if mech in GROUP_BREAKS or mech == "chawla_uip":
            if mech in ("qirana_weighted_coverage", "chawla_uip"):
                lines.append("    \\midrule")
        cells = [NAMES[mech]]
        for w in ("sqlshare_main", "sqlshare_15", "tpch"):
            for metric in ("revenue_strategic_op_norm", "welfare_strategic_op_norm"):
                r = get.get((w, metric, mech))
                if r is None:
                    cells.append("--")
                    continue
                txt = ci(r.ci_low, r.ci_high)
                if r.ci_excludes_zero:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
            r = get.get((w, "revenue_strategic_op_norm", mech))
            cells.append("--" if r is None else f"{r.settings_a_higher}/{r.settings}")
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_comparisons.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{2.7pt}}
  \caption{{Paired 95\% bootstrap intervals of the mean difference with UBP (mechanism minus UBP), operational buyer (Section~\ref{{sec:metrics}}). Bold: the interval excludes zero. ``higher'': settings in which the mechanism earns more revenue than UBP. Symbols as in Table~\ref{{tab:main}}.}}
  \label{{tab:comparisons}}
  \begin{{tabular}}{{@{{}}lccrccrccr@{{}}}}
    \toprule
    & \multicolumn{{3}}{{c}}{{\sqlshare{{}} main market}} & \multicolumn{{3}}{{c}}{{\sqlshare{{}}, 15 markets}} & \multicolumn{{3}}{{c}}{{\tpch{{}}}} \\
    \cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
    mechanism & $\Delta$ revenue & $\Delta$ welfare & higher & $\Delta$ revenue & $\Delta$ welfare & higher & $\Delta$ revenue & $\Delta$ welfare & higher \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def _runs_main():
    r = csv("sqlshare_revised", "main_market", "runs.csv")
    return r[r.arm == "same"]


def _runs_markets():
    f = csv("sqlshare_revised", "markets", "full_protocol_runs.csv")
    return f[f.arm == "same"]


def _runs_tpch():
    t = csv("tpch_revised", "runs.csv")
    return t[(t.arm == "same") & (t.n_train == t.n_test)]


def tab_audits():
    # means over runs of laptop per-run records (the analysis wrote only some of them)
    mm, mk, tp = _runs_main(), _runs_markets(), _runs_tpch()

    def mean(df, col, mech, by_market=False):
        g = df[df.mechanism == mech]
        if col not in g or g.empty:
            return float("nan")
        if by_market:
            return float(g.groupby("cid")[col].mean().mean())
        return float(g[col].mean())

    lines = []
    for mech in ORDER:
        if mech in GROUP_BREAKS:
            lines.append("    \\midrule")
        cells = [NAMES[mech]]
        cells += [f3(mean(mm, "mono_violation_rate_sem", mech)), f3(mean(mm, "comb_violation_rate_op", mech)),
                  f3(mean(mm, "comb_max_gain_op", mech)), f3(mean(mm, "comb_violation_rate", mech)),
                  f3(mean(mm, "comb_max_gain", mech))]
        cells += [f3(mean(mk, "comb_violation_rate_op", mech, True)), f3(mean(mk, "comb_violation_rate", mech, True))]
        cells += [f3(mean(tp, "mono_violation_rate", mech)), f3(mean(tp, "comb_violation_rate_op", mech)),
                  f3(mean(tp, "comb_max_gain_op", mech))]
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_audits.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \caption{{Arbitrage audits (Equation~\eqref{{eq:audits}}), mean over runs. $r_{{\mathrm{{I}}}}$: share of query-text pairs whose prices invert. $r_{{\mathrm{{C}}}}$: share of assets rebuilt for less than the posted price; max: mean largest gain. Attackers as in Section~\ref{{sec:attackers}}. Symbols as in Table~\ref{{tab:main}}.}}
  \label{{tab:audits}}
  \begin{{tabular}}{{@{{}}lccccccccccc@{{}}}}
    \toprule
    & \multicolumn{{5}}{{c}}{{\sqlshare{{}} main market}} & \multicolumn{{2}}{{c}}{{15 markets}} & \multicolumn{{3}}{{c}}{{\tpch{{}}}} \\
    \cmidrule(lr){{2-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-11}}
    & & \multicolumn{{2}}{{c}}{{operational}} & \multicolumn{{2}}{{c}}{{privileged}} & oper. & priv. & & \multicolumn{{2}}{{c}}{{operational}} \\
    \cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{10-11}}
    mechanism & $r_{{\mathrm{{I}}}}$ & $r_{{\mathrm{{C}}}}$ & max & $r_{{\mathrm{{C}}}}$ & max & $r_{{\mathrm{{C}}}}$ & $r_{{\mathrm{{C}}}}$ & $r_{{\mathrm{{I}}}}$ & $r_{{\mathrm{{C}}}}$ & max \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_relations():
    rel = csv("analysis", "relations_summary.csv")
    allr = rel[rel.cid == "all"].iloc[0]
    main = rel[rel.cid.astype(str) == "474"].iloc[0]
    write("tab_relations.tex", rf"""
\begin{{table}}[t]
  \centering
  \small
  \setlength{{\tabcolsep}}{{3.5pt}}
  \caption{{The three relations of Section~\ref{{sec:relations}} on \sqlshare{{}}: pairs found, and pairs refuted by a support instance that changes the target answer but not the source answer (Corollary~\ref{{cor:falsifier}}). Pricing support: 1,000 \qirana{{}} updates per market.}}
  \label{{tab:relations}}
  \begin{{tabular}}{{@{{}}lrrrr@{{}}}}
    \toprule
    & \multicolumn{{2}}{{c}}{{main market}} & \multicolumn{{2}}{{c}}{{15 markets}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
    relation & pairs & refuted & pairs & refuted \\
    \midrule
    derivability on $D$ (Def.~\ref{{def:derivability}}) & {intc(main.lattice_pairs)} & {intc(main.lattice_refuted_pricing)} & {intc(allr.lattice_pairs)} & {intc(allr.lattice_refuted_pricing)} \\
    query-text guarantee (Def.~\ref{{def:querytext}}) & {intc(main.query_text_pairs)} & {intc(main.query_text_refuted_pricing)} & {intc(allr.query_text_pairs)} & {intc(allr.query_text_refuted_pricing)} \\
    both & {intc(main.both)} & -- & {intc(allr.both)} & -- \\
    \midrule
    \multicolumn{{5}}{{@{{}}l}}{{main market, other supports (lattice pairs refuted)}} \\
    \quad 30,000 size-proportional deletions & & {intc(main.lattice_refuted_curve)} & & \\
    \quad 1,000 readership-weighted deletions & & {intc(main.lattice_refuted_readership)} & & \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    return rel


def tab_causes():
    cs = csv("analysis", "causes.csv")
    order = ["instance-specific derivation (refuted by the pricing support)",
             "no bundle guarantee (information-arbitrage-free only)",
             "target not for sale (infinite price)",
             "baseline without an arbitrage guarantee",
             "unexplained (conflict sets cover the target)"]
    label = {order[0]: "instance-specific coincidence", order[1]: "no bundle guarantee (UEG)",
             order[2]: "target not for sale ($\\pi = \\infty$)", order[3]: "baseline, no guarantee",
             order[4]: "conflict sets cover the target"}
    get = {(r.workload, r.attacker, r.cause): r.distinct_plans for r in cs.itertuples()}
    lines = []
    for c in order:
        vals = [get.get(("sqlshare_main", "privileged" if False else "oracle", c), 0),
                get.get(("sqlshare_main", "operational", c), 0), get.get(("tpch", "operational", c), 0)]
        lines.append(f"    {label[c]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    body = "\n".join(lines)
    write("tab_causes.tex", rf"""
\begin{{table}}[t]
  \centering
  \small
  \caption{{Free rebuilds: distinct verified plans (target, sources) with gain 1.00 under a construction, main arm, by cause (Section~\ref{{sec:causes}}). Baselines are not counted.}}
  \label{{tab:causes}}
  \begin{{tabular}}{{@{{}}lrrr@{{}}}}
    \toprule
    & \multicolumn{{2}}{{c}}{{\sqlshare{{}} main}} & \tpch{{}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-4}}
    cause & privileged & operational & operational \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_arms():
    a = csv("analysis", "arms_sizes.csv")
    mechs = ["chawla_ubp", "chawla_lpip", "chawla_cip", "qirana_weighted_coverage", "monopoly_per_asset"]
    short = {"chawla_ubp": "UBP", "chawla_lpip": "LPIP", "chawla_cip": "CIP",
             "qirana_weighted_coverage": "WC", "monopoly_per_asset": "mon."}
    rows = [("same", 800, "same demand (main arm)"), ("shift10", 800, "10\\% of ranks permuted"),
            ("shift25", 800, "25\\% of ranks permuted"), ("shift50", 800, "50\\% of ranks permuted"),
            ("redrawn", 800, "redrawn ranking"), None,
            ("same", 100, "100 training buyers"), ("same", 200, "200"), ("same", 400, "400"),
            ("same", 1600, "1,600")]
    get = {(r.workload, r.demand_arm, r.n_train, r.mechanism): r for r in a.itertuples()}
    lines = []
    for spec in rows:
        if spec is None:
            lines.append("    \\midrule")
            continue
        arm, n, lab = spec
        r0 = get[("sqlshare_main", arm, n, "chawla_ubp")]
        tau = f"{r0.kendall_tau_mean:.2f}" if arm != "same" else "1.00"
        cells = [lab, tau]
        for w in ("sqlshare_main", "tpch"):
            for m in mechs:
                r = get.get((w, arm, n, m))
                cells.append("--" if r is None else f3(r.generalization_gap_mean))
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    hdr = " & ".join(short[m] for m in mechs)
    write("tab_arms.tex", rf"""
\begin{{table*}}[t]
  \centering
  \small
  \caption{{Generalization gap $\Gamma$ (Equation~\eqref{{eq:gap}}, share of training revenue lost on test buyers; negative: a gain) by test arm (800 training buyers) and by training size (same demand). $\tau$: Kendall rank correlation between training and test popularity (main market). Mean over 4 families $\times$ 5 seeds. WC: \qirana{{}} weighted coverage; mon.: per-asset monopoly.}}
  \label{{tab:arms}}
  \begin{{tabular}}{{@{{}}lcccccccccccc@{{}}}}
    \toprule
    & & \multicolumn{{5}}{{c}}{{\sqlshare{{}} main market}} & \multicolumn{{5}}{{c}}{{\tpch{{}}}} \\
    \cmidrule(lr){{3-7}}\cmidrule(lr){{8-12}}
    test arm or training size & $\tau$ & {hdr} & {hdr} \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_studyA():
    a = csv("analysis", "study_A_contrasts.csv")
    a = a[(a.metric == "revenue_over_values") & (a.support == "updates") & (a.lp_cap == 0)]
    get = {(r.valuation, r.buyers, r.protocol): r for r in a.itertuples()}
    vals = [("ours", "ours (4 families)"), ("uniform10", "Uniform$[1, 10]$"), ("uniform100", "Uniform$[1, 100]$"),
            ("zipf1.5", "Zipf, $a = 1.5$"), ("zipf2.5", "Zipf, $a = 2.5$")]
    lines = []
    for v, lab in vals:
        cells = [lab]
        for b in ("per_query", "popularity"):
            for p in ("in_sample", "held_out"):
                r = get.get((v, b, p))
                if r is None:
                    cells.append("--")
                    continue
                txt = f"{f3(r.mean_diff, plus=True)} ({r.workloads_lpip_higher}/{r.workloads})"
                if (r.ci_low > 0) or (r.ci_high < 0):
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_studyA.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{Study A: LPIP minus UBP revenue as a share of the buyers' total value (the normalization of Chawla et al.), mean over 16 workloads (15 markets and \tpch{{}}), \qirana{{}} supports, no LP cap; in brackets, workloads where LPIP earns more. Bold: the 95\% bootstrap interval over workloads excludes zero.}}
  \label{{tab:studyA}}
  \begin{{tabular}}{{@{{}}lcccc@{{}}}}
    \toprule
    & \multicolumn{{2}}{{c}}{{one buyer per query}} & \multicolumn{{2}}{{c}}{{800 buyers by popularity}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
    valuations & in sample & held out & in sample & held out \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_sampling():
    s = csv("sqlshare_studies", "B", "study_B_supports.csv")
    m = s[s.cid == 474]
    name = {"size_deletions": "size-prop.\\ deletions", "readership_deletions": "readership deletions",
            "qirana_updates": "\\qirana{} updates"}
    lines = []
    for smp in ("size_deletions", "readership_deletions", "qirana_updates"):
        e = m[(m.sampler == smp) & (m.condition == "equal_size")]
        cells = [name[smp], intc(e.spent_seconds.mean()), intc(e.spent_queries.mean()),
                 f"{e.assets_priceable.mean():.0f}", f"{e.distinct_price_levels.mean():.0f}",
                 f"{e.lattice_pairs_refuted.mean():.0f}"]
        if smp == "readership_deletions":
            cells += ["--", "--", "--"]
        else:
            b = m[(m.sampler == smp) & (m.condition == "budget_of_readership_deletions") & (m.currency == "seconds")]
            capped = " (cap)" if b.draw_capped.any() else ""
            cells += [intc(b.support_n.mean()) + capped, f"{b.assets_priceable.mean():.0f}",
                      f"{b.lattice_pairs_refuted.mean():.0f}"]
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_sampling.tex", rf"""
\begin{{table*}}[t]
  \centering
  \small
  \caption{{Study B on the main market, mean over three sampler seeds: cost of 1,000 support instances, the priceable assets, price levels, and refuted lattice pairs, and the same at the time budget of the readership sampler. (cap): stopped at 20,000 instances.}}
  \label{{tab:sampling}}
  \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
    \toprule
    & \multicolumn{{5}}{{c}}{{equal size, $|\supportset| = 1000$}} & \multicolumn{{3}}{{c}}{{equal time budget}} \\
    \cmidrule(lr){{2-6}}\cmidrule(lr){{7-9}}
    sampler & seconds & queries & priceable & levels & refuted & $|\supportset|$ & priceable & refuted \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_matched():
    c = csv("analysis", "study_C_comparisons.csv")
    s = pd.concat([csv("sqlshare_studies", "C", "study_C_summary.csv"), csv("tpch_studies", "C", "study_C_summary.csv")])
    mechs = ["chawla_lpip", "chawla_xos", "chawla_uip", "qirana_weighted_coverage", "qirana_uniform_entropy_gain",
             "querymarket_viewcover", "monopoly_per_asset"]
    lines = []
    for mech in mechs:
        cells = [NAMES[mech]]
        for match, lvl in (("revenue", 0.4), ("coverage", 0.5)):
            for w in ("sqlshare_474", "tpch"):
                r = s[(s.workload == w) & (s.view == "operational") & (s.match == match) & (s.level == lvl)]
                u = r[r.mechanism == "chawla_ubp"]
                x = r[r.mechanism == mech]
                cells.append("--" if x.empty or u.empty else f3(float(x.welfare_norm.iloc[0] - u.welfare_norm.iloc[0]), plus=True))
            g = c[(c.view == "operational") & (c.match == match) & (c.level == lvl) & (c.mechanism == mech)]
            cells.append(f"{int((g.ci_low > 0).sum())}/{int((g.ci_high < 0).sum())}/{len(g)}")
        lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_matched.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{Study C: welfare minus that of UBP at matched revenue (40\% of the reference) and matched coverage (50\% served), operational buyer; mean over matched runs. $k$: workloads with the 95\% interval above / below zero / workloads compared.}}
  \label{{tab:matched}}
  \begin{{tabular}}{{@{{}}lcccccc@{{}}}}
    \toprule
    & \multicolumn{{3}}{{c}}{{revenue 0.40}} & \multicolumn{{3}}{{c}}{{coverage 0.50}} \\
    \cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
    mechanism & main & \tpch{{}} & $k$ & main & \tpch{{}} & $k$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_costs():
    c = csv("analysis", "costs.csv")
    sup = c[c.item.str.startswith("support build (pricing)")]
    main_sup = sup[sup.workload == "sqlshare_474"].seconds_mean.iloc[0]
    others = sup[sup.workload != "sqlshare_474"].seconds_mean
    tb = c[(c.workload == "tpch") & (c.item == "support (qirana_updates)")].seconds_mean.iloc[0]
    tdel = c[(c.workload == "tpch") & (c.item == "support (deletions)")].seconds_mean.iloc[0]

    def fit(w, m):
        x = c[(c.workload == w) & (c.item == "price fitting") & (c.mechanism == m)]
        return float(x.seconds_mean.iloc[0]) if len(x) else float("nan")

    def plan(w, m, kind):
        x = c[(c.workload == w) & (c.item == f"planning ({kind})") & (c.mechanism == m)]
        return float(x.seconds_mean.iloc[0]) if len(x) else float("nan")

    def s(x):
        if not math.isfinite(x):
            return "--"
        return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 1 else f"{x:.3f}")

    rows = [("support build, \\qirana{} updates, 1,000 instances", s(main_sup),
             f"{s(others.median())} [{s(others.min())}, {s(others.max())}]", s(tb))]
    for m, lab in (("chawla_lpip", "fit LPIP (one LP per buyer)"), ("chawla_cip", "fit CIP"),
                   ("chawla_ubp", "fit UBP"), ("qirana_weighted_coverage", "fit \\qirana{} weighted coverage")):
        rows.append((lab, s(fit("sqlshare_main", m)), s(fit("sqlshare_15", m)), s(fit("tpch", m))))
    rows.append(("planning, operational buyer (LPIP prices)", s(plan("sqlshare_main", "chawla_lpip", "operational")),
                 s(plan("sqlshare_15", "chawla_lpip", "operational")), s(plan("tpch", "chawla_lpip", "operational"))))
    rows.append(("planning, privileged attacker (LPIP prices)", s(plan("sqlshare_main", "chawla_lpip", "oracle")),
                 s(plan("sqlshare_15", "chawla_lpip", "oracle")), "--"))
    body = "\n".join("    " + " & ".join(r) + r" \\" for r in rows)
    write("tab_costs.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{Costs in seconds on one laptop (Section~\ref{{sec:environment}}): support construction once per market, price fitting and planning per run (mean). Other markets: median [range] over the 14 markets for the support, mean over 15 markets otherwise. \tpch{{}} deletion support: {s(tdel)} s.}}
  \label{{tab:costs}}
  \begin{{tabular}}{{@{{}}p{{0.43\columnwidth}}rrr@{{}}}}
    \toprule
    & main & other markets & \tpch{{}} \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


# ---------------------------------------------------------------------------------------
# appendix tables
# ---------------------------------------------------------------------------------------
def tab_app_markets():
    st = csv("sqlshare_revised", "markets", "structure.csv")
    rel = csv("analysis", "relations_summary.csv")
    rel = rel[rel.cid != "all"].copy()
    rel["cid"] = rel.cid.astype(int)
    fp = _runs_markets()
    ex = csv("sqlshare_extract", "assets.csv")
    lines = []
    for r in st.sort_values("rank").itertuples():
        rr = rel[rel.cid == r.cid]
        e = ex[ex.cid == r.cid]
        g = fp[fp.cid == r.cid]
        def rv(m):
            x = g[g.mechanism == m].revenue_strategic_op_norm
            return f3(float(x.mean())) if len(x) else "--"
        def wv(m):
            x = g[g.mechanism == m].welfare_strategic_op_norm
            return f3(float(x.mean())) if len(x) else "--"
        frag = int(e.in_fragment.astype(bool).sum()) if len(e) else 0
        pri = int((e.conflict_size_pricing > 0).sum()) if len(e) else 0
        lines.append("    " + " & ".join([str(r.cid), str(r.assets), str(frag), str(pri),
                                          intc(rr.lattice_pairs.iloc[0]), intc(rr.query_text_pairs.iloc[0]),
                                          intc(rr.lattice_refuted_pricing.iloc[0]),
                                          rv("chawla_ubp"), rv("chawla_lpip"), rv("qirana_weighted_coverage"),
                                          wv("chawla_ubp"), wv("chawla_lpip")]) + r" \\")
    body = "\n".join(lines)
    write("tab_app_markets.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \caption{{The 15 \sqlshare{{}} markets with the full protocol (market 474 is the main market): assets, assets in the operational fragment, assets priced above zero by the \qirana{{}} support, lattice and query-text pairs, lattice pairs refuted, and revenue and welfare against the operational buyer (main arm, mean over 20 runs).}}
  \label{{tab:app-markets}}
  \begin{{tabular}}{{@{{}}rrrrrrrccccc@{{}}}}
    \toprule
    & & & & \multicolumn{{3}}{{c}}{{pairs}} & \multicolumn{{3}}{{c}}{{revenue}} & \multicolumn{{2}}{{c}}{{welfare}} \\
    \cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}\cmidrule(lr){{11-12}}
    market & assets & fragment & priceable & lattice & query text & refuted & UBP & LPIP & \qirana{{}} WC & UBP & LPIP \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_app_family():
    p = csv("analysis", "per_family.csv")
    fams = [("answer_cells", "cells"), ("support_rows", "support"), ("rows_x_log_support", "rows$\\cdot$log"), ("flat", "flat")]
    for metric, nm, cap in (("revenue_strategic_op_norm", "tab_app_family_revenue.tex", "Revenue"),
                            ("welfare_strategic_op_norm", "tab_app_family_welfare.tex", "Welfare")):
        lines = []
        for mech in ORDER:
            if mech in GROUP_BREAKS:
                lines.append("    \\midrule")
            cells = [NAMES[mech]]
            for w in ("sqlshare_main", "sqlshare_15", "tpch"):
                for f, _ in fams:
                    x = p[(p.workload == w) & (p.family == f) & (p.mechanism == mech)]
                    cells.append("--" if x.empty else f3(float(x[metric].iloc[0])))
            lines.append("    " + " & ".join(cells) + r" \\")
        body = "\n".join(lines)
        hdr = " & ".join(lab for _, lab in fams)
        write(nm, rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{{cap} per valuation family against the operational buyer (main arm; 5 seeds per cell). Families: index proportional to answer cells, to support rows read, to rows times $\log(1 + \text{{support rows}})$, or flat.}}
  \label{{tab:app-family-{cap.lower()}}}
  \begin{{tabular}}{{@{{}}lcccccccccccc@{{}}}}
    \toprule
    & \multicolumn{{4}}{{c}}{{\sqlshare{{}} main market}} & \multicolumn{{4}}{{c}}{{\sqlshare{{}}, 15 markets}} & \multicolumn{{4}}{{c}}{{\tpch{{}}}} \\
    \cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}\cmidrule(lr){{10-13}}
    mechanism & {hdr} & {hdr} & {hdr} \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_app_studyA():
    a = csv("analysis", "study_A_contrasts.csv")
    lines = []
    for metric, mlab in (("revenue_over_values", "share of value"), ("revenue_norm_main", "monopoly ref.")):
        for support in ("updates", "deletions"):
            for cap in (0, 60):
                x = a[(a.metric == metric) & (a.support == support) & (a.lp_cap == cap)]
                if x.empty:
                    continue
                for v in ("ours", "uniform10", "uniform100", "zipf1.5", "zipf2.5"):
                    cells = [mlab, support, "none" if cap == 0 else "60", v.replace("uniform", "U").replace("zipf", "Zipf ")]
                    for b in ("per_query", "popularity"):
                        for pr in ("in_sample", "held_out"):
                            r = x[(x.valuation == v) & (x.buyers == b) & (x.protocol == pr)]
                            if r.empty:
                                cells.append("--")
                                continue
                            r = r.iloc[0]
                            cells.append(f"{f3(r.mean_diff, plus=True)} ({r.workloads_lpip_higher}/{r.workloads})")
                    lines.append("    " + " & ".join(cells) + r" \\")
                lines.append("    \\midrule")
    body = "\n".join(lines[:-1])
    write("tab_app_studyA.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \caption{{Study A, all cells: LPIP minus UBP revenue, mean over workloads (in brackets: workloads where LPIP earns more). Normalization by the buyers' total value (Chawla et al.) or by the training monopoly revenue (the main protocol). Supports: \qirana{{}} updates (16 workloads) or row deletions (main market and \tpch{{}}).}}
  \label{{tab:app-studyA}}
  \begin{{tabular}}{{@{{}}llllcccc@{{}}}}
    \toprule
    & & & & \multicolumn{{2}}{{c}}{{one buyer per query}} & \multicolumn{{2}}{{c}}{{popularity}} \\
    \cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}
    normalization & support & LP cap & valuations & in sample & held out & in sample & held out \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_app_workloads():
    w = pd.concat([csv("sqlshare_studies", "A", "study_A_workloads.csv"), csv("tpch_studies", "A", "study_A_workloads.csv")])
    lines = []
    for r in w.itertuples():
        lines.append("    " + " & ".join([r.workload.replace("sqlshare_", "").replace("tpch", "\\tpch{}"), r.support,
                                          str(r.queries), intc(r.items_used), f"{r.empty_edge_share:.2f}",
                                          f"{r.edge_size_mean:.1f}", f"{r.mean_jaccard:.3f}", f"{r.nested_share:.3f}",
                                          f"{r.disjoint_share:.3f}"]) + r" \\")
    body = "\n".join(lines)
    write("tab_app_workloads.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{2.5pt}}
  \caption{{Workload structure in study A: hyperedges (conflict sets) of the queries on each support. Shares are over pairs of non-empty hyperedges.}}
  \label{{tab:app-workloads}}
  \begin{{tabular}}{{@{{}}llrrrrrrr@{{}}}}
    \toprule
    workload & support & queries & items & empty & size & Jaccard & nested & disjoint \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_app_studyB():
    b = csv("analysis", "study_B_table.csv")
    b = b[b.mechanism.isin(["qirana_weighted_coverage", "chawla_lpip"])]
    name = {"size_deletions": "size del.", "readership_deletions": "readership del.", "qirana_updates": "\\qirana{} upd."}
    cond = {"equal_size": "equal size", "budget_of_size_deletions": "budget of size del.",
            "budget_of_readership_deletions": "budget of readership del.", "budget_of_qirana_updates": "budget of \\qirana{} upd."}
    lines = []
    for smp in ("size_deletions", "readership_deletions", "qirana_updates"):
        for cd in ("equal_size", "budget_of_size_deletions", "budget_of_readership_deletions", "budget_of_qirana_updates"):
            for cur in ("-", "seconds", "queries"):
                x = b[(b.sampler == smp) & (b.condition == cd) & (b.currency == cur)]
                if x.empty:
                    continue
                q = x[x.mechanism == "qirana_weighted_coverage"].iloc[0]
                l = x[x.mechanism == "chawla_lpip"].iloc[0]
                lines.append("    " + " & ".join([name[smp], cond[cd], cur if cur != "-" else "", intc(q.support_n),
                                                  f"{q.assets_priceable:.1f}", f"{q.distinct_price_levels:.1f}",
                                                  f"{q.lattice_pairs_refuted:.1f}", f3(q.revenue_strategic_op_norm),
                                                  f3(l.revenue_strategic_op_norm), f3(l.welfare_strategic_op_norm)]) + r" \\")
        lines.append("    \\midrule")
    body = "\n".join(lines[:-1])
    write("tab_app_studyB.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \caption{{Study B, all conditions: mean over markets 474, 4930, 7635, 8126 and three sampler seeds. Revenue and welfare against the operational buyer (main arm, 4 families).}}
  \label{{tab:app-studyB}}
  \begin{{tabular}}{{@{{}}lllrrrrccc@{{}}}}
    \toprule
    sampler & condition & currency & $|\supportset|$ & priceable & levels & refuted & WC revenue & LPIP revenue & LPIP welfare \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_app_studyC():
    c = csv("analysis", "study_C_comparisons.csv")
    lines = []
    for match, levels in (("revenue", (0.3, 0.4, 0.5)), ("coverage", (0.25, 0.5, 0.75))):
        for lvl in levels:
            for view in ("naive", "operational"):
                x = c[(c.view == view) & (c.match == match) & (c.level == lvl)]
                cells = [match, f"{lvl:.2f}", view]
                for mech in ("chawla_lpip", "chawla_xos", "qirana_weighted_coverage", "qirana_uniform_entropy_gain",
                             "querymarket_viewcover", "monopoly_per_asset"):
                    g = x[x.mechanism == mech]
                    cells.append(f"{int((g.ci_low > 0).sum())}/{int((g.ci_high < 0).sum())}/{len(g)}")
                lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write("tab_app_studyC.tex", rf"""
\begin{{table*}}[t]
  \centering
  \footnotesize
  \caption{{Study C, all levels: workloads where the 95\% interval of welfare minus that of UBP is above / below zero / workloads compared (runs matched for both mechanisms). Naive: buyers pay posted prices; operational: buyers rebuild when cheaper.}}
  \label{{tab:app-studyC}}
  \begin{{tabular}}{{@{{}}lllcccccc@{{}}}}
    \toprule
    match & level & view & LPIP & XOS & \qirana{{}} WC & \qirana{{}} UEG & \querymarket{{}} & monopoly \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def tab_app_studyD():
    s = pd.concat([csv("sqlshare_studies", "D", "study_D_supports.csv"), csv("tpch_studies", "D", "study_D_supports.csv")])
    d = csv("analysis", "study_D_comparisons.csv")
    d = d[d.metric == "revenue_strategic_op_norm"]
    lines = []
    for r in s.itertuples():
        lines.append("    " + " & ".join([r.workload.replace("sqlshare_474", "main market").replace("tpch", "\\tpch{}"),
                                          r.donors, str(r.assets_priceable), str(r.distinct_price_levels),
                                          f"{r.median_conflict_set:.1f}", str(r.lattice_pairs_refuted)]) + r" \\")
    body1 = "\n".join(lines)
    lines = []
    for mech in ("qirana_weighted_coverage", "qirana_shannon", "qirana_uniform_entropy_gain", "chawla_uip",
                 "chawla_lpip", "chawla_cip", "chawla_layering", "chawla_xos"):
        cells = [NAMES[mech]]
        for w in ("sqlshare_474", "tpch"):
            g = d[(d.workload == w) & (d.mechanism == mech)]
            cells.append("--" if g.empty else f"{f3(float(g.frequency_minus_distinct.iloc[0]), plus=True)} {ci(float(g.ci_low.iloc[0]), float(g.ci_high.iloc[0]))}")
        lines.append("    " + " & ".join(cells) + r" \\")
    body2 = "\n".join(lines)
    write("tab_app_studyD.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{Study D: \qirana{{}} supports with values drawn uniformly from the distinct values (distinct, the main protocol) or copied from a random row (frequency); same relations, seeds, and size. Top: support quality. Bottom: revenue against the operational buyer, frequency minus distinct, with the 95\% interval over the 20 paired runs.}}
  \label{{tab:app-studyD}}
  \begin{{tabular}}{{@{{}}llrrrr@{{}}}}
    \toprule
    workload & donors & priceable & levels & median $|C|$ & refuted \\
    \midrule
{body1}
    \bottomrule
  \end{{tabular}}

  \medskip
  \begin{{tabular}}{{@{{}}lcc@{{}}}}
    \toprule
    mechanism & main market & \tpch{{}} \\
    \midrule
{body2}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_app_costs():
    c = csv("analysis", "costs.csv")
    sup = c[c.item.str.startswith("support build")]
    lines = []
    for r in sup.itertuples():
        lines.append(f"    {r.workload.replace('sqlshare_', '')} & {r.item.replace('support build ', '')} & {r.seconds_mean:,.1f} & {int(r.draws):,} \\\\")
    body = "\n".join(lines)
    write("tab_app_costs.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \caption{{Support construction per \sqlshare{{}} market (seconds; candidates drawn to accept 1,000 instances, 30,000 for the deletion curve). Pricing: \qirana{{}} updates.}}
  \label{{tab:app-costs}}
  \begin{{tabular}}{{@{{}}llrr@{{}}}}
    \toprule
    market & support & seconds & draws \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def tab_app_fragment():
    s = json.load(open(P("sqlshare_revised", "main_market", "summary.json"), encoding="utf-8"))
    op = s["operational"]
    reasons = sorted(op["outside_fragment_reasons"].items(), key=lambda x: -x[1])
    body = "\n".join(f"    {k} & {v} \\\\" for k, v in reasons)
    write("tab_app_fragment.tex", rf"""
\begin{{table}}[t]
  \centering
  \footnotesize
  \caption{{Operational fragment on the main market: {op['in_fragment']} of {op['assets']} query texts are in the fragment, {op['targets_with_sources']} targets have query-text sources, and {op['semantic_pairs']} query-text pairs hold. Reasons for the queries outside the fragment (first reason found).}}
  \label{{tab:app-fragment}}
  \begin{{tabular}}{{@{{}}lr@{{}}}}
    \toprule
    reason & queries \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def numbers():
    """Values quoted in the text, written to tools/numbers.json for the checker."""
    out = {}
    s = json.load(open(P("sqlshare_revised", "main_market", "summary.json"), encoding="utf-8"))
    out["main_summary"] = {k: s[k] for k in ("assets", "claimed_edges", "plans_verified", "plans_failed",
                                             "corr_size_vs_readership", "zero_priced_pricing_support")}
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tab_funnel()
    tab_main()
    tab_comparisons()
    tab_audits()
    tab_relations()
    tab_causes()
    tab_arms()
    tab_studyA()
    tab_sampling()
    tab_matched()
    tab_costs()
    tab_app_markets()
    tab_app_family()
    tab_app_studyA()
    tab_app_workloads()
    tab_app_studyB()
    tab_app_studyC()
    tab_app_studyD()
    tab_app_costs()
    tab_app_fragment()
    numbers()
    # Compare every written table with the copy that the paper and appendix print.
    if os.path.isdir(EXPECTED):
        same = [n for n in WRITTEN if os.path.exists(os.path.join(EXPECTED, n))
                and open(os.path.join(OUT, n), encoding="utf-8").read()
                == open(os.path.join(EXPECTED, n), encoding="utf-8").read()]
        diff = [n for n in WRITTEN if n not in same]
        print(f"Tables: {len(same)} of {len(WRITTEN)} match the paper and appendix")
        for n in diff:
            print("  DIFFERS:", n)
        sys.exit(1 if diff else 0)

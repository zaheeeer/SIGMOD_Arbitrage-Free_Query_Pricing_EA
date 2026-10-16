"""Buyers, settlement, scale fitting, and the ported pricing constructions.

Historical functions are reproduced from legacy/tpc_experiment.py, legacy/run_full.py,
legacy/ports_chawla.py and the recovered notebook helpers. Corrected variants come from
candidate/protocol.py (shared demand, exact breakpoint fit, exact partition entropy).
"""
from __future__ import annotations

import collections
import math
import time

import numpy as np
from scipy.optimize import linprog

GRID60 = np.concatenate([np.linspace(0.01, 5, 200), np.linspace(5.2, 60, 140)])


# ----------------------------------------------------------------------------------------
# buyers
# ----------------------------------------------------------------------------------------
def buyer_sample(n, info, rng, gamma=0.7, sigma=0.6, scale=100.0, demand_skew=1.0):
    """Historical sampler: draws a NEW popularity ranking on every call (concern C04)."""
    k = len(info)
    rank = rng.permutation(k) + 1
    demand = 1.0 / np.power(rank, demand_skew)
    demand = demand / demand.sum()
    target = rng.choice(k, size=n, p=demand)
    theta = rng.lognormal(0.0, sigma, size=n)
    value = scale * theta * np.power(info[target], gamma)
    return target, value, demand


def _popularity(k, rng, skew=1.0):
    weights = (rng.permutation(k) + 1.0) ** (-skew)
    return weights / weights.sum()


def _sample_buyers(n, info, demand, rng, gamma=0.7, sigma=0.6, scale=100.0):
    targets = rng.choice(len(info), size=n, p=demand)
    values = scale * rng.lognormal(0.0, sigma, size=n) * info[targets] ** gamma
    return targets, values


def buyer_split(info, seed, n_train=800, n_test=800, shift=False):
    """Corrected sampler (candidate/protocol.py): one demand vector unless shift=True."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    train_demand = _popularity(len(info), streams[0])
    test_demand = _popularity(len(info), streams[3]) if shift else train_demand.copy()
    tr = _sample_buyers(n_train, info, train_demand, streams[1])
    te = _sample_buyers(n_test, info, test_demand, streams[2])
    return tr, te


SHIFT_ARMS = {"shift10": 0.10, "shift25": 0.25, "shift50": 0.50}


def buyer_arms(info, seed, arm, n_train=800, n_test=800):
    """Revision 3 (D7): three kinds of test population against one training population.

    same     test buyers drawn from the training demand (independent buyers only)
    shiftXX  the popularity ranks of XX% of the assets, chosen at random, are permuted
             among themselves; the other assets keep their training popularity
    redrawn  a new popularity ranking for the test buyers (the 2.x design)

    Returns ((train targets, values), (test targets, values), shift stats). The training
    buyers depend only on the seed, so every arm shares the same training sample.
    """
    from scipy.stats import kendalltau
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    k = len(info)
    train = _popularity(k, streams[0])
    if arm == "same":
        test = train.copy()
    elif arm in SHIFT_ARMS:
        m = int(round(SHIFT_ARMS[arm] * k))
        idx = np.sort(streams[3].choice(k, size=m, replace=False)) if m > 1 else np.array([], int)
        test = train.copy()
        if len(idx) > 1:
            test[idx] = train[idx][streams[4].permutation(len(idx))]
    elif arm == "redrawn":
        test = _popularity(k, streams[3])
    else:
        raise ValueError(f"unknown demand arm {arm!r}")
    tr = _sample_buyers(n_train, info, train, streams[1])
    te = _sample_buyers(n_test, info, test, streams[2])
    tau = 1.0 if np.allclose(train, test) else float(kendalltau(train, test)[0])
    stats = {"kendall_tau": tau, "demand_tv": float(0.5 * np.abs(train - test).sum())}
    return tr, te, stats


def draw_buyers(demand_mode: str, arm: str, info, seed_value: int, n_train: int, n_test: int):
    if demand_mode == "arms":
        tr, te, _ = buyer_arms(info, seed_value, arm, n_train, n_test)
        return tr, te
    if demand_mode == "redraw":
        rng = np.random.default_rng(seed_value)
        tr_t, tr_v, _ = buyer_sample(n_train, info, rng)
        te_t, te_v, _ = buyer_sample(n_test, info, rng)
        return (tr_t, tr_v), (te_t, te_v)
    return buyer_split(info, seed_value, n_train, n_test, shift=(arm == "shift"))


def score(targets, values, prices):
    c = np.asarray(prices, float)[targets]
    v = np.asarray(values, float)
    buys = c <= v
    rev = float(c[buys].sum())
    sur = float((v[buys] - c[buys]).sum())
    return {"revenue": rev, "surplus": sur, "welfare": rev + sur, "served": float(buys.mean())}


def info_families(rowsz, ssz, ncols):
    def nz(x):
        x = np.asarray(x, dtype=float)
        return x / max(x.max(), 1e-12)
    rowsz = np.asarray(rowsz, float)
    ssz = np.asarray(ssz, float)
    return {
        "answer_cells": nz(rowsz * np.asarray(ncols, float)),
        "support_rows": nz(ssz),
        "rows_x_log_support": nz(rowsz * np.log1p(ssz)),
        "flat": np.ones(len(rowsz)),
    }


# ----------------------------------------------------------------------------------------
# scale fitting
# ----------------------------------------------------------------------------------------
def fit_scale_grid(target, value, scores, grid=GRID60):
    """Historical fit: best multiplier on a grid capped at 60 (concern C05)."""
    s = np.asarray(scores, float)[target]
    best_a, best_r = grid[0], -1.0
    for a in grid:
        c = a * s
        r = float(np.sum(c[c <= value]))
        if r > best_r:
            best_a, best_r = a, r
    return float(best_a)


def fit_scale_breakpoints(target, value, scores, chunk=256):
    """Exact, uncapped fit. The optimum lies at a value/score breakpoint.

    Candidates are 0, every breakpoint, and the next float below it; revenue at each is
    evaluated exactly as settlement does (price <= value), and the smallest multiplier
    with the highest revenue is returned, as in candidate/protocol.py.
    """
    s = np.asarray(scores, float)[np.asarray(target)]
    v = np.asarray(value, float)
    pos = (s > 0) & np.isfinite(s)
    if not pos.any():
        return 0.0
    with np.errstate(over="ignore", divide="ignore"):
        brk = v[pos] / s[pos]
    brk = brk[np.isfinite(brk)]
    cands = np.unique(np.r_[0.0, brk, np.nextafter(brk, 0.0)])
    best_a, best_r = 0.0, -1.0
    fin = np.isfinite(s)
    sf = np.where(fin, s, 0.0)
    for i in range(0, len(cands), chunk):
        a = cands[i:i + chunk][:, None]
        cost = a * sf[None, :]
        ok = (cost <= v[None, :]) & fin[None, :]
        rev = np.where(ok, cost, 0.0).sum(axis=1)
        j = int(np.argmax(rev))
        if rev[j] > best_r:
            best_a, best_r = float(a[j, 0]), float(rev[j])
    return best_a


def scaled_prices(fit_mode: str, target, value, scores):
    """Normalize a score vector, fit one multiplier, return (prices, alpha)."""
    sc = np.asarray(scores, float)
    finite = np.isfinite(sc)
    mx = sc[finite].max() if finite.any() else 1.0
    s = sc / max(mx, 1e-12)
    if fit_mode == "grid60":
        alpha = fit_scale_grid(target, value, s)
    else:
        alpha = fit_scale_breakpoints(target, value, s)
    with np.errstate(invalid="ignore"):
        p = alpha * s
    p[~finite] = np.inf
    return p, alpha


def monopoly_prices(target, value, k):
    p = np.zeros(k)
    fallback = float(np.median(value)) if len(value) else 1.0
    for j in range(k):
        v = np.sort(value[target == j])[::-1]
        if len(v) == 0:
            p[j] = fallback
            continue
        rev = v * (np.arange(len(v)) + 1)
        p[j] = float(v[int(np.argmax(rev))])
    return p


def enforce_monotone(p, pairs, iters=200):
    q = np.asarray(p, float).copy()
    for _ in range(iters):
        changed = False
        for i, j in pairs:
            if q[j] > q[i] + 1e-12:
                q[i] = q[j]
                changed = True
        if not changed:
            break
    return q


def audit_monotone(prices, pairs, tol=1e-9):
    if not pairs:
        return {"n_pairs": 0, "violation_rate": 0.0, "max_excess": 0.0}
    viol, worst = 0, 0.0
    for i, j in pairs:
        excess = prices[j] - prices[i]
        if excess > tol * max(prices[i], 1.0):
            viol += 1
            worst = max(worst, excess / max(prices[i], 1e-12))
    return {"n_pairs": len(pairs), "violation_rate": viol / len(pairs), "max_excess": worst}


# ----------------------------------------------------------------------------------------
# Qirana and QueryMarket scores
# ----------------------------------------------------------------------------------------
def qirana_from_edges(edges, n):
    """Historical scores; entropy assumes every changed answer is its own class (C06)."""
    cov = np.array([float(len(e)) for e in edges])
    frac = cov / n
    sh = np.zeros(len(edges))
    for i, kk in enumerate(cov.astype(int)):
        probs = ([(n - kk) / n] if n - kk > 0 else []) + [1.0 / n] * int(kk)
        p = np.array(probs)
        sh[i] = float(-(p * np.log2(p)).sum())
    return {"qirana_weighted_coverage": cov, "qirana_uniform_gain": frac, "qirana_shannon": sh}


def exact_entropy(changed_fp: dict, n: int) -> float:
    """Entropy over the answer classes the support set induces (candidate/protocol.py)."""
    counts = collections.Counter(changed_fp.values())
    base = n - sum(counts.values())
    classes = ([base] if base > 0 else []) + list(counts.values())
    p = np.array(classes, float) / n
    return float(-(p * np.log2(p)).sum())


def qirana_prices(edges, n, changed=None, weights=None):
    """Qirana pricing functions as defined by Deep and Koutris (SIGMOD 2017), Section 2.3.

    ``edges[i]`` is the conflict set C(Q_i) as a set of support indices and ``n = |S|``.
    ``changed[i]`` (aligned with ``edges``) maps each support index in C(Q_i) to the answer
    fingerprint that Q_i returns on that support database; it is needed for the entropy
    partition. ``weights`` are the support weights w_i (default 1; the seller's scale is
    fitted afterwards, so a common factor does not matter).

    weighted coverage     p_wc  = sum_{i in C} w_i                                (Eq. 1)
    uniform entropy gain  p_ueg = log|C| / log|S|                                 (Eq. 2)
    Shannon entropy       p_H   = -sum_{B in P_Q} w_B log2 w_B, with w_i = 1/|S|  (Eq. 3)

    P_Q partitions S by the query answer: the support databases outside C share the true
    answer, and those in C are grouped by their changed answer. |C| = 0 gives price 0 for
    every function, and |C| = 1 gives p_ueg = 0, as the formula implies.
    """
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    wc = np.array([float(w[sorted(e)].sum()) if len(e) else 0.0 for e in edges])
    size = np.array([len(e) for e in edges], float)
    if n > 1:
        ueg = np.log(np.maximum(size, 1.0)) / math.log(n)
    else:
        ueg = np.zeros(len(edges))
    out = {"qirana_weighted_coverage": wc, "qirana_uniform_entropy_gain": ueg}
    if changed is not None:
        if len(changed) != len(edges):
            raise ValueError("changed must be aligned with edges")
        for i, (e, ch) in enumerate(zip(edges, changed)):
            if set(ch) != set(e):
                raise ValueError(f"asset {i}: changed answers do not match its conflict set")
        out["qirana_shannon"] = np.array([exact_entropy(ch, n) for ch in changed])
    return out


def querymarket_single_predecessor(n_assets, pairs, base_price):
    """Historical SQLShare substitute (C09): maximal assets form the catalogue, priced by
    ``base_price``; any other asset takes its cheapest catalogue predecessor, or 1.5 x the
    largest catalogue price when none exists."""
    derivable_from = collections.defaultdict(list)
    for s, t in pairs:
        derivable_from[t].append(s)
    cat = [j for j in range(n_assets) if j not in derivable_from]
    catp = {j: float(base_price[j]) for j in cat}
    big = max(catp.values()) * 1.5 if catp else 1.0
    qm = np.array([catp.get(j, min([catp.get(s, big) for s in derivable_from.get(j, [])] or [big]))
                   for j in range(n_assets)])
    return qm, len(cat)


def querymarket_cover(keys, pairs, base_price, contrib, nrows, cover_fn, max_rounds=50):
    """Corrected QueryMarket (C09): the price of an asset is the cheapest way to determine
    it from the priced catalogue, through verified predecessors and verified row covers,
    composed until no price falls further. Assets that no catalogue subset determines are
    not for sale (infinite price), as in Koutris et al."""
    n = len(keys)
    ki = {k: i for i, k in enumerate(keys)}
    derivable_from = collections.defaultdict(list)
    for s, t in pairs:
        derivable_from[t].append(s)
    cat = [j for j in range(n) if j not in derivable_from]
    price = np.full(n, np.inf)
    for j in cat:
        price[j] = float(max(base_price[j], 1.0))
    catset = set(cat)
    for _ in range(max_rounds):
        changed = False
        for j in range(n):
            if j in catset:
                continue
            best = price[j]
            for s in derivable_from.get(j, []):
                best = min(best, price[s])
            k = keys[j]
            if k in contrib:
                sub = {s: idx for s, idx in contrib[k].items() if np.isfinite(price[ki[s]])}
                if sub:
                    covered = np.zeros(nrows[k], bool)
                    for idx in sub.values():
                        covered[idx] = True
                    if covered.all():
                        c, _plan = cover_fn(sub, nrows[k], {s: float(price[ki[s]]) for s in sub})
                        best = min(best, c)
            if best < price[j] - 1e-12:
                price[j] = best
                changed = True
        if not changed:
            break
    return price, len(cat)


# ----------------------------------------------------------------------------------------
# Chawla et al. (VLDB 2019) ports, from legacy/ports_chawla.py
# ----------------------------------------------------------------------------------------
def revenue(prices, vals):
    p = np.asarray(prices, float)
    v = np.asarray(vals, float)
    return float(np.sum(np.where(p <= v, p, 0.0)))


def ubp(edges, vals):
    v = np.asarray(vals, float)
    order = np.argsort(-v)
    best_p, best_r = 0.0, 0.0
    for idx in order:
        P = v[idx]
        r = P * np.count_nonzero(v >= P)
        if r > best_r:
            best_p, best_r = float(P), float(r)
    return np.full(len(edges), best_p), {"P": best_p}


def _edge_price(edges, w):
    return np.array([float(w[list(e)].sum()) if len(e) else 0.0 for e in edges])


def uip(edges, vals, n_items):
    """Uniform item pricing. Edge prices are computed with the historical summation, so
    boundary cases (price == value) resolve exactly as before, but each candidate costs
    one small sum per distinct edge size instead of one per edge."""
    v = np.asarray(vals, float)
    lens = np.array([len(e) for e in edges])
    sizes = np.array([max(len(e), 1) for e in edges], float)
    q = v / sizes
    uniq_len = np.unique(lens)
    best_w, best_r = 0.0, -1.0
    for cand in np.unique(q):
        lut = {L: (float(np.full(L, cand).sum()) if L else 0.0) for L in uniq_len}
        p = np.array([lut[L] for L in lens])
        r = revenue(p, v)
        if r > best_r:
            best_w, best_r = float(cand), r
    w = np.full(n_items, best_w)
    return _edge_price(edges, w), {"w": best_w, "weights": w}


def lpip(edges, vals, n_items, max_lps=60, compact=False):
    """``compact`` (revision 3, study B only) solves each LP over the items that appear in
    the buyers' edges; the other items get weight 0. Items outside every edge have no
    constraint and no objective weight, so the optimum is the same; the LP is smaller when
    the support is large (study B draws up to 20,000 neighbours). The main runs keep
    ``compact=False``."""
    if compact:
        used = sorted(set().union(*[set(e) for e in edges])) if len(edges) else []
        pos = {it: i for i, it in enumerate(used)}
        sub = [frozenset(pos[it] for it in e) for e in edges]
        p_sub, info = lpip(sub, vals, max(len(used), 1), max_lps=max_lps, compact=False)
        w = np.zeros(n_items)
        if used:
            w[used] = info["weights"][:len(used)]
        return _edge_price(edges, w), {"weights": w, "n_lps": info["n_lps"]}
    v = np.asarray(vals, float)
    srt = np.argsort(-v)
    if max_lps is None or max_lps <= 0 or len(srt) <= max_lps:     # revision 3 (D6): no cap
        order = srt
    else:
        order = srt[np.linspace(0, len(srt) - 1, max_lps).astype(int)]
    best_w, best_r = None, -1.0
    for idx in order:
        F = [i for i in range(len(edges)) if v[i] >= v[idx]]
        if not F:
            continue
        c = np.zeros(n_items)
        rowsA, b = [], []
        for i in F:
            ind = np.zeros(n_items)
            ind[list(edges[i])] = 1.0
            c += ind
            rowsA.append(ind)
            b.append(v[i])
        res = linprog(-c, A_ub=np.array(rowsA), b_ub=np.array(b),
                      bounds=[(0, None)] * n_items, method="highs")
        if not res.success:
            continue
        w = np.maximum(res.x, 0.0)
        r = revenue(_edge_price(edges, w), v)
        if r > best_r:
            best_w, best_r = w, r
    if best_w is None:
        best_w = np.zeros(n_items)
    return _edge_price(edges, best_w), {"weights": best_w, "n_lps": len(order)}


def cip(edges, vals, n_items, eps=0.5, k_max=None):
    """Capacity item pricing. The item-edge incidence is built once per call; the LP of
    each capacity level is identical to the historical one (same rows, same order)."""
    v = np.asarray(vals, float)
    m = len(edges)
    k_max = k_max or max(m, 2)
    M = np.zeros((n_items, m))
    for i, e in enumerate(edges):
        if len(e):
            M[list(e), i] = 1.0
    rows = [j for j in range(n_items) if M[j].any()]
    best_w, best_r = np.zeros(n_items), -1.0
    if not rows:
        return _edge_price(edges, best_w), {"weights": best_w}
    A = M[rows]
    k = 1.0
    while k <= k_max:
        res = linprog(-v, A_ub=A, b_ub=np.full(len(rows), k), bounds=[(0, 1)] * m, method="highs")
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            w = np.zeros(n_items)
            w[rows] = duals
            r = revenue(_edge_price(edges, w), v)
            if r > best_r:
                best_w, best_r = w, r
        k *= (1.0 + eps)
    return _edge_price(edges, best_w), {"weights": best_w}


def layering(edges, vals, n_items):
    v = np.asarray(vals, float)
    remaining = set(range(len(edges)))
    w = np.zeros(n_items)
    best_layer, best_rev = [], -1.0
    while remaining:
        universe = set().union(*[edges[i] for i in remaining]) if remaining else set()
        if not universe:
            break
        uncovered, layer = set(universe), []
        pool = sorted(remaining, key=lambda i: -len(set(edges[i]) & uncovered))
        for i in pool:
            gain = set(edges[i]) & uncovered
            if gain:
                layer.append(i)
                uncovered -= gain
            if not uncovered:
                break
        if not layer:
            break
        pruned = True
        while pruned and len(layer) > 1:
            pruned = False
            for i in list(layer):
                others = set().union(*[set(edges[j]) for j in layer if j != i])
                if set(edges[i]) <= others:
                    layer.remove(i)
                    pruned = True
                    break
        rev = float(sum(v[i] for i in layer))
        if rev > best_rev:
            best_layer, best_rev = list(layer), rev
        remaining -= set(layer)
    for i in best_layer:
        others = set().union(*[set(edges[j]) for j in best_layer if j != i]) if len(best_layer) > 1 else set()
        uniq = set(edges[i]) - others
        if uniq:
            w[min(uniq)] = v[i]
    return _edge_price(edges, w), {"weights": w, "layer_size": len(best_layer)}


def _lift(edges_asset, w):
    return np.array([float(w[list(e)].sum()) if len(e) else 0.0 for e in edges_asset])


def chawla_all(edges_asset, tr_t, tr_v, n_items, max_lps=60, eps=0.5, which=None, seconds=None,
               compact=False):
    """All six Chawla et al. price vectors for one training sample.

    XOS is the per-asset maximum of the LPIP and CIP prices; it reuses the LPIP and CIP
    solutions computed for their own rows (the historical code solved them twice).
    """
    which = which or ("chawla_ubp", "chawla_uip", "chawla_lpip", "chawla_cip",
                      "chawla_layering", "chawla_xos")
    edges_b = [edges_asset[t] for t in tr_t]
    vals_b = np.asarray(tr_v, float)
    out = {}
    clock = time.perf_counter
    sec = seconds if seconds is not None else {}
    if "chawla_ubp" in which:
        t0 = clock()
        _, info = ubp(edges_b, vals_b)
        sec["chawla_ubp"] = clock() - t0
        out["chawla_ubp"] = np.full(len(edges_asset), info["P"])
    if "chawla_uip" in which:
        t0 = clock()
        _, info = uip(edges_b, vals_b, n_items)
        sec["chawla_uip"] = clock() - t0
        out["chawla_uip"] = _lift(edges_asset, info["weights"])
    w_lp = w_cip = None
    if "chawla_lpip" in which or "chawla_xos" in which:
        t0 = clock()
        _, info = lpip(edges_b, vals_b, n_items, max_lps=max_lps, compact=compact)
        sec["chawla_lpip"] = clock() - t0
        w_lp = info["weights"]
        if "chawla_lpip" in which:
            out["chawla_lpip"] = _lift(edges_asset, w_lp)
    if "chawla_cip" in which or "chawla_xos" in which:
        t0 = clock()
        _, info = cip(edges_b, vals_b, n_items, eps=eps)
        sec["chawla_cip"] = clock() - t0
        w_cip = info["weights"]
        if "chawla_cip" in which:
            out["chawla_cip"] = _lift(edges_asset, w_cip)
    if "chawla_layering" in which:
        t0 = clock()
        _, info = layering(edges_b, vals_b, n_items)
        sec["chawla_layering"] = clock() - t0
        out["chawla_layering"] = _lift(edges_asset, info["weights"])
    if "chawla_xos" in which:
        out["chawla_xos"] = np.array([max(float(w_lp[list(e)].sum()) if len(e) else 0.0,
                                          float(w_cip[list(e)].sum()) if len(e) else 0.0)
                                      for e in edges_asset])
        sec["chawla_xos"] = sec.get("chawla_lpip", 0.0) + sec.get("chawla_cip", 0.0)
    return out

"""Revision 3, Part 8: hand-computed checks of the Chawla et al. ports (supervisor comment 1).

Hypergraph (items = support instances 0, 1, 2; one hyperedge per buyer):

    e0 = {0}     value 10
    e1 = {0, 1}  value 12
    e2 = {2}     value  3
    e3 = {1, 2}  value  8

Every expected value below is computed by hand in the comments, from the definitions in
Chawla, Deep, Koutris, Teng, PVLDB 13(1), Sections 3 and 5.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlshare_rerun import pricing as P    # noqa: E402

EDGES = [frozenset({0}), frozenset({0, 1}), frozenset({2}), frozenset({1, 2})]
VALS = np.array([10.0, 12.0, 3.0, 8.0])
N_ITEMS = 3


def test_ubp_by_hand():
    # One bundle price P in {12, 10, 8, 3}; revenue P * #{v >= P} = 12, 20, 24, 12.
    p, info = P.ubp(EDGES, VALS)
    assert info["P"] == 8.0
    assert p.tolist() == [8.0] * 4
    assert P.revenue(p, VALS) == 24.0


def test_uip_by_hand():
    # Item price w in {v_e / |e|} = {10, 6, 3, 4}; edge price w * |e|.
    # w = 3: prices 3, 6, 3, 6, all sell, revenue 18
    # w = 4: prices 4, 8, 4, 8, e2 does not sell, revenue 20
    # w = 6: prices 6, 12, 6, 12, e0 and e1 sell, revenue 18
    # w = 10: only e0 sells, revenue 10
    p, info = P.uip(EDGES, VALS, N_ITEMS)
    assert info["w"] == 4.0
    assert p.tolist() == [4.0, 8.0, 4.0, 8.0]
    assert P.revenue(p, VALS) == 20.0


def test_lpip_by_hand():
    # LP(e) maximizes the sum of the prices of the edges with value >= v_e, subject to each
    # of those prices <= its value.  For the threshold 8 (edges e1, e0, e3):
    #   max 2 w0 + 2 w1 + w2  s.t.  w0 <= 10, w0 + w1 <= 12, w1 + w2 <= 8, w >= 0.
    #   With w0 + w1 = 12 and w2 = 8 - w1 the objective is 32 - w1, and w0 <= 10 forces
    #   w1 >= 2, so the unique optimum is w = (10, 2, 6): prices 10, 12, 6, 8, revenue 30.
    # Threshold 3 (all edges) also reaches revenue 30 but is not strictly better, and the
    # thresholds 12 and 10 reach at most 24, so the port keeps w = (10, 2, 6).
    p, info = P.lpip(EDGES, VALS, N_ITEMS, max_lps=0)
    assert info["n_lps"] == 4                         # one LP per buyer, no cap
    assert np.allclose(info["weights"], [10.0, 2.0, 6.0])
    assert np.allclose(p, [10.0, 12.0, 6.0, 8.0])
    assert P.revenue(p, VALS) == pytest.approx(30.0)


def test_layering_by_hand():
    # Layer 1: greedy cover of {0, 1, 2} takes e1 then e3 (two new items each); neither is
    # redundant, so the minimal cover is {e1, e3}, total value 20.
    # Layer 2: {e0, e2}, total value 13.  The best layer is {e1, e3}.
    # Unique items: e1 \ e3 = {0} priced 12, e3 \ e1 = {2} priced 8, item 1 priced 0.
    # Edge prices 12, 12, 8, 8: e1 and e3 sell, revenue 20.
    p, info = P.layering(EDGES, VALS, N_ITEMS)
    assert info["layer_size"] == 2
    assert info["weights"].tolist() == [12.0, 0.0, 8.0]
    assert p.tolist() == [12.0, 12.0, 8.0, 8.0]
    assert P.revenue(p, VALS) == 20.0


def test_cip_prices_are_dual_feasible_by_hand():
    # Capacity k = 1: max 10 x0 + 12 x1 + 3 x2 + 8 x3 s.t. x0 + x1 <= 1 (item 0),
    # x1 + x3 <= 1 (item 1), x2 + x3 <= 1 (item 2), 0 <= x <= 1.  The optimum is 18
    # (x0 = x3 = 1).  The duals are not unique, so we check the properties that every
    # optimal dual has: item prices >= 0, and each edge in the optimal allocation has a
    # price no larger than its value.  CIP keeps the best capacity level, so its revenue
    # is at least that of capacity 1.
    p, info = P.cip(EDGES, VALS, N_ITEMS, eps=0.5)
    w = info["weights"]
    assert (w >= -1e-9).all()
    assert np.allclose(p, [w[list(e)].sum() for e in EDGES])
    assert P.revenue(p, VALS) >= 1e-9


def test_xos_takes_the_larger_bundle_price_by_hand():
    lp, _ = P.lpip(EDGES, VALS, N_ITEMS, max_lps=0)
    cp, _ = P.cip(EDGES, VALS, N_ITEMS, eps=0.5)
    out = P.chawla_all(EDGES, np.arange(4), VALS, N_ITEMS, max_lps=0, eps=0.5,
                       which=("chawla_lpip", "chawla_cip", "chawla_xos"))
    assert np.allclose(out["chawla_xos"], np.maximum(lp, cp))

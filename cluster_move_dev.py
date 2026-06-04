"""Overlap-based (Houdayer/ICM-style) cluster move for Max-Cut -- dev/validation.

Given two configs (incumbent + pool member), build connected components on the
DISAGREEMENT set, flip one component in the incumbent. This is the coordinated
multi-vertex move that single connected sub-QUBO re-solves cannot do.
For optimization (not sampling): propose, keep if cut improves.
"""
import numpy as np
import networkx as nx
from src.graph import load_gset
from src.local_search import compute_cut_value, random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend


def overlap_components(G, xA, xB):
    """Return connected components of the disagreement set between xA, xB,
    after spin-reversal alignment. Largest-first."""
    nodes = list(G.nodes())
    agree = sum(1 for v in nodes if xA[v] == xB[v])
    if agree < len(nodes) / 2:                 # spin-reversal symmetry
        xB = {v: 1 - xB[v] for v in nodes}
    N = [v for v in nodes if xA[v] != xB[v]]
    if not N:
        return [], xB
    HN = G.subgraph(N)
    comps = sorted(nx.connected_components(HN), key=len, reverse=True)
    return [set(c) for c in comps], xB


def cluster_move(G, x, x_other, rng, try_all=True):
    """Propose cluster flips from the overlap of x and x_other.
    Returns the best improving result (or x unchanged if none improve).
    try_all: evaluate flipping each component; keep best improvement."""
    comps, _ = overlap_components(G, x, x_other)
    if not comps:
        return dict(x), 0.0
    base = compute_cut_value(G, x)
    best_x, best_gain = dict(x), 0.0
    cand = comps if try_all else comps[:1]
    for comp in cand:
        xc = dict(x)
        for v in comp:
            xc[v] = 1 - xc[v]
        gain = compute_cut_value(G, xc) - base
        if gain > best_gain:
            best_gain, best_x = gain, xc
    return best_x, best_gain


if __name__ == "__main__":
    G = load_gset("data/gset/G11.txt")
    neal = get_backend("neal")

    # build two distinct good configs: AQLS-550 and Fiedler-564
    xA, _ = adaptive_qls(G, budget_seconds=20,
                         selector=select_frustrated_connected, backend=neal,
                         k_min=400, k_max=400, n_reads=100, seed=0,
                         acceptance='improvement', best_known=564)
    xB, _ = adaptive_qls(G, budget_seconds=20,
                         selector=select_fiedler, backend=neal,
                         k_min=400, k_max=400, n_reads=100, seed=0,
                         acceptance='improvement', best_known=564)
    cutA, cutB = compute_cut_value(G, xA), compute_cut_value(G, xB)
    print(f"config A (FConn): {cutA}")
    print(f"config B (Fiedler): {cutB}")

    comps, _ = overlap_components(G, xA, xB)
    print(f"disagreement components: {len(comps)}, "
          f"sizes (top5): {[len(c) for c in comps[:5]]}")

    # apply cluster move to A using B as the other config
    xNew, gain = cluster_move(G, xA, xB, np.random.default_rng(0))
    cutNew = compute_cut_value(G, xNew)
    # validate the result is a real cut and bookkeeping is consistent
    assert all(xNew[v] in (0, 1) for v in G.nodes()), "invalid assignment"
    assert abs(cutNew - (cutA + gain)) < 1e-6, "gain bookkeeping mismatch"
    print(f"\ncluster move on A: {cutA} -> {cutNew} (gain {gain:+.1f})")
    print("VALID + improves" if gain > 0 else "valid, no single-component improvement")

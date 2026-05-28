"""
Unit tests for QUBO builder.
Critical test: K4 with unit weights has known optimal cut = 4.
"""

import pytest
import networkx as nx
from src.qubo import build_local_qubo, qubo_energy, merge_proposal
from src.gain_cache import GainCache
from src.local_search import compute_cut_value


def make_k4():
    """Complete graph on 4 vertices with unit weights."""
    G = nx.complete_graph(4)
    return G


def exact_solve(Q, S):
    """Brute-force exact solver for small subproblems."""
    S = list(S)
    n = len(S)
    best_energy = float('inf')
    best_x = None
    for mask in range(2 ** n):
        x_local = {}
        for i, v in enumerate(S):
            x_local[v] = (mask >> i) & 1
        e = qubo_energy(Q, x_local, S)
        if e < best_energy:
            best_energy = e
            best_x = dict(x_local)
    return best_x, best_energy


def test_k4_full_subproblem():
    """
    K4 with all 4 vertices in S and a fixed initial assignment.
    The optimal Max-Cut for K4 is 4 (bipartition {0,1} vs {2,3}).
    """
    G = make_k4()
    x = {0: 0, 1: 0, 2: 0, 3: 0}  # all on same side — worst cut = 0
    S = list(G.nodes())

    Q = build_local_qubo(G, x, S)
    x_local, _ = exact_solve(Q, S)
    x_prop = merge_proposal(x, x_local, S)

    cut = compute_cut_value(G, x_prop)
    assert cut == 4.0, f"Expected cut=4 for K4, got {cut}"


def test_k4_partial_subproblem():
    """
    K4 with only 2 vertices in S, 2 fixed.
    Check that boundary conditions are applied correctly.
    """
    G = make_k4()
    x = {0: 0, 1: 0, 2: 1, 3: 1}  # already optimal cut = 4
    S = [0, 1]  # only optimise vertices 0 and 1

    Q = build_local_qubo(G, x, S)
    x_local, _ = exact_solve(Q, S)
    x_prop = merge_proposal(x, x_local, S)

    cut = compute_cut_value(G, x_prop)
    # optimal with {2,3} fixed to side 1 is to put {0,1} on side 0
    assert cut >= 4.0, f"Expected cut >= 4 for K4 partial, got {cut}"


def test_qubo_energy_zero_assignment():
    """All zeros assignment should give zero or negative energy."""
    G = make_k4()
    x = {0: 0, 1: 0, 2: 0, 3: 0}
    S = list(G.nodes())
    Q = build_local_qubo(G, x, S)
    x_local = {v: 0 for v in S}
    energy = qubo_energy(Q, x_local, S)
    assert energy == 0.0, f"Expected 0 energy for all-zero assignment, got {energy}"


def test_merge_proposal():
    """Merge should only update vertices in S."""
    x = {0: 0, 1: 0, 2: 0, 3: 0}
    x_local = {0: 1, 1: 1}
    S = [0, 1]
    x_prop = merge_proposal(x, x_local, S)
    assert x_prop[0] == 1
    assert x_prop[1] == 1
    assert x_prop[2] == 0  # unchanged
    assert x_prop[3] == 0  # unchanged
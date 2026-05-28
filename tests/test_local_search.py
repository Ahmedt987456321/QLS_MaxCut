"""
Tests for graph loading and local search.
"""

import networkx as nx
import numpy as np
from src.graph import generate_random_regular, generate_erdos_renyi, generate_sbm
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, compute_cut_value, random_cut


def test_random_regular_generation():
    """Check a 3-regular graph on 10 vertices is generated correctly."""
    G = generate_random_regular(10, 3, seed=42)
    assert G.number_of_nodes() == 10
    for v, d in G.degree():
        assert d == 3, f"Vertex {v} has degree {d}, expected 3"


def test_erdos_renyi_generation():
    """Check ER graph has correct number of vertices."""
    G = generate_erdos_renyi(20, 0.3, seed=42)
    assert G.number_of_nodes() == 20


def test_one_flip_ls_reaches_local_optimum():
    """
    After running one_flip_ls to completion, no single flip
    should improve the cut.
    """
    G = generate_random_regular(20, 3, seed=42)
    rng = np.random.default_rng(42)
    x = random_cut(G, rng)
    gc = GainCache()

    x_opt, gc, at_opt, n_flips = one_flip_ls(G, x, gc)

    assert at_opt is True, "Should reach local optimum"

    # verify: no improving flip exists
    gc.assert_valid()
    for v in G.nodes():
        assert gc.gain[v] <= 0, (
            f"Vertex {v} has positive gain {gc.gain[v]} at claimed local optimum"
        )


def test_cut_improves_after_ls():
    """Local search should never decrease cut value."""
    G = generate_erdos_renyi(30, 0.3, seed=99)
    rng = np.random.default_rng(99)
    x = random_cut(G, rng)
    gc = GainCache()

    cut_before = compute_cut_value(G, x)
    x_opt, gc, at_opt, n_flips = one_flip_ls(G, x, gc)
    cut_after = compute_cut_value(G, x_opt)

    assert cut_after >= cut_before, (
        f"Cut decreased: {cut_before} -> {cut_after}"
    )


def test_random_cut_valid():
    """Random cut should assign every vertex 0 or 1."""
    G = generate_random_regular(10, 3, seed=0)
    rng = np.random.default_rng(0)
    x = random_cut(G, rng)
    for v in G.nodes():
        assert x[v] in (0, 1), f"Vertex {v} has invalid assignment {x[v]}"

def test_qls_runs_end_to_end():
    """
    QLS should run without errors and return a valid cut
    better than a random cut on a small graph.
    """
    from src.qls import qls
    from src.backends import backend_exact

    G = generate_random_regular(20, 3, seed=42)

    x_best, metrics = qls(
        G=G,
        budget_seconds=5,
        k=6,
        backend=backend_exact,
        n_reads=None,
        seed=42
    )

    # check assignment is valid
    for v in G.nodes():
        assert x_best[v] in (0, 1)

    # check metrics recorded something
    assert metrics.qls_calls > 0
    assert metrics.best_cut > 0
    assert len(metrics.cut_trace) > 0

    print(f"\nQLS smoke test — best cut: {metrics.best_cut}")
    metrics.summary()

def test_sa_runs_end_to_end():
    """SA should run and produce a valid cut."""
    from src.baselines import simulated_annealing

    G = generate_random_regular(20, 3, seed=42)
    x_best, metrics = simulated_annealing(G, budget_seconds=3, seed=42)

    for v in G.nodes():
        assert x_best[v] in (0, 1)
    assert metrics.best_cut > 0
    print(f"\nSA best cut: {metrics.best_cut}")


def test_tabu_runs_end_to_end():
    """Tabu search should run and produce a valid cut."""
    from src.baselines import tabu_search

    G = generate_random_regular(20, 3, seed=42)
    x_best, metrics = tabu_search(G, budget_seconds=3, seed=42)

    for v in G.nodes():
        assert x_best[v] in (0, 1)
    assert metrics.best_cut > 0
    print(f"\nTabu best cut: {metrics.best_cut}")

def test_all_selectors_return_valid_subsets():
    """All selectors should return valid vertex subsets of size <= k."""
    from src.selectors import (select_random, select_frustrated,
                                select_impact, select_meta_rule)

    G = generate_random_regular(20, 3, seed=42)
    rng = np.random.default_rng(42)
    x = random_cut(G, rng)
    gc = GainCache()
    gc.update(G, x)
    nodes = set(G.nodes())
    k = 6

    for sel_fn in [select_random, select_frustrated,
                   select_impact, select_meta_rule]:
        S = sel_fn(G, gc, k, rng=rng)
        assert len(S) <= k, f"{sel_fn.__name__} returned |S|={len(S)} > k={k}"
        assert all(v in nodes for v in S), \
            f"{sel_fn.__name__} returned vertices not in graph"
        print(f"{sel_fn.__name__}: S={S}")


def test_frustrated_vs_impact_different():
    """
    Frustrated and impact selectors should pick different vertices
    (frustrated picks low |gain|, impact picks high |gain|).
    """
    from src.selectors import select_frustrated, select_impact

    G = generate_random_regular(30, 3, seed=42)
    rng = np.random.default_rng(42)
    x = random_cut(G, rng)
    gc = GainCache()

    # run to local optimum first so gains are meaningful
    x, gc, at_opt, _ = one_flip_ls(G, x, gc)

    k = 8
    S_frustrated = set(select_frustrated(G, gc, k))
    S_impact = set(select_impact(G, gc, k))

    # they should not be identical
    assert S_frustrated != S_impact, \
        "Frustrated and impact selectors returned identical subsets"
    print(f"\nFrustrated: {sorted(S_frustrated)}")
    print(f"Impact:     {sorted(S_impact)}")

def test_adaptive_qls_all_selectors():
    """
    Adaptive QLS should run with all four selectors
    and produce valid results.
    """
    from src.adaptive_qls import adaptive_qls
    from src.backends import backend_exact
    from src.selectors import (select_random, select_frustrated,
                                select_impact, select_meta_rule)

    G = generate_random_regular(20, 3, seed=42)

    selectors = {
        'random':    select_random,
        'frustrated': select_frustrated,
        'impact':    select_impact,
        'meta_rule': select_meta_rule,
    }

    results = {}
    for name, sel in selectors.items():
        x_best, metrics = adaptive_qls(
            G=G,
            budget_seconds=5,
            selector=sel,
            backend=backend_exact,
            k_min=4,
            k_max=10,
            n_reads=None,
            seed=42
        )
        for v in G.nodes():
            assert x_best[v] in (0, 1)
        assert metrics.best_cut > 0
        results[name] = metrics.best_cut
        print(f"\nAdaptive QLS [{name}]: "
              f"cut={metrics.best_cut} "
              f"escape_rate={metrics.escape_rate():.3f} "
              f"calls={metrics.qls_calls}")

    print(f"\nResults: {results}")

def test_frustrated_connected_is_connected():
    """
    Connected frustrated selector should return a connected subgraph
    on a connected graph.
    """
    from src.selectors import select_frustrated_connected

    G = generate_random_regular(30, 3, seed=42)
    rng = np.random.default_rng(42)
    x = random_cut(G, rng)
    gc = GainCache()
    x, gc, _, _ = one_flip_ls(G, x, gc)

    k = 8
    S = select_frustrated_connected(G, gc, k)

    assert len(S) == k, f"Expected {k} vertices, got {len(S)}"

    subgraph = G.subgraph(S)
    assert nx.is_connected(subgraph), \
        f"Subgraph not connected: {list(nx.connected_components(subgraph))}"

    gains = sorted([abs(gc.gain[v]) for v in S])
    print(f"\nConnected frustrated gains: {[f'{g:.3f}' for g in gains]}")


def test_connected_vs_plain_frustrated_internal_edges():
    """
    Connected selector should have more internal edges than plain frustrated.
    More internal edges = more meaningful QUBO subproblem.
    """
    from src.selectors import select_frustrated, select_frustrated_connected

    G = generate_random_regular(50, 3, seed=42)
    rng = np.random.default_rng(42)
    x = random_cut(G, rng)
    gc = GainCache()
    x, gc, _, _ = one_flip_ls(G, x, gc)

    k = 10
    S_plain = select_frustrated(G, gc, k)
    S_connected = select_frustrated_connected(G, gc, k)

    edges_plain = G.subgraph(S_plain).number_of_edges()
    edges_connected = G.subgraph(S_connected).number_of_edges()

    print(f"\nPlain frustrated internal edges: {edges_plain}")
    print(f"Connected frustrated internal edges: {edges_connected}")

    assert edges_connected >= edges_plain, \
        f"Connected ({edges_connected}) has fewer edges than plain ({edges_plain})"
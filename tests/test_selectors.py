import networkx as nx
import numpy as np
from src.graph import generate_random_regular
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, random_cut
from src.selectors import get_selector

ALL = ['random','frustrated','frustrated_connected','impact','meta_rule',
       'clustering','fiedler','adaptive_spectral','smart_adaptive',
       'cut_polytope','topological','hybrid_topo']

def _setup(n=40, d=4, seed=1):
    G = generate_random_regular(n, d, seed=seed)
    rng = np.random.default_rng(seed)
    x = random_cut(G, rng)
    gc = GainCache()
    x, gc, _, _ = one_flip_ls(G, x, gc)   # gains meaningful at a local opt
    return G, gc, x, rng

def test_every_selector_returns_valid_subset():
    """Level 1: every selector returns <=k vertices, all in the graph."""
    G, gc, x, rng = _setup()
    nodes = set(G.nodes())
    k = 8
    pool = [random_cut(G, np.random.default_rng(s)) for s in range(6)]
    for name in ALL:
        gc.update(G, x)                    # revalidate before each call
        sel = get_selector(name)
        S = sel(G, gc, k, pool=pool, rng=rng, x=x)
        assert len(S) <= k, f"{name}: |S|={len(S)} > k={k}"
        assert len(S) == len(set(S)), f"{name}: returned duplicates"
        assert all(v in nodes for v in S), f"{name}: returned non-graph vertex"

def test_connected_selectors_are_connected():
    """Level 2: FConn and Fiedler should return connected subgraphs."""
    G, gc, x, rng = _setup()
    k = 8
    for name in ['frustrated_connected', 'fiedler']:
        gc.update(G, x)
        S = get_selector(name)(G, gc, k, rng=rng, x=x)
        assert nx.is_connected(G.subgraph(S)), f"{name}: subgraph not connected"
        
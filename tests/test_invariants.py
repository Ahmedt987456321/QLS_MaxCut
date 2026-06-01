import numpy as np
import networkx as nx
from src.gain_cache import GainCache
from src.local_search import compute_cut_value, random_cut


def _signed_graph(n=40, d=4, neg_frac=0.5, seed=0):
    """Random d-regular graph with a fraction of -1 edges (mimics G11-style ±1)."""
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(d, n, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = -1.0 if rng.random() < neg_frac else 1.0
    return G


def test_gain_equals_delta_signed_and_unsigned():
    """E2 ≡ E1: g_v(x) must equal C(flip v) - C(x) for every vertex, on
    both +1 and ±1 graphs. This certifies the entire gain machinery."""
    for neg_frac in (0.0, 0.5):                 # +1 graph, then ±1 graph
        G = _signed_graph(neg_frac=neg_frac, seed=7)
        rng = np.random.default_rng(7)
        for trial in range(50):                 # many random assignments
            x = random_cut(G, rng)
            gc = GainCache()
            gc.update(G, x)
            base = compute_cut_value(G, x)
            for v in G.nodes():
                x_flip = dict(x); x_flip[v] = 1 - x_flip[v]
                true_delta = compute_cut_value(G, x_flip) - base
                assert abs(gc.gain[v] - true_delta) < 1e-9, (
                    f"gain mismatch at v={v}: cache={gc.gain[v]} "
                    f"true={true_delta} (neg_frac={neg_frac})"
                )


def test_incremental_equals_full_recompute():
    """E4: after a sequence of incremental updates the gain dict must equal
    a fresh full recompute. Proves no gain drifts stale."""
    G = _signed_graph(neg_frac=0.5, seed=3)
    rng = np.random.default_rng(3)
    x = random_cut(G, rng)
    gc = GainCache(); gc.update(G, x)
    for _ in range(30):
        v = int(rng.integers(0, G.number_of_nodes()))
        x[v] = 1 - x[v]
        gc.incremental_update(G, x, v)
    fresh = GainCache(); fresh.update(G, x)
    for v in G.nodes():
        assert abs(gc.gain[v] - fresh.gain[v]) < 1e-9, (
            f"stale gain at v={v}: incr={gc.gain[v]} full={fresh.gain[v]}"
        )




def test_qubo_drop_equals_cut_gain():
    """E7/E8 ≡ E1: solving the local QUBO and applying it must change the
    real cut by exactly the negative of the QUBO energy change.
    Uses backend_exact as the oracle, on both +1 and ±1 graphs."""
    from src.qubo import build_local_qubo, qubo_energy, merge_proposal
    from src.backends import backend_exact
    from src.local_search import compute_cut_value, random_cut
    from src.gain_cache import GainCache
    from src.selectors import select_frustrated_connected

    for neg_frac in (0.0, 0.5):
        G = _signed_graph(neg_frac=neg_frac, seed=11)
        rng = np.random.default_rng(11)
        for _ in range(20):
            x = random_cut(G, rng)
            gc = GainCache(); gc.update(G, x)
            S = select_frustrated_connected(G, gc, k=12, rng=rng, x=x)

            Q = build_local_qubo(G, x, S)
            x_local_before = {v: x[v] for v in S}
            x_local_after = backend_exact(Q, S)

            energy_drop = qubo_energy(Q, x_local_after, S) - qubo_energy(Q, x_local_before, S)
            x_after = merge_proposal(x, x_local_after, S)
            cut_gain = compute_cut_value(G, x_after) - compute_cut_value(G, x)

            assert abs(cut_gain - (-energy_drop)) < 1e-9, (
                f"QUBO drop != cut gain (neg_frac={neg_frac}): "
                f"cut_gain={cut_gain} -energy_drop={-energy_drop}")




if __name__ == "__main__":
    test_gain_equals_delta_signed_and_unsigned()
    test_incremental_equals_full_recompute()
    test_qubo_drop_equals_cut_gain()
    print("All invariant tests PASSED")

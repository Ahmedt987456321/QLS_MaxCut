"""Check: does the number of edge-disjoint zero-cost crossings grow with L?
If yes, a simple counting argument may close the gap without near-critical
scaling theory."""
import numpy as np
import networkx as nx

def build_torus_unweighted_plus(L, M, seed):
    """Build subgraph of +1 edges only (the percolation subgraph)."""
    G = nx.grid_2d_graph(L, M, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u,v in G.edges():
        if rng.random() < 0.5:   # p_c = 1/2: each edge +1 with prob 1/2
            H.add_edge(u, v)
    return H

def count_edge_disjoint_crossings(H, L, M):
    """Count max number of edge-disjoint zero-cost crossings using
    max-flow (= max edge-disjoint paths) from column M/2-1 to M/2."""
    F = nx.DiGraph()
    for u,v in H.edges():
        F.add_edge(u, v, capacity=1)
        F.add_edge(v, u, capacity=1)
    SOURCE = L*M; SINK = L*M+1
    for i in range(L):
        F.add_edge(SOURCE, i*M + M//2-1, capacity=L)
        F.add_edge(i*M + M//2, SINK, capacity=L)
    try:
        flow = nx.maximum_flow_value(F, SOURCE, SINK)
        return int(flow)
    except:
        return 0

print(f"{'L':>5} {'M':>5} {'seeds':>6} {'mean_disjoint':>14} "
      f"{'min_disjoint':>13} {'P(>=1)':>8} {'P(>=2)':>8}")
print("-"*70)
SEEDS = 100
for L in [4, 8, 16, 32]:
    M = 4*L
    counts = []
    for seed in range(SEEDS):
        H = build_torus_unweighted_plus(L, M, seed)
        c = count_edge_disjoint_crossings(H, L, M)
        counts.append(c)
    mean = np.mean(counts)
    mn = min(counts)
    p1 = sum(1 for c in counts if c >= 1)/SEEDS
    p2 = sum(1 for c in counts if c >= 2)/SEEDS
    print(f"{L:5d} {M:5d} {SEEDS:6d} {mean:14.3f} {mn:13d} {p1:8.3f} {p2:8.3f}")

print("\nIf mean_disjoint -> infinity: counting argument closes the gap.")
print("If min_disjoint >= 1 always: trivially P(crossing)=1 (too strong).")
print("If P(>=1) -> 1 but min_disjoint stays at 0 sometimes: need more.")

"""Phase 3b setup: measure the ACTUAL correlated crossing cost on the +-J
torus across multiple L values. Compare to:
- O(1): the physical saturation prediction (Hartmann-Young, Khoshbakht-Weigel)
- O(log L): the i.i.d. FPP prediction (Kesten + CCD)
- O(L): the worst case (Theorem 1)

This tells us which regime we are actually in and what the proof needs to show."""
import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu

def build_torus(L, M, seed):
    G = nx.grid_2d_graph(L, M, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    for u,v in G.edges():
        G[u][v]["weight"] = int(rng.choice([-1,1]))
    return G

def fiedler_strip(L, M):
    """Vertices in the Fiedler strip: columns 0..M/2-1."""
    return set(i*M + j for i in range(L) for j in range(M//2))

def min_crossing_cost(G, L, M):
    """Minimum number of -1 (frustrated) edges on any path crossing
    the Fiedler boundary (between columns M/2-1 and M/2).
    Uses Dijkstra with {0,1} dual weights: w*=0 if +1, w*=1 if -1."""
    # Build auxiliary graph: nodes = original vertices + source + sink
    # Source connects to all vertices in column M/2-1
    # Sink connects from all vertices in column M/2
    # Edge weights: w* = (1 - w_orig) / 2
    H = nx.DiGraph()
    for u,v in G.edges():
        w = G[u][v]["weight"]
        w_dual = 0 if w == 1 else 1
        H.add_edge(u, v, weight=w_dual)
        H.add_edge(v, u, weight=w_dual)
    # add source and sink
    SOURCE = L*M
    SINK = L*M + 1
    for i in range(L):
        left_col = i*M + (M//2 - 1)   # last column of strip
        right_col = i*M + M//2          # first column outside strip
        H.add_edge(SOURCE, left_col, weight=0)
        H.add_edge(right_col, SINK, weight=0)
    # shortest path = minimum frustrated edges to cross
    try:
        length = nx.shortest_path_length(H, SOURCE, SINK, weight="weight")
        return length
    except nx.NetworkXNoPath:
        return float("inf")

# sweep L values; fix M = 2*L (elongated torus to keep boundary = 2L)
# and M = L (square torus) to see both regimes
print(f"{'L':>5} {'M':>5} {'n':>6} {'|?S*|=2L':>10} "
      f"{'mean_cross':>12} {'std':>8} {'cross/L':>10} {'cross/logL':>12}")
print("-"*85)

results = {}
SEEDS = 20
for L in [4, 8, 12, 16, 20, 24]:
    M = 4*L   # elongated: boundary = 2L
    costs = []
    for seed in range(SEEDS):
        G = build_torus(L, M, seed)
        c = min_crossing_cost(G, L, M)
        if c < float("inf"):
            costs.append(c)
    if costs:
        mean_c = np.mean(costs)
        std_c = np.std(costs)
        import math
        log_L = math.log(L) if L > 1 else 1
        print(f"{L:5d} {M:5d} {L*M:6d} {2*L:10d} "
              f"{mean_c:12.3f} {std_c:8.3f} {mean_c/L:10.4f} {mean_c/log_L:12.4f}")
        results[L] = {"M":M,"mean":mean_c,"std":std_c}

print("\nREAD:")
print("  cross/L -> 0 as L grows  => sub-linear (o(L)) confirmed")
print("  cross/logL -> constant   => O(log L) (i.i.d. FPP prediction)")
print("  mean_cross -> constant   => O(1) (physical saturation prediction)")
print("  cross/L -> constant      => O(L) (worst case -- would be bad)")

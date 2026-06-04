"""Phase 3b deeper verification: is the crossing cost truly zero,
or just very small? Test more seeds at L=8 and check for any non-zero cases.
Also test the crossing probability (fraction of instances with cost=0)."""
import numpy as np
import networkx as nx

def build_torus(L, M, seed):
    G = nx.grid_2d_graph(L, M, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    for u,v in G.edges():
        G[u][v]["weight"] = int(rng.choice([-1,1]))
    return G

def min_crossing_cost(G, L, M):
    H = nx.DiGraph()
    for u,v in G.edges():
        w = G[u][v]["weight"]
        H.add_edge(u, v, weight=0 if w==1 else 1)
        H.add_edge(v, u, weight=0 if w==1 else 1)
    SOURCE = L*M; SINK = L*M+1
    for i in range(L):
        H.add_edge(SOURCE, i*M + M//2-1, weight=0)
        H.add_edge(i*M + M//2, SINK, weight=0)
    try:
        return nx.shortest_path_length(H, SOURCE, SINK, weight="weight")
    except:
        return float("inf")

print("Deep verification: crossing cost distribution")
print(f"{'L':>4} {'M':>4} {'seeds':>6} {'cost=0':>8} {'cost=1':>8} "
      f"{'cost?2':>8} {'P(cost=0)':>10}")
print("-"*60)
SEEDS = 100
for L in [4, 8, 16, 32]:
    M = 4*L
    costs = [min_crossing_cost(build_torus(L,M,s), L, M) for s in range(SEEDS)]
    c0 = sum(1 for c in costs if c==0)
    c1 = sum(1 for c in costs if c==1)
    c2 = sum(1 for c in costs if c>=2)
    print(f"{L:4d} {M:4d} {SEEDS:6d} {c0:8d} {c1:8d} {c2:8d} {c0/SEEDS:10.3f}")

print("\nKEY QUESTION: does P(cost=0) -> 1 as L grows?")
print("If yes: the crossing cost is TRIVIALLY o(L) because it's")
print("essentially zero -- the percolation structure at p_c explains it.")
print("If P(cost=0) -> some constant < 1: the O(log L) regime is real.")

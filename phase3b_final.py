"""Phase 3b final verification: measure P(cost=0) for larger L to
confirm the limit is 1, not just a large finite-size constant."""
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

# test larger L and different aspect ratios
print(f"{'L':>5} {'M':>5} {'aspect':>8} {'seeds':>6} "
      f"{'P(cost=0)':>10} {'mean_cost':>10}")
print("-"*55)
for L,M in [(4,16),(8,32),(16,64),(32,128),(8,16),(8,64)]:
    SEEDS = 200
    costs = [min_crossing_cost(build_torus(L,M,s),L,M) for s in range(SEEDS)]
    p0 = sum(1 for c in costs if c==0)/SEEDS
    mean = np.mean(costs)
    aspect = M/(2*L) if L>0 else 0  # strip aspect ratio M/2 : L
    print(f"{L:5d} {M:5d} {aspect:8.2f} {SEEDS:6d} {p0:10.3f} {mean:10.4f}")
print("\nAspect ratio = (strip width M/2) / (strip height L)")
print("Higher aspect ratio = wider strip = easier to cross")
print("At p_c, crossing prob depends on aspect ratio (Cardy formula)")

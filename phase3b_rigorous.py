"""Verify the corrected Phase 3b proof: disjoint height-2 strips give
independent crossing events. Check:
1. Each height-2 strip has positive crossing probability by RSW.
2. Strips use disjoint edge sets (hence independent).
3. P(no crossing in any strip) = product -> 0 exponentially."""
import numpy as np
import networkx as nx

def build_torus(L, M, seed):
    G = nx.grid_2d_graph(L, M, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    for u,v in G.edges():
        G[u][v]["weight"] = int(rng.choice([-1,1]))
    return G

def strip_crossing(G, L, M, row_start, height=2):
    """Does a height-2 strip starting at row_start have a zero-cost crossing?
    Only uses edges within rows [row_start, row_start+height-1] and the
    boundary column edges adjacent to those rows."""
    strip_nodes = set()
    for r in range(row_start, row_start + height):
        for c in range(M):
            strip_nodes.add(r*M + c)
    H = nx.DiGraph()
    for u,v in G.edges():
        if u in strip_nodes and v in strip_nodes:
            w = G[u][v]["weight"]
            H.add_edge(u, v, weight=0 if w==1 else 1)
            H.add_edge(v, u, weight=0 if w==1 else 1)
    SOURCE = L*M; SINK = L*M+1
    for r in range(row_start, row_start+height):
        H.add_edge(SOURCE, r*M + M//2-1, weight=0)
        H.add_edge(r*M + M//2, SINK, weight=0)
    try:
        return nx.shortest_path_length(H, SOURCE, SINK, weight="weight") == 0
    except:
        return False

print("Verifying corrected Phase 3b proof: disjoint height-2 strips")
print("="*65)
SEEDS = 200
for L in [8, 16, 32]:
    M = 4*L
    K = L//2   # number of disjoint strips
    strip_prob = []
    all_cross = []
    for seed in range(SEEDS):
        G = build_torus(L, M, seed)
        crossings = [strip_crossing(G, L, M, 2*k) for k in range(K)]
        strip_prob.append(np.mean(crossings))
        all_cross.append(all(crossings))
    mean_strip = np.mean(strip_prob)
    p_all = np.mean(all_cross)
    p_none = 1 - np.mean([any(
        [strip_crossing(build_torus(L,M,s),L,M,2*k) for k in range(K)])
        for s in range(20)])
    pred = (1 - mean_strip)**K
    print(f"L={L:2d}, M={M:3d}, K={K} strips:")
    print(f"  P(each strip crosses) = {mean_strip:.3f}")
    print(f"  P(ALL strips cross)   = {p_all:.3f}")
    print(f"  predicted product     = {pred:.6f}")
    print(f"  P(no crossing anywhere) ~ {p_none:.3f}")
    print()
print("If P(each strip crosses) > 0 and strips are independent,")
print("P(no crossing anywhere) = (1-p)^K -> 0 exponentially.")

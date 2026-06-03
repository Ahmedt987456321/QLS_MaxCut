"""Check variance of edge-disjoint crossing count to apply second-moment
method: if Var(N) = o((E[N])^2), then P(N>=1) -> 1 by Chebyshev."""
import numpy as np
import networkx as nx

def build_plus(L, M, seed):
    G = nx.grid_2d_graph(L, M, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u,v in G.edges():
        if rng.random() < 0.5:
            H.add_edge(u, v)
    return H

def count_disjoint(H, L, M):
    F = nx.DiGraph()
    for u,v in H.edges():
        F.add_edge(u, v, capacity=1)
        F.add_edge(v, u, capacity=1)
    S = L*M; T = L*M+1
    for i in range(L):
        F.add_edge(S, i*M+M//2-1, capacity=L)
        F.add_edge(i*M+M//2, T, capacity=L)
    try:
        return int(nx.maximum_flow_value(F, S, T))
    except:
        return 0

print(f"{'L':>5} {'E[N]':>8} {'Var(N)':>10} {'Var/E[N]^2':>12} "
      f"{'CV=std/mean':>12} {'P(N>=1)':>8}")
print("-"*65)
SEEDS = 200
for L in [4, 8, 16, 32]:
    M = 4*L
    counts = [count_disjoint(build_plus(L,M,s),L,M) for s in range(SEEDS)]
    EN = np.mean(counts)
    VarN = np.var(counts)
    CV = np.std(counts)/EN if EN > 0 else float("inf")
    p1 = sum(1 for c in counts if c>=1)/SEEDS
    print(f"{L:5d} {EN:8.3f} {VarN:10.3f} {VarN/EN**2:12.4f} "
          f"{CV:12.4f} {p1:8.3f}")

print("\nIf Var/E[N]^2 -> 0: second moment method gives P(N>=1) -> 1.")
print("If CV=std/mean -> 0: N concentrates around mean -> P(N>=1) -> 1.")
print("Chebyshev: P(N=0) <= P(|N-E[N]|>=E[N]) <= Var(N)/E[N]^2")

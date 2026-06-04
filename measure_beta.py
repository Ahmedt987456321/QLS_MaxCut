import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def fiedler_boundary(G):
    nodes = list(G.nodes())
    n = len(nodes)
    G_unw = nx.Graph()
    G_unw.add_nodes_from(nodes)
    G_unw.add_edges_from(G.edges())
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    idx = np.argsort(vals)
    v = vecs[:, idx[1]]
    k = n // 2
    S = set(nodes[i] for i in np.argsort(v)[-k:])
    cut = sum(1 for u,w in G_unw.edges() if (u in S) != (w in S))
    beta = cut / n
    lam2 = nx.algebraic_connectivity(G_unw, method="lanczos")
    return beta, lam2, cut, n

TAU_LAM2 = 0.2
BETA_STAR = 0.10

print(f"{'instance':25} {'n':>5} {'lambda2':>9} {'beta':>8} "
      f"{'lam2_route':>12} {'beta_route':>12} {'actual':>10}")
print("-"*85)

gset_winners = {
    "G11":"Fiedler","G12":"Fiedler","G13":"Fiedler",
    "G32":"Fiedler","G33":"Fiedler","G34":"Fiedler",
    "G1":"FConn","G22":"FConn"
}
for name, actual in gset_winners.items():
    try:
        G = load_gset(f"data/gset/{name}.txt")
        beta, lam2, cut, n = fiedler_boundary(G)
        r_lam2 = "Fiedler" if lam2 < TAU_LAM2 else "FConn"
        r_beta = "Fiedler" if beta < BETA_STAR else "FConn"
        print(f"{name:25} {n:5d} {lam2:9.4f} {beta:8.4f} "
              f"{r_lam2+(' ?' if r_lam2==actual else ' ?'):>12} "
              f"{r_beta+(' ?' if r_beta==actual else ' ?'):>12} {actual:>10}")
    except Exception as e:
        print(f"{name}: {e}")

for seed in range(3):
    G = nx.random_regular_graph(3, 800, seed=seed)
    rng = np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    beta, lam2, cut, n = fiedler_boundary(G)
    r_lam2 = "Fiedler" if lam2 < TAU_LAM2 else "FConn"
    r_beta  = "Fiedler" if beta  < BETA_STAR else "FConn"
    actual  = "FConn"
    print(f"{'d3-reg-s'+str(seed):25} {n:5d} {lam2:9.4f} {beta:8.4f} "
          f"{r_lam2+(' ?' if r_lam2==actual else ' ?'):>12} "
          f"{r_beta+(' ?' if r_beta==actual else ' ?'):>12} {actual:>10}")

for seed in range(2):
    G = nx.random_regular_graph(4, 800, seed=seed)
    rng = np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    beta, lam2, cut, n = fiedler_boundary(G)
    r_lam2 = "Fiedler" if lam2 < TAU_LAM2 else "FConn"
    r_beta  = "Fiedler" if beta  < BETA_STAR else "FConn"
    actual  = "FConn"
    print(f"{'d4-reg-s'+str(seed):25} {n:5d} {lam2:9.4f} {beta:8.4f} "
          f"{r_lam2+(' ?' if r_lam2==actual else ' ?'):>12} "
          f"{r_beta+(' ?' if r_beta==actual else ' ?'):>12} {actual:>10}")

print(f"\nThreshold: lambda2 < {TAU_LAM2}, beta < {BETA_STAR}")

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def fiedler_beta(G):
    """
    Compute normalised Fiedler boundary beta.
    beta = |cut_edges| / |E|  (fraction of edges on boundary)
    This is scale-invariant across sparse and dense graphs.
    Also compute conductance phi = |cut_edges| / (d_avg * k)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    G_unw = nx.Graph()
    G_unw.add_nodes_from(nodes)
    G_unw.add_edges_from(G.edges())
    m = G_unw.number_of_edges()
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    v = vecs[:, np.argsort(vals)[1]]
    k = n // 2
    S = set(nodes[i] for i in np.argsort(v)[-k:])
    cut = sum(1 for u,w in G_unw.edges() if (u in S) != (w in S))
    beta_n   = cut / n          # per-vertex (fails on dense)
    beta_e   = cut / m          # per-edge (scale-invariant)
    d_avg    = 2*m/n
    phi      = cut / (d_avg * k) # conductance
    lam2     = nx.algebraic_connectivity(G_unw, method="lanczos")
    return dict(n=n, m=m, lam2=lam2, cut=cut,
                beta_n=beta_n, beta_e=beta_e, phi=phi, d_avg=d_avg)

BETA_STAR_E = 0.10   # threshold on beta_e
PHI_STAR    = 0.05   # threshold on conductance

print(f"{'instance':22} {'n':>5} {'d_avg':>6} {'lam2':>8} "
      f"{'beta_e':>8} {'phi':>8} {'route_beta_e':>14} {'actual':>10}")
print("-"*90)

instances = [
    ("G11","Fiedler"),("G12","Fiedler"),("G13","Fiedler"),
    ("G32","Fiedler"),("G33","Fiedler"),("G34","Fiedler"),
    ("G1","FConn"),("G22","FConn"),
]
for name,actual in instances:
    try:
        G = load_gset(f"data/gset/{name}.txt")
        r = fiedler_beta(G)
        route = "Fiedler" if r['beta_e'] < BETA_STAR_E else "FConn"
        ok = "?" if route==actual else "?"
        print(f"{name:22} {r['n']:5d} {r['d_avg']:6.1f} {r['lam2']:8.4f} "
              f"{r['beta_e']:8.4f} {r['phi']:8.4f} {route+ok:>14} {actual:>10}")
    except Exception as e:
        print(f"{name}: {e}")

# d=3 random regular
for seed in range(3):
    G = nx.random_regular_graph(3, 800, seed=seed)
    rng = np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    r = fiedler_beta(G)
    route = "Fiedler" if r['beta_e'] < BETA_STAR_E else "FConn"
    ok = "?" if route=="FConn" else "?"
    print(f"{'d3-reg-s'+str(seed):22} {r['n']:5d} {r['d_avg']:6.1f} "
          f"{r['lam2']:8.4f} {r['beta_e']:8.4f} {r['phi']:8.4f} "
          f"{route+ok:>14} {'FConn':>10}")

# d=4 regular
for seed in range(2):
    G = nx.random_regular_graph(4, 800, seed=seed)
    rng = np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    r = fiedler_beta(G)
    route = "Fiedler" if r['beta_e'] < BETA_STAR_E else "FConn"
    ok = "?" if route=="FConn" else "?"
    print(f"{'d4-reg-s'+str(seed):22} {r['n']:5d} {r['d_avg']:6.1f} "
          f"{r['lam2']:8.4f} {r['beta_e']:8.4f} {r['phi']:8.4f} "
          f"{route+ok:>14} {'FConn':>10}")

print(f"\nbeta_e = cut_edges / total_edges (fraction of edges on boundary)")
print(f"phi    = cut_edges / (d_avg * k)  (conductance of Fiedler cut)")
print(f"Threshold: beta_e < {BETA_STAR_E} -> Fiedler, else FConn")
print(f"\nExpected:")
print(f"  G-set toroidal: beta_e << 0.10 (sparse boundary)")
print(f"  d=3 regular:    beta_e ~ 0.16  (dense boundary despite low lam2)")
print(f"  G1/G22/dense:   beta_e ~ 0.45  (nearly half edges on boundary)")

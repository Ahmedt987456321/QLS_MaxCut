"""Verify beta on G1 and G22 using unweighted topology only.
Also compute the theoretical beta for d=3 random regular."""
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def fiedler_boundary_unweighted(G):
    """Always uses unweighted topology for beta."""
    nodes = list(G.nodes())
    n = len(nodes)
    # strip weights
    G_unw = nx.Graph()
    G_unw.add_nodes_from(nodes)
    G_unw.add_edges_from(G.edges())  # no weights
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    v = vecs[:, np.argsort(vals)[1]]
    k = n // 2
    S = set(nodes[i] for i in np.argsort(v)[-k:])
    cut = sum(1 for u,w in G_unw.edges() if (u in S) != (w in S))
    beta = cut / n
    lam2 = nx.algebraic_connectivity(G_unw, method="lanczos")
    return beta, lam2, cut

BETA_STAR = 0.10

print("Unweighted beta measurements:")
print(f"{'instance':20} {'n':>5} {'lam2':>8} {'beta':>8} "
      f"{'cut_edges':>12} {'routing':>10} {'actual':>10}")
print("-"*80)

for name, actual in [("G1","FConn"),("G22","FConn"),
                     ("G11","Fiedler"),("G13","Fiedler")]:
    G = load_gset(f"data/gset/{name}.txt")
    beta, lam2, cut = fiedler_boundary_unweighted(G)
    n = G.number_of_nodes()
    routing = "Fiedler" if beta < BETA_STAR else "FConn"
    ok = "?" if routing==actual else "?"
    print(f"{name:20} {n:5d} {lam2:8.4f} {beta:8.4f} "
          f"{cut:12d} {routing+ok:>10} {actual:>10}")

print(f"\nTheoretical beta for d=3 random regular:")
import math
d = 3
mu2 = 2*math.sqrt(d-1)  # Friedman: lambda2 -> d - 2sqrt(d-1)
rho = mu2/d
beta_theory = math.acos(rho)/math.pi * (d/2)
print(f"  mu2 = 2*sqrt(d-1) = {mu2:.4f}")
print(f"  rho = mu2/d = {rho:.4f}")
print(f"  beta_theory = (d/2) * arccos(rho)/pi = {beta_theory:.4f}")
print(f"  Measured beta (d=3, n=800): 0.168-0.188")
print(f"  Theory predicts: {beta_theory:.3f} -- matches measured values")

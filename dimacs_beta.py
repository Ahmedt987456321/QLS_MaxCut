import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def load_dimacs(path):
    """Load DIMACS .dat format: first line is 'n m', rest are 'u v w'."""
    G = nx.Graph()
    with open(path) as f:
        lines = f.readlines()
    n, m = map(int, lines[0].split())
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            G.add_edge(u, v, weight=w)
    return G

def fiedler_beta_e(G):
    nodes = list(G.nodes()); n = len(nodes)
    G_unw = nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m = G_unw.number_of_edges()
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    v = vecs[:, np.argsort(vals)[1]]
    k = n // 2
    S = set(nodes[i] for i in np.argsort(v)[-k:])
    cut = sum(1 for u,w in G_unw.edges() if (u in S) != (w in S))
    lam2 = nx.algebraic_connectivity(G_unw, method="lanczos")
    beta_e = cut / m
    return {"n":n,"m":m,"lam2":lam2,"cut":cut,"beta_e":beta_e,
            "lam2_route":"Fiedler" if lam2<0.2 else "FConn",
            "beta_route":"Fiedler" if beta_e<0.10 else "FConn"}

INSTANCES = {
    "torusg3-8":      "data/dimacs/torusg3-8.dat",
    "toruspm3-8-50":  "data/dimacs/toruspm3-8-50.dat",
    "torusg3-15":     "data/dimacs/torusg3-15.dat",
    "toruspm3-15-50": "data/dimacs/toruspm3-15-50.dat",
}

# Best known values (Rehfeldt-Koch-Shinano 2023)
BEST_KNOWN = {
    "torusg3-8":      None,    # proven optimal -- value unknown
    "toruspm3-8-50":  None,    # proven optimal -- value unknown
    "torusg3-15":     286626481,  # PROVEN OPTIMAL
    "toruspm3-15-50": 3010,    # best-known, NOT proven optimal (1.8% gap)
}

print(f"{'instance':20} {'n':>5} {'d':>4} {'lam2':>8} {'beta_e':>8} "
      f"{'lam2_route':>12} {'beta_route':>12}")
print("-"*80)

results = {}
for name, path in INSTANCES.items():
    G = load_dimacs(path)
    r = fiedler_beta_e(G)
    d_avg = 2*r['m']/r['n']
    print(f"{name:20} {r['n']:5d} {d_avg:4.1f} {r['lam2']:8.4f} "
          f"{r['beta_e']:8.4f} {r['lam2_route']:>12} {r['beta_route']:>12}")
    results[name] = r

print("\nExpected:")
print("  3D torus (genus>1): beta_e could be higher than 2D torus")
print("  These are 3D periodic lattices -- NOT genus-1 like G-set toroidal")
print("  Proposition 1 applies to genus-1; genus>1 may have larger boundary")
print("\nPrediction check:")
print("  If beta_e < 0.10 -> Fiedler should win")
print("  If beta_e > 0.10 -> FConn should win")
print("  We will run AQLS to verify after measuring")

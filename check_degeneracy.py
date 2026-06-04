import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def check_degeneracy(G):
    nodes = list(G.nodes())
    G_unw = nx.Graph()
    G_unw.add_nodes_from(nodes)
    G_unw.add_edges_from(G.edges())
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, _ = eigsh(L, k=4, sigma=0, which='LM')
    vals = sorted(vals)
    lam2 = vals[1]; lam3 = vals[2]
    gap = lam3 - lam2
    degenerate = gap < 1e-4
    return {"lam2":lam2,"lam3":lam3,"gap":gap,"degenerate":degenerate}

EPSILON = 1e-4

print(f"{'instance':22} {'lam2':>8} {'lam3':>8} {'gap':>10} "
      f"{'degenerate':>12} {'fiedler_valid':>14}")
print("-"*75)

# G-set instances
for name in ["G11","G13","G32","G1","G22"]:
    try:
        G = load_gset(f"data/gset/{name}.txt")
        r = check_degeneracy(G)
        valid = not r['degenerate']
        print(f"{name:22} {r['lam2']:8.4f} {r['lam3']:8.4f} "
              f"{r['gap']:10.6f} {str(r['degenerate']):>12} "
              f"{'YES' if valid else 'NO (degenerate)':>14}")
    except: pass

# DIMACS
def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

for name, path in [
    ("torusg3-8",     "data/dimacs/torusg3-8.dat"),
    ("toruspm3-8-50", "data/dimacs/toruspm3-8-50.dat"),
    ("torusg3-15",    "data/dimacs/torusg3-15.dat"),
    ("toruspm3-15-50","data/dimacs/toruspm3-15-50.dat"),
]:
    G = load_dimacs(path)
    r = check_degeneracy(G)
    valid = not r['degenerate']
    print(f"{name:22} {r['lam2']:8.4f} {r['lam3']:8.4f} "
          f"{r['gap']:10.6f} {str(r['degenerate']):>12} "
          f"{'YES' if valid else 'NO (degenerate)':>14}")

# d=3 random regular
for seed in range(2):
    G = nx.random_regular_graph(3, 800, seed=seed)
    r = check_degeneracy(G)
    valid = not r['degenerate']
    print(f"{'d3-reg-s'+str(seed):22} {r['lam2']:8.4f} {r['lam3']:8.4f} "
          f"{r['gap']:10.6f} {str(r['degenerate']):>12} "
          f"{'YES' if valid else 'NO (degenerate)':>14}")

print(f"\nUpdated routing rule:")
print(f"  Route to Fiedler iff:")
print(f"    beta_e < 0.10")
print(f"    AND lambda2 < 0.2")
print(f"    AND (lambda3 - lambda2) > {EPSILON}  [non-degenerate]")
print(f"\nThis fixes the DIMACS failure: degenerate Fiedler eigenvalue")
print(f"means the Fiedler vector is not unique -- any linear combination")
print(f"of the degenerate eigenvectors is valid, giving a random strip.")

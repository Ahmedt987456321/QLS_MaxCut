"""The real diagnostic: on degenerate instances, does the Fiedler
eigenvector returned by eigsh have a dominant spatial direction?
Measure: ratio of max to min Fiedler vector variance across the
two (or three) spatial dimensions of the lattice."""
import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def fiedler_asymmetry(G, name):
    nodes = list(G.nodes())
    n = len(nodes)
    G_unw = nx.Graph()
    G_unw.add_nodes_from(nodes)
    G_unw.add_edges_from(G.edges())
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=4, sigma=0, which='LM')
    idx = np.argsort(vals)
    vals = vals[idx]; vecs = vecs[:, idx]
    lam2 = vals[1]; lam3 = vals[2]
    gap = lam3 - lam2
    v2 = vecs[:, 1]  # Fiedler vector
    v3 = vecs[:, 2]  # next eigenvector
    # Spatial asymmetry: how much does v2 vary vs v3?
    # On asymmetric 2D torus: v2 varies a lot (long direction)
    # v3 varies less (short direction)
    # On symmetric 3D torus: v2 and v3 vary equally
    var2 = np.var(v2)
    var3 = np.var(v3)
    asym = max(var2,var3)/min(var2,var3) if min(var2,var3)>1e-10 else 1.0
    # Also check: does v2 look like a clean gradient?
    # Compute std of v2 values on each "row" if we know structure
    print(f"{name:22} lam2={lam2:.4f} gap={gap:.6f} "
          f"var2={var2:.6f} var3={var3:.6f} asymmetry={asym:.2f}")
    return asym

def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

print(f"{'instance':22} {'lam2':>8} {'gap':>10} "
      f"{'var(v2)':>10} {'var(v3)':>10} {'asymmetry':>10}")
print("-"*75)
print("G-set (Fiedler wins):")
for name in ["G11","G13","G32"]:
    G = load_gset(f"data/gset/{name}.txt")
    fiedler_asymmetry(G, name)

print("\nDIMACS 3D torus (FConn wins):")
for name, path in [
    ("torusg3-15",    "data/dimacs/torusg3-15.dat"),
    ("toruspm3-15-50","data/dimacs/toruspm3-15-50.dat"),
]:
    G = load_dimacs(path)
    fiedler_asymmetry(G, name)

print("\nG-set dense (FConn wins):")
for name in ["G1","G22"]:
    G = load_gset(f"data/gset/{name}.txt")
    fiedler_asymmetry(G, name)

print("\nHYPOTHESIS: G11/G13/G32 should have HIGH asymmetry (var2 >> var3)")
print("because 2D torus is asymmetric (L<<M), one direction dominates.")
print("DIMACS 3D torus should have LOW asymmetry (var2 ? var3)")
print("because 3D cubic torus is symmetric (L=M=15), all directions equal.")
print("\nIf confirmed: asymmetry ratio is the correct routing signal,")
print("not degeneracy per se.")

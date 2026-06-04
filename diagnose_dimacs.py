"""Diagnose DIMACS L=15 failure: why does FConn win despite beta_e=0.075?
Check: Fiedler vector character on 3D torus vs 2D torus.
Check: k=640 appropriate for n=3375?
Check: do Gaussian weights give FConn a structural advantage?"""
import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh

def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

def analyse(name, path):
    G = load_dimacs(path)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    d_avg = 2*m/n
    weights = [abs(G[u][v]["weight"]) for u,v in G.edges()]

    # Fiedler vector character
    G_unw = nx.Graph(); G_unw.add_nodes_from(G.nodes()); G_unw.add_edges_from(G.edges())
    L = nx.laplacian_matrix(G_unw, nodelist=list(G.nodes())).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    v = vecs[:, np.argsort(vals)[1]]
    # smoothness: std of differences across edges
    edge_diffs = [abs(v[list(G.nodes()).index(u)] -
                     v[list(G.nodes()).index(w)])
                 for u,w in list(G.edges())[:1000]]

    print(f"\n{name}: n={n}, m={m}, d_avg={d_avg:.1f}")
    print(f"  Weight range: [{min(weights):.1f}, {max(weights):.1f}]")
    print(f"  Weight std: {np.std(weights):.1f}")
    print(f"  lambda2: {vals.min():.4f}")
    print(f"  Fiedler vector: min={v.min():.4f} max={v.max():.4f} std={v.std():.4f}")
    print(f"  Edge diff mean: {np.mean(edge_diffs):.4f} (smoothness)")
    print(f"  Optimal k (n/8): {n//8}, used k=640")

for name, path in [
    ("torusg3-15",    "data/dimacs/torusg3-15.dat"),
    ("toruspm3-15-50","data/dimacs/toruspm3-15-50.dat"),
]:
    analyse(name, path)

# Compare with G11 for reference
from src.graph import load_gset
G11 = load_gset("data/gset/G11.txt")
G_unw = nx.Graph(); G_unw.add_nodes_from(G11.nodes()); G_unw.add_edges_from(G11.edges())
L = nx.laplacian_matrix(G_unw, nodelist=list(G11.nodes())).astype(float)
vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
v = vecs[:, np.argsort(vals)[1]]
print(f"\nG11 (reference): n=800, d=4, lambda2={vals.min():.4f}")
print(f"  Fiedler: min={v.min():.4f} max={v.max():.4f} std={v.std():.4f}")
print(f"  Optimal k used: 400 (50% of graph)")
print(f"\nKEY: if Fiedler vector std is similar on 3D torus and G11,")
print(f"the issue is k value or weight scale, not Fiedler character.")

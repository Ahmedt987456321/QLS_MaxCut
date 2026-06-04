"""Step 1.2 corrected: use UNSIGNED Laplacian eigenvectors (no weights).
The Fiedler selector uses the unsigned topology, not the signed instance."""
import numpy as np
import networkx as nx
from src.graph import load_gset

instances = {"G11":(8,100),"G13":(32,25),"G32":(50,40),"G33":(40,50),"G34":(50,40)}

print(f"{'inst':6} {'L':>4} {'M':>4} {'|S|':>6} {'|?S|pred':>10} "
      f"{'|?S|actual':>12} {'phi_pred':>10} {'phi_actual':>12} {'match?':>8}")
print("-"*80)
for name,(L,M) in instances.items():
    G = load_gset(f"data/gset/{name}.txt")
    n = G.number_of_nodes()
    k = n//2
    # UNSIGNED Laplacian -- remove weights
    G_unw = nx.Graph()
    G_unw.add_nodes_from(G.nodes())
    G_unw.add_edges_from(G.edges())   # no weights
    L_mat = np.array(nx.laplacian_matrix(G_unw).todense(), dtype=float)
    evecs = np.linalg.eigh(L_mat)[1]
    fv = evecs[:, 1]   # unsigned Fiedler vector
    S = set(i for i in range(n) if fv[i] >= 0)
    bdy = sum(1 for u,v in G_unw.edges() if (u in S)!=(v in S))
    phi_actual = bdy/len(S)
    phi_pred = 4/M
    bdy_pred = 2*L
    match = abs(phi_actual - phi_pred) < 0.01
    print(f"{name:6} {L:4d} {M:4d} {len(S):6d} {bdy_pred:10d} "
          f"{bdy:12d} {phi_pred:10.4f} {phi_actual:12.4f} "
          f"{'YES' if match else 'NO':>8}")

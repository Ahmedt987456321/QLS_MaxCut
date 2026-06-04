"""Check G11 specifically -- try various factorizations."""
import numpy as np
n = 800
lam2_measured = 0.0039
print("G11: checking all factor pairs of 800")
print(f"{'L':>5} {'M':>5} {'lambda2':>10} {'error':>10}")
for L in range(2, 401):
    if n % L == 0:
        M = n // L
        lam2 = 4 * np.sin(np.pi / max(L,M))**2
        err = abs(lam2 - lam2_measured)
        if err < 0.001:
            print(f"{L:5d} {M:5d} {lam2:10.6f} {err:10.6f}")
print("\nIf nothing prints, G11 may not be a simple rectangular torus,")
print("OR the measured lambda2 is from the SIGNED Laplacian (+-1 weights)")
print("rather than the unsigned topology.")
# also check: what does networkx give for the actual G11 file?
import os
if os.path.exists("data/gset/G11.txt"):
    import networkx as nx
    from src.graph import load_gset
    G = load_gset("data/gset/G11.txt")
    lam2_actual = nx.algebraic_connectivity(G, method="lanczos")
    print(f"\nActual unsigned lambda2 of G11 graph: {lam2_actual:.6f}")
    print(f"n={G.number_of_nodes()}, edges={G.number_of_edges()}, "
          f"avg_degree={2*G.number_of_edges()/G.number_of_nodes():.1f}")

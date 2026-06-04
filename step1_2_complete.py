"""Step 1.2 complete: compute exact Fiedler support geometry for all
confirmed toroidal instances. These numbers go into Proposition 1."""
import numpy as np
import networkx as nx
from src.graph import load_gset

instances = {
    "G11": (8,   100),
    "G13": (32,  25),
    "G32": (50,  40),
    "G33": (40,  50),
    "G34": (50,  40),
}

print(f"{'inst':6} {'L':>4} {'M':>4} {'n':>5} {'k=n/2':>6} "
      f"{'|?S|=2L':>8} {'phi=|?S|/k':>12} {'4/M':>8} {'1/sqrt(n)':>10}")
print("-"*75)
for name,(L,M) in instances.items():
    n = L*M
    k = n//2           # support size = half the graph
    boundary = 2*L     # two horizontal strip boundaries, each L edges
    phi = boundary/k
    pred_phi = 4/M     # analytical prediction: phi = 4/M
    inv_sqrtn = 1/np.sqrt(n)
    print(f"{name:6} {L:4d} {M:4d} {n:5d} {k:6d} "
          f"{boundary:8d} {phi:12.4f} {pred_phi:8.4f} {inv_sqrtn:10.4f}")

print("\nFormula: Fiedler sweep selects columns ell where cos(2*pi*ell/M) >= 0")
print("= columns 0..M/2-1 and 3M/2..M-1 (two half-strips, total k=n/2 vertices)")
print("Boundary = 2 column-gaps x L edges per gap = 2L edges")
print("phi = 2L / (n/2) = 4L/n = 4/M")
print("\nKey: phi = 4/M << 1 for large M (the long dimension)")
print("On expanders: phi = Omega(1) regardless of selector")
print("Ratio (expander/torus): Omega(M/4) = Omega(sqrt(n)) for square torus")

# verify against actual graphs
print("\n--- Verification against actual graph files ---")
for name,(L,M) in instances.items():
    try:
        G = load_gset(f"data/gset/{name}.txt")
        lam2 = nx.algebraic_connectivity(G, method="lanczos")
        lam2_exact = 4*np.sin(np.pi/max(L,M))**2
        # compute actual Fiedler vector boundary
        evecs = np.linalg.eigh(
            np.array(nx.laplacian_matrix(G).todense(),dtype=float))[1]
        fv = evecs[:,1]
        S = set(i for i in range(G.number_of_nodes()) if fv[i]>=0)
        bdy = sum(1 for u,v in G.edges() if (u in S)!=(v in S))
        phi_actual = bdy/len(S)
        print(f"{name}: lambda2={lam2:.6f} (exact={lam2_exact:.6f}), "
              f"|S|={len(S)}, |?S|={bdy}, phi={phi_actual:.4f} "
              f"(predicted {4/M:.4f})")
    except Exception as e:
        print(f"{name}: {e}")

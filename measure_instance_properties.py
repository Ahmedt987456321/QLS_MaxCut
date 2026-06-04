"""Measure structural properties of the cross-table instances to see which
co-vary with the selector winner (and which confounds need breaking)."""
import networkx as nx
import numpy as np
from src.graph import load_gset

# (instance, cross-table winner)
instances = [("G11","Fiedler"), ("G13","Fiedler"),
             ("G1","FConn"), ("G22","FConn")]

print(f"{'inst':5} {'winner':8} {'n':>5} {'edges':>7} {'avg_deg':>8} "
      f"{'bipartite':>10} {'lambda2':>10}")
for name, winner in instances:
    G = load_gset(f"data/gset/{name}.txt")
    n = G.number_of_nodes()
    e = G.number_of_edges()
    deg = 2*e/n
    bip = nx.is_bipartite(G)
    try:
        lam2 = nx.algebraic_connectivity(G, method="lanczos")
    except Exception:
        lam2 = float("nan")
    print(f"{name:5} {winner:8} {n:5d} {e:7d} {deg:8.1f} "
          f"{str(bip):>10} {lam2:10.4f}")

print("\nLook for: does the winner line up with ONE property cleanly,")
print("or do several properties co-vary (confounded)?")
print("  Fiedler-wins instances vs FConn-wins instances:")
print("  - bipartite?  (your routing finding)")
print("  - density/degree?  (sparsity confound)")
print("  - lambda2?  (the spectral quantity)")

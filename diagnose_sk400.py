"""Diagnose the n=400 SK failure: -inf cut values."""
import numpy as np
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

neal = get_backend("neal")

# test a single n=400 SK instance
rng = np.random.default_rng(0)
G = nx.complete_graph(400)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

k = 200
print(f"n=400 SK: edges={G.number_of_edges()}, k={k}")
print("Running single trial...")
try:
    result = adaptive_qls(G, budget_seconds=15,
        selector=select_frustrated_connected,
        backend=neal, k_min=k, k_max=k,
        n_reads=100, seed=0, best_known=None)
    print(f"Result: best_cut={result[1].best_cut}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

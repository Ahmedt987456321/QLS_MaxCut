import numpy as np, networkx as nx, time
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

neal = get_backend("neal")
rng = np.random.default_rng(0)
G = nx.complete_graph(400)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

print("Testing SK n=400 with warm-start cap fix, budget=30s...")
t0 = time.time()
result = adaptive_qls(G, budget_seconds=30,
    selector=select_frustrated_connected,
    backend=neal, k_min=100, k_max=100,
    n_reads=100, seed=0, best_known=None)
elapsed = time.time() - t0
state = result[1]
print(f"Elapsed: {elapsed:.1f}s")
print(f"best_cut: {state.best_cut}")
print(f"Fix worked: {state.best_cut != float('-inf')}")

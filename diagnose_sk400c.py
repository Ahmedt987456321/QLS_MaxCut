import numpy as np
import networkx as nx
from src.gain_cache import GainCache
from src.local_search import compute_cut_value, random_cut
from src.qubo import build_local_qubo
from src.backends import get_backend

rng = np.random.default_rng(0)
G = nx.complete_graph(400)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

print("Step 1: random_cut")
x = random_cut(G)
print(f"  OK: cut={compute_cut_value(G,x):.4f}")

print("Step 2: GainCache.update")
gc = GainCache()
gc.update(G, x)
print(f"  OK: valid={gc.valid}, n_gains={len(gc.gain)}")
gains = list(gc.gain.values())
print(f"  gain range: [{min(gains):.4f}, {max(gains):.4f}]")
print(f"  any inf: {any(np.isinf(g) for g in gains)}")

print("Step 3: build_local_qubo on k=200 support")
S = list(G.nodes())[:200]
try:
    Q = build_local_qubo(G, x, S)
    print(f"  OK: Q has {len(Q)} terms")
    if Q:
        vals = list(Q.values())
        print(f"  Q value range: [{min(vals):.4f}, {max(vals):.4f}]")
        print(f"  any inf: {any(np.isinf(v) for v in vals)}")
except Exception as e:
    print(f"  ERROR: {e}")

print("Step 4: neal on this QUBO")
neal = get_backend("neal")
try:
    result = neal(Q, S, n_reads=10)
    print(f"  OK: result type={type(result)}")
except Exception as e:
    print(f"  ERROR: {e}")

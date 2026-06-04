import numpy as np
import networkx as nx
from src.local_search import compute_cut_value, random_cut
from src.gain_cache import GainCache

rng = np.random.default_rng(0)

# small SK
G = nx.complete_graph(50)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

print("Test 1: compute_cut_value on small SK (n=50)")
x = random_cut(G)
cut = compute_cut_value(G, x)
print(f"  cut value: {cut}")
print(f"  type: {type(cut)}")

print("\nTest 2: weights finite?")
weights = [G[u][v]["weight"] for u,v in G.edges()]
print(f"  min={min(weights):.4f} max={max(weights):.4f}")
print(f"  any inf: {any(np.isinf(w) for w in weights)}")

print("\nTest 3: compute_cut_value on SK n=400")
rng2 = np.random.default_rng(0)
G2 = nx.complete_graph(400)
for u,v in G2.edges():
    G2[u][v]["weight"] = float(rng2.normal(0,1))
x2 = random_cut(G2)
cut2 = compute_cut_value(G2, x2)
print(f"  cut value n=400: {cut2}")
print(f"  type: {type(cut2)}")
print(f"  is -inf: {cut2 == float('-inf')}")

print("\nTest 4: GainCache on SK n=400")
try:
    gc = GainCache(G2, x2)
    gains = [gc.gain(v) for v in list(G2.nodes())[:5]]
    print(f"  first 5 gains: {gains}")
    print(f"  any inf gain: {any(np.isinf(g) for g in gains)}")
except Exception as e:
    print(f"  ERROR: {e}")

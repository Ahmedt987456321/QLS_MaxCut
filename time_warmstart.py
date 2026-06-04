import numpy as np, networkx as nx, time
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.local_search import compute_cut_value

rng = np.random.default_rng(0)
G = nx.complete_graph(400)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

print("Timing warm-start pool on SK n=400 (5 x one_flip_ls):")
t0 = time.time()
pool = []
for i in range(5):
    x_init = random_cut(G, rng)
    gc_temp = GainCache()
    x_opt, gc_temp, _, _ = one_flip_ls(G, x_init, gc_temp)
    pool.append(dict(x_opt))
    print(f"  iteration {i+1}: {time.time()-t0:.1f}s")
print(f"Total warm-start: {time.time()-t0:.1f}s")
print(f"Budget was: 15s")
print(f"Time left for main loop: {15-(time.time()-t0):.1f}s")

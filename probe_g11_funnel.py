import numpy as np
from collections import defaultdict
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls, compute_cut_value

G = load_gset("data/gset/G11.txt")
nodes = list(G.nodes())
rng = np.random.default_rng(0)

def canon(x):
    # fold global spin-flip: pick the orientation with node-1 on side 0
    if x[nodes[0]] == 1:
        x = {v: 1-x[v] for v in nodes}
    return tuple(x[v] for v in nodes)

R = 2000
best = -1
hits = defaultdict(int)        # canonical optimum -> times reached
val_of = {}
for _ in range(R):
    x = random_cut(G, rng)
    gc = GainCache()
    x, gc, _, _ = one_flip_ls(G, x, gc)
    c = compute_cut_value(G, x)
    k = canon(x)
    hits[k] += 1; val_of[k] = c
    best = max(best, c)

opt_keys = [k for k in hits if val_of[k] == best]
total_opt_hits = sum(hits[k] for k in opt_keys)
print(f"restarts={R}  best_cut={best}")
print(f"fraction of restarts reaching best: {total_opt_hits/R:.3f}")
print(f"distinct optimal configs (after spin-flip fold): {len(opt_keys)}")
print(f"distinct *suboptimal* optima found: {len(hits)-len(opt_keys)}")

# zero-cost free-flip count at one optimum: how big is the neutral plateau locally?
k0 = max(opt_keys, key=lambda k: hits[k])
x0 = {v: k0[i] for i,v in enumerate(nodes)}
gc = GainCache(); gc.update(G, x0)
free = sum(1 for v in nodes if abs(gc.gain[v]) < 1e-9)
print(f"free (zero-gain) single flips at a top optimum: {free}  (neutrality signal)")

import numpy as np
from collections import defaultdict
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls, compute_cut_value

for inst, bks in [("G11", 564), ("G14", 3064)]:
    G = load_gset(f"data/gset/{inst}.txt")
    nodes = list(G.nodes()); rng = np.random.default_rng(0)
    def canon(x):
        if x[nodes[0]] == 1: x = {v: 1-x[v] for v in nodes}
        return tuple(x[v] for v in nodes)
    R = 2000; hits = defaultdict(int); val = {}; best = -1
    for _ in range(R):
        x = random_cut(G, rng); gc = GainCache()
        x, gc, _, _ = one_flip_ls(G, x, gc)
        c = compute_cut_value(G, x); k = canon(x)
        hits[k] += 1; val[k] = c; best = max(best, c)
    opt = [k for k in hits if val[k] == best]
    print(f"{inst}: best_descent={best:.0f}  (BKS {bks}, gap {bks-best:.0f})  "
          f"reach_best={sum(hits[k] for k in opt)/R:.3f}  "
          f"distinct_optima={len(hits)}")

import numpy as np
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected

print("Connected-subproblem treewidth (FConn selector)")
print("expect: G11/G13 low (feasible), G14 intermediate, G1/G22 high (infeasible)")
print()
print(f"{'inst':<6}{'k':<6}{'treewidth':<12}{'verdict'}")
print("-" * 40)

for name in ["G11", "G13", "G14", "G1", "G22"]:
    G = load_gset(f"data/gset/{name}.txt")
    rng = np.random.default_rng(0)
    x = random_cut(G, rng)
    gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc)
    for k in [160, 400, 640]:
        if k >= G.number_of_nodes():
            continue
        gc.update(G, x)
        S = select_frustrated_connected(G, gc, k, rng=rng, x=x)
        H = G.subgraph(S)
        tw, _ = treewidth_min_fill_in(H)
        verdict = "feasible" if tw <= 28 else "INFEASIBLE"
        print(f"{name:<6}{k:<6}{tw:<12}{verdict}")
    print()

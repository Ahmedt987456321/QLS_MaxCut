"""Validate QUBO -> (J,h) -> GTN against brute-force exact, on real G11 sub-QUBOs."""
import numpy as np, json, subprocess, os
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected
from src.qubo import build_local_qubo, qubo_energy
from src.backends import backend_exact

def qubo_to_ising(Q, S):
    """QUBO (minimise) -> Ising J,h for GTN (maximise). Returns edges, J, h, idx."""
    S = list(S)
    idx = {v: i for i, v in enumerate(S)}
    n = len(S)
    h = np.zeros(n)
    Jdict = {}
    for (a, b), w in Q.items():
        if a == b:
            h[idx[a]] += w / 2.0          # linear: Q_ii x_i,  x=(1+s)/2
        else:
            i, j = idx[a], idx[b]
            key = (min(i, j), max(i, j))
            Jdict[key] = Jdict.get(key, 0.0) + w / 4.0
            h[idx[a]] += w / 4.0
            h[idx[b]] += w / 4.0
    # negate for min->max
    edges = [[i + 1, j + 1] for (i, j) in Jdict]   # 1-indexed for Julia
    J = [-Jdict[(i, j)] for (i, j) in Jdict]
    h = (-h).tolist()
    return edges, J, h, idx, S

def solve_gtn(edges, J, h, n):
    payload = {"n": n, "edges": edges, "J": J, "h": h}
    with open("tn_in.json", "w") as f:
        json.dump(payload, f)
    subprocess.run(["julia", "solve_tn_worker.jl"], check=True)
    with open("tn_out.json") as f:
        return json.load(f)

# build a real G11 sub-QUBO
G = load_gset("data/gset/G11.txt")
rng = np.random.default_rng(0)
x = random_cut(G, rng)
gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc)
gc.update(G, x)
S = select_frustrated_connected(G, gc, 14, rng=rng, x=x)   # small: exact-checkable

Q = build_local_qubo(G, x, S)

# exact brute force (your oracle)
sol_exact = backend_exact(Q, S)
e_exact = qubo_energy(Q, sol_exact, S)

# GTN path
edges, J, h, idx, Slist = qubo_to_ising(Q, S)
out = solve_gtn(edges, J, h, len(Slist))
spins = out["config"]   # list of 0/1 (GTN spin readout)
# GTN config -> QUBO assignment: spin 1 -> x=1, spin 0 -> x=0 (s=2x-1)
sol_gtn = {Slist[i]: int(spins[i]) for i in range(len(Slist))}
e_gtn = qubo_energy(Q, sol_gtn, S)

print(f"exact QUBO energy: {e_exact:.1f}")
print(f"GTN   QUBO energy: {e_gtn:.1f}")
print("MATCH" if abs(e_exact - e_gtn) < 1e-6 else "MISMATCH - mapping wrong")

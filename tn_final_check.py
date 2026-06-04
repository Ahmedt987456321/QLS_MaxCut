import numpy as np
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected
from src.qubo import build_local_qubo, qubo_energy
from src.backends import get_backend, backend_exact

tn = get_backend("tn")
G = load_gset("data/gset/G11.txt")
print("Validating src.backends.backend_tn vs exact (G11 k=14):")
ok = True
for seed in range(5):
    rng = np.random.default_rng(seed)
    x = random_cut(G, rng)
    gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc); gc.update(G, x)
    S = select_frustrated_connected(G, gc, 14, rng=rng, x=x)
    Q = build_local_qubo(G, x, S)
    e_tn = qubo_energy(Q, tn(Q, S), S)
    e_ex = qubo_energy(Q, backend_exact(Q, S), S)
    m = abs(e_tn - e_ex) < 1e-6
    ok = ok and m
    print(f"  seed {seed}: TN={e_tn:.1f} exact={e_ex:.1f} {'OK' if m else 'FAIL'}")
print("ALL MATCH" if ok else "FAILURES")

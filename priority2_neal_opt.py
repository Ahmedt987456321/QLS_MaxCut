"""Priority 2: is neal solving the extracted k=400 sub-QUBOs optimally?
Compare neal vs backend_tn (exact, treewidth-gated) on real extracted sub-QUBOs."""
import numpy as np
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected
from src.qubo import build_local_qubo, qubo_energy
from src.backends import get_backend

G = load_gset("data/gset/G11.txt")
neal = get_backend("neal")
tn = get_backend("tn")   # exact via GTN when treewidth allows

rng = np.random.default_rng(0)
x = random_cut(G, rng); gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc); gc.update(G, x)

print("neal vs exact-TN on real G11 k=400 sub-QUBOs:")
gaps = []
for i in range(15):
    gc.update(G, x)
    S = select_frustrated_connected(G, gc, 400, rng=rng, x=x)
    Q = build_local_qubo(G, x, S)
    if not Q or all(abs(v) < 1e-10 for v in Q.values()):
        continue
    e_neal = qubo_energy(Q, neal(Q, S, n_reads=100), S)
    e_tn = qubo_energy(Q, tn(Q, S), S)   # exact if tw<=28, else neal-fallback
    gap = e_neal - e_tn   # >0 means neal worse (higher energy = worse)
    gaps.append(gap)
    flag = "neal SUBOPTIMAL" if gap > 1e-6 else "neal optimal"
    print(f"  sub {i}: neal={e_neal:.1f} tn={e_tn:.1f} gap={gap:.1f}  {flag}")

if gaps:
    n_sub = sum(1 for g in gaps if g > 1e-6)
    print(f"\nneal suboptimal on {n_sub}/{len(gaps)} sub-QUBOs; "
          f"mean gap={np.mean(gaps):.2f}, max gap={max(gaps):.1f}")

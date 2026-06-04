"""Fair TN vs neal: warm up Julia worker before timing; 20 trials."""
import numpy as np, statistics as stats, json
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend, backend_tn

G = load_gset("data/gset/G11.txt")
K, BUDGET, TRIALS = 400, 30, 20

# WARM UP the Julia worker so cold-start isn't charged to trial 1
print("warming up TN worker...")
from src.qubo import build_local_qubo
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
rng = np.random.default_rng(99)
x = random_cut(G, rng); gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc); gc.update(G, x)
S = select_frustrated_connected(G, gc, 14, rng=rng, x=x)
backend_tn(build_local_qubo(G, x, S), S)  # triggers Julia load now
print("warm.\n")

def run(name):
    be = get_backend(name); cuts = []
    for t in range(TRIALS):
        _, m = adaptive_qls(G, budget_seconds=BUDGET,
                            selector=select_frustrated_connected,
                            backend=be, k_min=K, k_max=K, n_reads=100, seed=t)
        cuts.append(m.best_cut)
        print(f"  {name} {t+1}/{TRIALS}: {m.best_cut}")
    return cuts

print("=== neal ==="); neal = run("neal")
print("=== tn ===");   tn = run("tn")
print(f"\nneal: median={stats.median(neal):.1f} max={max(neal):.1f}")
print(f"tn:   median={stats.median(tn):.1f} max={max(tn):.1f}")
json.dump({"neal": neal, "tn": tn}, open("results/tn_vs_neal_G11_warm.json","w"))

"""Live-loop benchmark: AQLS with backend_tn vs backend_neal on G11."""
import numpy as np, time, statistics as stats, json
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

G = load_gset("data/gset/G11.txt")
K = 400
BUDGET = 30
TRIALS = 10

def run(backend_name):
    cuts = []
    be = get_backend(backend_name)
    for t in range(TRIALS):
        x, m = adaptive_qls(G, budget_seconds=BUDGET,
                            selector=select_frustrated_connected,
                            backend=be, k_min=K, k_max=K,
                            n_reads=100, seed=t)
        cuts.append(m.best_cut)
        print(f"  {backend_name} trial {t+1}/{TRIALS}: cut={m.best_cut}")
    return cuts

print(f"G11 k={K} budget={BUDGET}s trials={TRIALS}\n")
print("=== neal ===")
neal = run("neal")
print("\n=== tn ===")
tn = run("tn")

print(f"\nneal: median={stats.median(neal):.1f} max={max(neal):.1f}")
print(f"tn:   median={stats.median(tn):.1f} max={max(tn):.1f}")
json.dump({"neal": neal, "tn": tn, "k": K, "budget": BUDGET},
          open("results/tn_vs_neal_G11_live.json", "w"))

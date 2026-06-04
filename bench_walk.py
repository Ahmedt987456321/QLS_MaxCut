"""A/B: greedy 'improvement' vs BLS-style 'walk' acceptance on G11."""
import numpy as np, statistics as stats, json
from scipy.stats import mannwhitneyu
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

G = load_gset("data/gset/G11.txt")
BUDGET, TRIALS, K = 30, 20, 400
neal = get_backend("neal")

def run(mode):
    cuts = []
    for t in range(TRIALS):
        x, m = adaptive_qls(G, budget_seconds=BUDGET,
                            selector=select_frustrated_connected,
                            backend=neal, k_min=K, k_max=K, n_reads=100,
                            seed=t, acceptance=mode, best_known=564)
        cuts.append(m.best_cut)
        print(f"  {mode} {t+1}/{TRIALS}: {m.best_cut}")
    return cuts

print("=== improvement (greedy, locked) ===")
imp = run("improvement")
print("=== walk (BLS-style accept-and-walk) ===")
walk = run("walk")

U, p = mannwhitneyu(imp, walk, alternative="two-sided")
print(f"\nimprovement: median={stats.median(imp):.1f} max={max(imp):.1f}")
print(f"walk:        median={stats.median(walk):.1f} max={max(walk):.1f}")
print(f"Mann-Whitney p = {p:.4g}  (BKS=564)")
json.dump({"improvement": imp, "walk": walk, "p": p},
          open("results/walk_vs_greedy_G11.json","w"))

"""A/B: plain AQLS vs AQLS + cluster move using MOST-DIFFERENT pool member."""
import numpy as np, statistics as stats, json
from scipy.stats import mannwhitneyu
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

G = load_gset("data/gset/G11.txt")
BUDGET, TRIALS, K = 30, 20, 400
neal = get_backend("neal")

def run(cluster):
    cuts = []
    for t in range(TRIALS):
        x, m = adaptive_qls(G, budget_seconds=BUDGET,
                            selector=select_frustrated_connected, backend=neal,
                            k_min=K, k_max=K, n_reads=100, seed=t,
                            acceptance='improvement', best_known=564,
                            cluster_moves=cluster, cluster_interval=20)
        cuts.append(m.best_cut)
        print(f"  {'cluster' if cluster else 'plain'} {t+1}/{TRIALS}: {m.best_cut}")
    return cuts

print("=== plain ===")
plain = run(False)
print("=== cluster (diverse member) ===")
clust = run(True)
U, p = mannwhitneyu(plain, clust, alternative="two-sided")
print(f"\nplain:   median={stats.median(plain):.1f} max={max(plain):.1f}")
print(f"cluster: median={stats.median(clust):.1f} max={max(clust):.1f}")
print(f"Mann-Whitney p = {p:.4g}  (BKS=564)")
json.dump({"plain": plain, "cluster_diverse": clust, "p": p},
          open("results/cluster_diverse_G11.json","w"))

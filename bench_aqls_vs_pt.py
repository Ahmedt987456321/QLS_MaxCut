"""Fair head-to-head: AQLS-FConn-LA vs Parallel Tempering on G11."""
import numpy as np, statistics as stats, json
from scipy.stats import mannwhitneyu
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend
from src.baselines import parallel_tempering

G = load_gset("data/gset/G11.txt")
BUDGET, TRIALS, K = 30, 30, 400

aqls, pt = [], []
neal = get_backend("neal")
for t in range(TRIALS):
    _, m = adaptive_qls(G, budget_seconds=BUDGET,
                        selector=select_frustrated_connected,
                        backend=neal, k_min=K, k_max=K, n_reads=100, seed=t)
    aqls.append(m.best_cut)
    _, mp = parallel_tempering(G, budget_seconds=BUDGET, best_known=564, seed=t)
    pt.append(mp.best_cut)
    print(f"  trial {t+1}/{TRIALS}: AQLS={m.best_cut} PT={mp.best_cut}")

U, p = mannwhitneyu(aqls, pt, alternative="two-sided")
print(f"\nAQLS: median={stats.median(aqls):.1f} max={max(aqls):.1f}")
print(f"PT:   median={stats.median(pt):.1f} max={max(pt):.1f}")
print(f"Mann-Whitney p = {p:.4g}")
json.dump({"aqls": aqls, "pt": pt, "p": p}, open("results/aqls_vs_pt_G11.json","w"))

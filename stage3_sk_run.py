"""Stage 3: SK complete graphs (Sherrington-Kirkpatrick model).
Complete K_n with Gaussian weights. lambda2 = n-1 >> tau=0.2.
Prediction: FConn wins (far above safe band).
Tests: does FConn still win in the far-above-band regime?"""
import numpy as np
import networkx as nx
import statistics
import json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS = 15; BUDGET = 25

print(f"{'n':>6} {'lambda2':>10} {'FConn_med':>10} {'Fiedler_med':>10} "
      f"{'winner':>10} {'p-value':>10} {'correct?':>10}")
print("-"*80)

summary = []
for n in [100, 200, 400]:
    k = min(n//2, 200)
    fc_cuts = []; fi_cuts = []; lam2s = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        G = nx.complete_graph(n)
        for u,v in G.edges():
            G[u][v]["weight"] = float(rng.normal(0, 1))
        lam2 = nx.algebraic_connectivity(G, method="lanczos")
        lam2s.append(lam2)
        for trial in range(TRIALS):
            s = seed*1000+trial
            r1 = adaptive_qls(G, budget_seconds=BUDGET,
                selector=select_frustrated_connected,
                backend=neal, k_min=k, k_max=k,
                n_reads=100, seed=s, best_known=None)
            r2 = adaptive_qls(G, budget_seconds=BUDGET,
                selector=select_fiedler,
                backend=neal, k_min=k, k_max=k,
                n_reads=100, seed=s, best_known=None)
            fc_cuts.append(r1[1].best_cut)
            fi_cuts.append(r2[1].best_cut)
    if fc_cuts:
        mean_lam2 = np.mean(lam2s)
        fc_med = statistics.median(fc_cuts)
        fi_med = statistics.median(fi_cuts)
        winner = "Fiedler" if fi_med>fc_med else \
                 ("FConn" if fc_med>fi_med else "tie")
        stat,pval = mannwhitneyu(fc_cuts,fi_cuts,alternative="two-sided")
        correct = winner=="FConn"
        print(f"{n:6d} {mean_lam2:10.2f} {fc_med:10.1f} {fi_med:10.1f} "
              f"{winner:>10} {pval:10.4f} {'YES' if correct else 'NO':>10}")
        summary.append({"n":n,"lam2":mean_lam2,"fc_med":fc_med,
            "fi_med":fi_med,"winner":winner,"pval":pval,"correct":correct})

json.dump(summary, open("results/stage3_sk_results.json","w"), indent=2)
print(f"\nRouting correct: {sum(s['correct'] for s in summary)}/{len(summary)}")
print("Prediction: FConn wins on all SK instances (lambda2 >> tau=0.2)")
print("If correct: routing rule extrapolates to far-above-band regime")

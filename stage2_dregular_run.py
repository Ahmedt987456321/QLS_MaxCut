"""Stage 2: Run AQLS-SSR on random d-regular instances.
Routing predictions already confirmed (lambda2 matches prediction perfectly).
Now test whether the ROUTED selector actually WINS on these instances.
d=3 -> Fiedler should win; d=4,5 -> FConn should win."""
import numpy as np
import networkx as nx
import statistics
import json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS = 15   # 15 trials per instance; 10 instances per cell
BUDGET = 25
K_VALUES = {3: 400, 4: 400, 5: 320}  # k scales with 1/lambda2

print(f"{'d':>3} {'n':>6} {'FConn_med':>10} {'Fiedler_med':>10} "
      f"{'winner':>10} {'p-value':>10} {'routing':>10} {'correct?':>10}")
print("-"*80)

summary = []
for d in [3, 4, 5]:
    for n in [800, 1200]:   # two n values per d
        k = K_VALUES[d]
        fc_cuts = []; fi_cuts = []
        for seed in range(10):  # 10 instances
            try:
                G = nx.random_regular_graph(d, n, seed=seed)
                rng = np.random.default_rng(seed)
                for u,v in G.edges():
                    G[u][v]["weight"] = int(rng.choice([-1,1]))
                lam2 = nx.algebraic_connectivity(G, method="lanczos")
                routing = "Fiedler" if lam2 < 0.2 else "FConn"
                for trial in range(TRIALS):
                    s = seed * 1000 + trial
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
            except Exception as e:
                print(f"d={d} n={n} seed={seed}: {e}")
        if fc_cuts and fi_cuts:
            fc_med = statistics.median(fc_cuts)
            fi_med = statistics.median(fi_cuts)
            winner = "Fiedler" if fi_med > fc_med else \
                     ("FConn" if fc_med > fi_med else "tie")
            stat, pval = mannwhitneyu(fc_cuts, fi_cuts, alternative="two-sided")
            routing = "Fiedler" if d==3 else "FConn"
            correct = winner == routing
            print(f"{d:3d} {n:6d} {fc_med:10.1f} {fi_med:10.1f} "
                  f"{winner:>10} {pval:10.4f} {routing:>10} "
                  f"{'YES' if correct else 'NO':>10}")
            summary.append({"d":d,"n":n,"fc_med":fc_med,"fi_med":fi_med,
                "winner":winner,"pval":pval,"routing":routing,"correct":correct})

json.dump(summary, open("results/stage2_dregular_results.json","w"), indent=2)
print(f"\nRouting correct: {sum(s['correct'] for s in summary)}/{len(summary)}")
print("If all correct: routing rule generalises to random d-regular graphs.")

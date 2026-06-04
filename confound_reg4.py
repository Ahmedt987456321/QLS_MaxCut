"""Confound-breaker: random 4-regular graph. Degree=4 (matches G11/G13, the
sparse Fiedler-winners) but HIGH lambda2 (random -> good expansion).
Does the selector winner follow degree (->Fiedler) or lambda2 (->FConn)?"""
import numpy as np, statistics as stats, json
import networkx as nx
from scipy.stats import mannwhitneyu
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from src.local_search import compute_cut_value

neal = get_backend("neal")
selectors = {"FConn": select_frustrated_connected, "Fiedler": select_fiedler}

# build a few random 4-regular graphs (n=800, same degree as G11/G13)
print("Building random 4-regular graphs (n=800, degree 4, expect high lambda2)...")
results = {}
for gi, gseed in enumerate([101, 202, 303]):
    G = nx.random_regular_graph(4, 800, seed=gseed)
    lam2 = nx.algebraic_connectivity(G, method="lanczos")
    bip = nx.is_bipartite(G)
    print(f"\nreg4 #{gi} (seed {gseed}): degree=4, lambda2={lam2:.4f}, bipartite={bip}")
    cuts_by_sel = {}
    for sname, sel in selectors.items():
        cuts = []
        for s in range(8):
            x, m = adaptive_qls(G, budget_seconds=30, selector=sel, backend=neal,
                                k_min=400, k_max=400, n_reads=100, seed=s, best_known=None)
            cuts.append(m.best_cut)
        cuts_by_sel[sname] = cuts
        print(f"  {sname}: median={stats.median(cuts):.1f} max={max(cuts):.1f}")
    U, p = mannwhitneyu(cuts_by_sel["FConn"], cuts_by_sel["Fiedler"], alternative="two-sided")
    win = "FConn" if stats.median(cuts_by_sel["FConn"]) > stats.median(cuts_by_sel["Fiedler"]) \
          else ("Fiedler" if stats.median(cuts_by_sel["Fiedler"]) > stats.median(cuts_by_sel["FConn"]) else "tie")
    print(f"  --> winner={win}, p={p:.4g}")
    results[f"reg4_seed{gseed}"] = {
        "lambda2": lam2,
        "FConn_med": stats.median(cuts_by_sel["FConn"]),
        "Fiedler_med": stats.median(cuts_by_sel["Fiedler"]),
        "winner": win, "p": p}

print("\n=== VERDICT ===")
print("These graphs are degree-4 (like G11/G13 where Fiedler won) but HIGH lambda2.")
print("If FConn wins here -> lambda2 drives the split (NOT degree). Spectral story holds.")
print("If Fiedler wins here -> degree/density drives it (NOT lambda2). Spectral story weaker.")
json.dump(results, open("results/confound_reg4.json","w"), indent=2, default=str)

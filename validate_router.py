"""Validate the lambda2-routed selector: does it (a) route correctly and
(b) match the best hand-picked selector on each instance?"""
import numpy as np, statistics as stats, json
import networkx as nx
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import (select_frustrated_connected, select_fiedler,
                           select_lambda2_routed)
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8
THRESH = 0.2

# instances + the selector that WON the cross-table/confound tests
known = [
    ("G11", 400, 30, "Fiedler", None),
    ("G13", 400, 30, "Fiedler", None),
    ("G1",  160, 30, "FConn",   None),
    ("G22", 640, 30, "FConn",   None),
]
# plus a fresh reg-4 (high lambda2, should route to FConn) as a held-out check
reg4 = nx.random_regular_graph(4, 800, seed=999)

def med(G, sel, k, budget):
    cuts = [adaptive_qls(G, budget_seconds=budget, selector=sel, backend=neal,
            k_min=k, k_max=k, n_reads=100, seed=s, best_known=None)[1].best_cut
            for s in range(SEEDS)]
    return stats.median(cuts)

print(f"threshold = {THRESH}\n")
print(f"{'inst':6} {'lambda2':>9} {'routes_to':>10} {'expected':>9} "
      f"{'router_med':>11} {'best_hand':>10} {'match?':>7}")

rows = []
for name, k, budget, winner, _ in known:
    G = load_gset(f"data/gset/{name}.txt")
    lam2 = nx.algebraic_connectivity(G, method="lanczos")
    routes_to = "Fiedler" if lam2 < THRESH else "FConn"
    r_med = med(G, select_lambda2_routed, k, budget)
    # best hand-picked = whichever selector the tests said won
    hand_sel = select_fiedler if winner == "Fiedler" else select_frustrated_connected
    h_med = med(G, hand_sel, k, budget)
    match = "YES" if routes_to == winner else "NO"
    print(f"{name:6} {lam2:9.4f} {routes_to:>10} {winner:>9} "
          f"{r_med:11.0f} {h_med:10.0f} {match:>7}")
    rows.append({"inst":name,"lambda2":lam2,"routes_to":routes_to,
                 "expected":winner,"router_med":r_med,"best_hand":h_med,
                 "match":match})

# held-out reg-4
lam2 = nx.algebraic_connectivity(reg4, method="lanczos")
routes_to = "Fiedler" if lam2 < THRESH else "FConn"
r_med = med(reg4, select_lambda2_routed, 400, 30)
fc_med = med(reg4, select_frustrated_connected, 400, 30)
fi_med = med(reg4, select_fiedler, 400, 30)
best = "FConn" if fc_med >= fi_med else "Fiedler"
print(f"{'reg4*':6} {lam2:9.4f} {routes_to:>10} {best:>9} "
      f"{r_med:11.0f} {max(fc_med,fi_med):10.0f} "
      f"{'YES' if routes_to==best else 'NO':>7}  (held-out)")

print("\nrouter matches best hand-pick when routes_to == expected/best on every row.")
json.dump(rows, open("results/router_validation.json","w"), indent=2, default=str)

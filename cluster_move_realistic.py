"""Honest test: does cluster move help between TWO SUB-OPTIMAL configs
(no oracle/optimum)? This is the realistic in-loop scenario."""
import numpy as np
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend
from cluster_move_dev import cluster_move, overlap_components

G = load_gset("data/gset/G11.txt")
neal = get_backend("neal")

print("Cluster move between two SUB-OPTIMAL FConn configs (no oracle):")
for seed_pair in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
    sA, sB = seed_pair
    xA,_ = adaptive_qls(G, budget_seconds=15, selector=select_frustrated_connected,
                        backend=neal, k_min=400,k_max=400,n_reads=100,seed=sA,
                        acceptance='improvement',best_known=564)
    xB,_ = adaptive_qls(G, budget_seconds=15, selector=select_frustrated_connected,
                        backend=neal, k_min=400,k_max=400,n_reads=100,seed=sB,
                        acceptance='improvement',best_known=564)
    cA, cB = compute_cut_value(G,xA), compute_cut_value(G,xB)
    comps,_ = overlap_components(G, xA, xB)
    # apply to whichever is worse, using the other as reference
    if cA <= cB:
        xNew, gain = cluster_move(G, xA, xB, np.random.default_rng(0))
        start = cA
    else:
        xNew, gain = cluster_move(G, xB, xA, np.random.default_rng(0))
        start = cB
    print(f"  seeds {sA},{sB}: A={cA} B={cB}, {len(comps)} comps, "
          f"move {start}->{start+gain:.0f} (gain {gain:+.0f})")

import numpy as np
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

G = load_gset("data/gset/G11.txt")
neal = get_backend("neal")
for cluster in (False, True):
    calls, cuts = [], []
    for t in range(5):
        x, m = adaptive_qls(G, budget_seconds=30, selector=select_frustrated_connected,
                            backend=neal, k_min=400, k_max=400, n_reads=100, seed=t,
                            acceptance='improvement', best_known=564,
                            cluster_moves=cluster, cluster_interval=20)
        calls.append(m.qls_calls); cuts.append(m.best_cut)
    tag = "cluster" if cluster else "plain  "
    print(f"{tag}: mean QLS calls={np.mean(calls):.0f}  mean cut={np.mean(cuts):.1f}")

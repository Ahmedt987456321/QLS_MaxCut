"""Priority 1: does AQLS's 550 incumbent differ from a 564 optimum by a
WINDING domain wall? Get 564 via Fiedler selector, overlay on AQLS-550 best,
map disagreement onto the torus, check if it wraps a periodic direction."""
import numpy as np
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from src.local_search import compute_cut_value

G = load_gset("data/gset/G11.txt")
neal = get_backend("neal")

# 1. get AQLS best (greedy) -- expect ~550
xA, mA = adaptive_qls(G, budget_seconds=30,
                      selector=select_frustrated_connected,
                      backend=neal, k_min=400, k_max=400, n_reads=100,
                      seed=0, acceptance='improvement', best_known=564)
cutA = compute_cut_value(G, xA)
print(f"AQLS-FConn best cut: {cutA}")

# 2. get a 564 optimum via Fiedler selector
xF, mF = adaptive_qls(G, budget_seconds=30,
                      selector=select_fiedler,
                      backend=neal, k_min=400, k_max=400, n_reads=100,
                      seed=0, acceptance='improvement', best_known=564)
cutF = compute_cut_value(G, xF)
print(f"Fiedler best cut: {cutF}")

if cutF < 564:
    print(f"(Fiedler got {cutF}, not 564 -- still usable as a better-config comparison)")

# 3. disagreement set N (vertices where partitions differ, up to global flip)
nodes = list(G.nodes())
agree = sum(1 for v in nodes if xA[v] == xF[v])
# account for spin-reversal symmetry: if <50% agree, flip one
if agree < len(nodes) / 2:
    xF = {v: 1 - xF[v] for v in nodes}
N = [v for v in nodes if xA[v] != xF[v]]
print(f"\nDisagreement set N: {len(N)} of {len(nodes)} vertices "
      f"({100*len(N)/len(nodes):.1f}%)")

# 4. is N connected? does it span (wrap) the torus?
import networkx as nx
HN = G.subgraph(N)
comps = list(nx.connected_components(HN))
print(f"N induces {len(comps)} connected component(s); "
      f"largest = {max(len(c) for c in comps) if comps else 0}")
print("\nInterpretation:")
print("  - few small blobs  -> local diffs, NOT topological (gap fixable locally)")
print("  - one large spanning band -> winding domain wall (topological, "
      "needs global move)")

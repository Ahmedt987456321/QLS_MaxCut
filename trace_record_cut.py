import numpy as np
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend
from src.metrics import Metrics
import unittest.mock as mock

neal = get_backend("neal")
rng = np.random.default_rng(0)
G = nx.complete_graph(100)  # smaller to run fast
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

# patch record_cut to trace calls
call_log = []
orig_record = Metrics.record_cut
def traced_record(self, cut):
    call_log.append(cut)
    orig_record(self, cut)

Metrics.record_cut = traced_record

result = adaptive_qls(G, budget_seconds=5,
    selector=select_frustrated_connected,
    backend=neal, k_min=50, k_max=50,
    n_reads=10, seed=0, best_known=None)

Metrics.record_cut = orig_record  # restore

state = result[1]
print(f"record_cut called {len(call_log)} times")
if call_log:
    print(f"first 5 values: {call_log[:5]}")
    print(f"max value: {max(call_log)}")
print(f"metrics.best_cut = {state.best_cut}")
print(f"metrics.cut_trace length = {len(state.cut_trace)}")

import numpy as np
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

neal = get_backend("neal")

rng = np.random.default_rng(0)
G = nx.complete_graph(400)
for u,v in G.edges():
    G[u][v]["weight"] = float(rng.normal(0,1))

print("Running adaptive_qls with budget=15s, k=200 on SK n=400...")
print("Watching for -inf...")

import src.adaptive_qls as aqls_module
import src.metrics as metrics_module

# patch to trace best_cut updates
orig = aqls_module.adaptive_qls
def traced(*args, **kwargs):
    result = orig(*args, **kwargs)
    state = result[1]
    print(f"  returned best_cut={state.best_cut}")
    print(f"  type: {type(state.best_cut)}")
    return result

result = adaptive_qls(G, budget_seconds=5,
    selector=select_frustrated_connected,
    backend=neal, k_min=200, k_max=200,
    n_reads=10, seed=0, best_known=None)

state = result[1]
print(f"best_cut={state.best_cut}")
print(f"is -inf: {state.best_cut == float('-inf')}")

# check what Metrics initialises best_cut to
import inspect
from src.metrics import Metrics
print("\nMetrics.__init__:")
print(inspect.getsource(Metrics.__init__))

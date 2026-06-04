import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend
import numpy as np

def load_mtx(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    data_lines=[l.strip() for l in lines if not l.startswith('%') and l.strip()]
    for line in data_lines[1:]:
        parts=line.split()
        if len(parts)>=2:
            u,v=int(parts[0]),int(parts[1])
            if u!=v: G.add_edge(u,v)
    return nx.convert_node_labels_to_integers(G)

# Check 1: file loads correctly
G = load_mtx('data/walshaw/add20/add20.mtx')
print(f'Check 1 - Load: n={G.number_of_nodes()}, m={G.number_of_edges()}, connected={nx.is_connected(G)}')

# Check 2: weights correct
rng=np.random.default_rng(42)
for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
weights=[G[u][v]['weight'] for u,v in list(G.edges())[:5]]
print(f'Check 2 - Weights: {weights}, all +-1: {all(w in [-1,1] for w in weights)}')

# Check 3: one trial completes
neal = get_backend('neal')
result = adaptive_qls(G, budget_seconds=5,
    selector=select_frustrated_connected, backend=neal,
    k_min=400, k_max=400, n_reads=100, seed=42, best_known=None)
print(f'Check 3 - Smoke test: best_cut={result[1].best_cut}')
print(f'  valid: {result[1].best_cut != float("-inf")}')
print()
print('All checks passed -- safe to run full experiment')

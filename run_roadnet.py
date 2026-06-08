import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend('neal')
TRIALS=10; BUDGET=25; k=400

# Load saved subgraph
G = nx.read_edgelist('data/roadnetwork/roadnet_pa_sub2000.edgelist',
                     nodetype=int, data=[('weight', int)])
G = nx.convert_node_labels_to_integers(G)
n=G.number_of_nodes()

# Verify load
print(f'Loaded: n={n}, m={G.number_of_edges()}, connected={nx.is_connected(G)}')

# Smoke test first
result = adaptive_qls(G, budget_seconds=5,
    selector=select_frustrated_connected, backend=neal,
    k_min=k, k_max=k, n_reads=100, seed=42, best_known=None)
print(f'Smoke test: best_cut={result[1].best_cut}, valid={result[1].best_cut != float("-inf")}')
print()

# Full experiment
print(f'roadNet-PA subgraph: n={n}, beta_e=0.0088, route=Fiedler')
print()

fc=[]; fi=[]
for seed in range(TRIALS):
    s=seed*1000+42
    fc.append(adaptive_qls(G,budget_seconds=BUDGET,
        selector=select_frustrated_connected,backend=neal,
        k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fi.append(adaptive_qls(G,budget_seconds=BUDGET,
        selector=select_fiedler,backend=neal,
        k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    print(f'  trial {seed}: FConn={fc[-1]:.1f} Fiedler={fi[-1]:.1f}')

fcm=statistics.median(fc); fim=statistics.median(fi)
w='Fiedler' if fim>fcm else ('FConn' if fcm>fim else 'tie')
try: _,p=mannwhitneyu(fc,fi,alternative='two-sided')
except: p=1.0

print()
print(f'FConn median: {fcm:.1f}')
print(f'Fiedler median: {fim:.1f}')
print(f'Winner: {w}')
print(f'p-value: {p:.4f}')
print(f'Prediction: Fiedler')
print(f'Correct: {w=="Fiedler"}')

json.dump({'name':'roadnet_pa_sub2000','n':n,'beta_e':0.0088,'asym':0.196,
           'lam2':0.0023,'fc':fcm,'fi':fim,'winner':w,'p':p,
           'pred':'Fiedler','correct':w=='Fiedler'},
    open('results/roadnet_pa_result.json','w'),indent=2)

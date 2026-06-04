import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

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

neal = get_backend('neal')
TRIALS=10; BUDGET=25; k=400

G = load_mtx('data/walshaw/3elt/3elt.mtx')
rng=np.random.default_rng(42)
for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
n=G.number_of_nodes()

print(f'Walshaw 3elt FEM mesh: n={n}, beta_e=0.0085, route=Fiedler')
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

json.dump({'name':'3elt','n':n,'beta_e':0.0085,'asym':0.241,
           'fc':fcm,'fi':fim,'winner':w,'p':p,'pred':'Fiedler',
           'correct':w=='Fiedler'},
    open('results/walshaw_3elt_result.json','w'),indent=2)

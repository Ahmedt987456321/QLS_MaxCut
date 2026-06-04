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

def beta_e_asym(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    v=nx.fiedler_vector(G_unw,method='lanczos')
    lam2=nx.algebraic_connectivity(G_unw,method='lanczos')
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    asym=abs(abs(v.max())-abs(v.min()))/max(abs(v.max()),abs(v.min()))
    return cut/m, lam2, asym

# First measure beta_e
G = load_mtx('data/walshaw/add20/add20.mtx')
rng=np.random.default_rng(42)
for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
n=G.number_of_nodes(); m=G.number_of_edges()
b, lam2, asym = beta_e_asym(G)
route='Fiedler' if (b<0.05 and asym<0.5) else 'FConn'

print(f'Walshaw add20: n={n}, m={m}, d_avg={2*m/n:.1f}')
print(f'  lambda2={lam2:.4f}, beta_e={b:.4f}, asym={asym:.4f}')
print(f'  routing: {route}')
print()

# Run AQLS
neal = get_backend('neal')
TRIALS=10; BUDGET=25; k=400

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
print(f'Prediction: {route}')
print(f'Correct: {w==route}')

json.dump({'name':'add20','n':n,'m':m,'beta_e':b,'asym':asym,
           'lam2':lam2,'fc':fcm,'fi':fim,'winner':w,'p':p,
           'pred':route,'correct':w==route},
    open('results/walshaw_add20_result.json','w'),indent=2)

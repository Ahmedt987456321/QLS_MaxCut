import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

def load_mtx(path):
    G = nx.Graph()
    with open(path) as f:
        lines = f.readlines()
    data_lines=[l.strip() for l in lines if not l.startswith('%') and l.strip()]
    for line in data_lines[1:]:
        parts=line.split()
        if len(parts)>=2:
            u,v=int(parts[0]),int(parts[1])
            if u!=v: G.add_edge(u,v)
    return nx.convert_node_labels_to_integers(G)

neal = get_backend('neal')
TRIALS=10; BUDGET=25

print('instance             n      beta_e  FConn    Fiedler  winner   p        pred     correct?')
print('-'*95)

results=[]
for name,beta_e_val,pred in [('delaunay_n10',0.0291,'Fiedler'),
                               ('delaunay_n11',0.0158,'Fiedler')]:
    G=load_mtx(f'data/delaunay/{name}.mtx')
    rng=np.random.default_rng(42)
    for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
    n=G.number_of_nodes()
    k=min(400, n//2)
    fc=[]; fi=[]
    for seed in range(TRIALS):
        s=seed*1000+42
        fc.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_frustrated_connected,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
        fi.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_fiedler,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fcm=statistics.median(fc); fim=statistics.median(fi)
    w='Fiedler' if fim>fcm else ('FConn' if fcm>fim else 'tie')
    try: _,p=mannwhitneyu(fc,fi,alternative='two-sided')
    except: p=1.0
    ok='YES' if w==pred else 'NO'
    print(f'{name:20} {n:5d} {beta_e_val:8.4f} {fcm:8.1f} {fim:8.1f} {w:>8} {p:8.4f} {pred:>8} {ok:>10}')
    results.append({'name':name,'n':n,'beta_e':beta_e_val,'fc':fcm,
                    'fi':fim,'winner':w,'p':p,'pred':pred,'correct':ok=='YES'})

json.dump(results,open('results/delaunay_results.json','w'),indent=2)
correct=sum(r['correct'] for r in results)
print(f'\nRouting correct: {correct}/{len(results)}')
print('Both predicted Fiedler -- if confirmed, Delaunay is in Fiedler regime')
print('This is the first genuinely spatially-embedded real-world test')

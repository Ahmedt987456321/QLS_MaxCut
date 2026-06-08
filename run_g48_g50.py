import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from src.graph import load_gset
from scipy.stats import mannwhitneyu

neal = get_backend('neal')
TRIALS=10; BUDGET=25

def beta_e_asym(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    v=nx.fiedler_vector(G_unw,method='lanczos')
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    asym=abs(abs(v.max())-abs(v.min()))/max(abs(v.max()),abs(v.min()))
    lam2=nx.algebraic_connectivity(G_unw,method='lanczos')
    return cut/m, lam2, asym

print('name      n   beta_e     lam2    FConn  Fiedler   winner        p  correct?')
print('-'*80)

results=[]
for name in ['G48','G49','G50']:
    G=load_gset(f'data/gset/{name}.txt')
    b,lam2,asym=beta_e_asym(G)
    k=640
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
    pred='Fiedler' if b<0.05 else 'FConn'
    ok='YES' if w==pred else 'NO'
    print(f'{name:6} {G.number_of_nodes():4d} {b:8.4f} {lam2:8.4f} {fcm:8.1f} {fim:8.1f} {w:>8} {p:8.4f} {ok:>8}')
    results.append({'name':name,'beta_e':b,'lam2':lam2,'fc':fcm,'fi':fim,
                    'winner':w,'p':p,'pred':pred,'correct':ok=='YES'})

json.dump(results,open('results/g48_g50_routing.json','w'),indent=2)
correct=sum(r['correct'] for r in results)
print(f'\nRouting correct: {correct}/{len(results)}')

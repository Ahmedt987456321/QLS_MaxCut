import numpy as np, networkx as nx, json
from src.graph import load_gset

def beta_e_asym(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    v=nx.fiedler_vector(G_unw,method='lanczos')
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    lam2=nx.algebraic_connectivity(G_unw,method='lanczos')
    asym=abs(abs(v.max())-abs(v.min()))/max(abs(v.max()),abs(v.min()))
    return cut/m, lam2, asym

BETA_STAR=0.05; ASYM_STAR=0.5

header = f"{'instance':10} {'n':>5} {'m':>6} {'lam2':>8} {'beta_e':>8} {'asym':>8} {'route':>10}"
print(header)
print('-'*65)

results=[]
for i in range(14,22):
    name = f'G{i}'
    try:
        G=load_gset(f'data/gset/{name}.txt')
        b,lam2,asym=beta_e_asym(G)
        route='Fiedler' if (b<BETA_STAR and asym<ASYM_STAR) else 'FConn'
        n=G.number_of_nodes(); m=G.number_of_edges()
        print(f'{name:10} {n:5d} {m:6d} {lam2:8.4f} {b:8.4f} {asym:8.4f} {route:>10}')
        results.append({'name':name,'n':n,'m':m,'lam2':lam2,
                        'beta_e':b,'asym':asym,'route':route})
    except Exception as e:
        print(f'{name}: {e}')

json.dump(results,open('results/g14_g21_beta.json','w'),indent=2)
print()
print(f'beta*={BETA_STAR}, asym*={ASYM_STAR}')
print('G14-G21 are planar/near-planar Gset instances')
print('Expected: beta_e near threshold 0.04-0.06')

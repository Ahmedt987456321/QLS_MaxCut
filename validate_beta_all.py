import numpy as np, networkx as nx, statistics, json, time
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset

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

def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    for line in lines[1:]:
        parts=line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return nx.convert_node_labels_to_integers(G)

def make_weighted(G, seed=42):
    rng = np.random.default_rng(seed)
    for u,v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = int(rng.choice([-1,1]))
    return G

def get_instances():
    instances = []
    for name, k in [('G11',400),('G12',400),('G13',400),('G32',640),('G33',640),('G34',640),('G48',640),('G49',640),('G50',640)]:
        instances.append((name, make_weighted(load_gset(f'data/gset/{name}.txt')), k))
    for name in ['G1','G22','G14','G15','G16','G17','G18','G19','G20','G21']:
        k = 160 if name=='G1' else 640 if name=='G22' else 400
        instances.append((name, make_weighted(load_gset(f'data/gset/{name}.txt')), k))
    for name, path, k in [('torusg3-8','data/dimacs/torusg3-8.dat',400),('toruspm3-8-50','data/dimacs/toruspm3-8-50.dat',400),('torusg3-15','data/dimacs/torusg3-15.dat',640),('toruspm3-15-50','data/dimacs/toruspm3-15-50.dat',640)]:
        instances.append((name, make_weighted(load_dimacs(path)), k))
    for name, path, k in [('delaunay_n10','data/delaunay/delaunay_n10.mtx',100),('delaunay_n11','data/delaunay/delaunay_n11.mtx',100)]:
        instances.append((name, make_weighted(load_mtx(path)), k))
    for name, path, k in [('walshaw_data','data/walshaw/data/data.mtx',400),('walshaw_3elt','data/walshaw/3elt/3elt.mtx',400),('walshaw_add20','data/walshaw/add20/add20.mtx',400)]:
        instances.append((name, make_weighted(load_mtx(path)), k))
    instances.append(('power_grid', make_weighted(load_mtx('data/roadnetwork/power/power.mtx')), 400))
    for d, k in [(3,400),(4,400),(5,320)]:
        for n in [800,1200]:
            instances.append((f'd{d}_reg_n{n}', make_weighted(nx.random_regular_graph(d,n,seed=42)), k))
    for n in [100,200]:
        rng=np.random.default_rng(42); G=nx.complete_graph(n)
        for u,v in G.edges(): G[u][v]['weight']=float(rng.normal(0,1))
        instances.append((f'SK_n{n}', G, min(n//2,200)))
    for n in [800,1200,1600]:
        G=nx.erdos_renyi_graph(n,2.0/n,seed=42)
        if not nx.is_connected(G):
            G=G.subgraph(max(nx.connected_components(G),key=len)).copy()
            G=nx.convert_node_labels_to_integers(G)
        instances.append((f'ER_n{n}', make_weighted(G), min(400,G.number_of_nodes()//2)))
    for m_param,n in [(2,800),(3,800)]:
        instances.append((f'BA_n{n}_m{m_param}', make_weighted(nx.barabasi_albert_graph(n,m_param,seed=42)), 400))
    for p in [0.01,0.1,0.5]:
        instances.append((f'WS_p{p}', make_weighted(nx.watts_strogatz_graph(800,4,p,seed=42)), 400))
    return instances

if __name__ == '__main__':
    neal = get_backend('neal')
    TRIALS=10; BUDGET=20
    instances = get_instances()
    print(f'Total instances: {len(instances)}')
    print(f'{"instance":25} {"route":>8} {"beta_e":>8} {"median":>10} {"time":>6}')
    print('-'*65)
    results = []
    for name, G, k in instances:
        t0 = time.time()
        try:
            cuts = []
            for seed in range(TRIALS):
                result = adaptive_qls(G, budget_seconds=BUDGET,
                    selector=select_beta_routed, backend=neal,
                    k_min=k, k_max=k, n_reads=100, seed=seed*1000+42, best_known=None)
                cuts.append(result[1].best_cut)
            cache = G.graph.get('_beta_routing_cache',{})
            route = cache.get('route','?')
            beta_e = cache.get('beta_e',-1)
            med = statistics.median(cuts)
            elapsed = time.time()-t0
            print(f'{name:25} {route:>8} {beta_e:>8.4f} {med:>10.1f} {elapsed:>5.0f}s')
            results.append({'name':name,'route':route,'beta_e':beta_e,'median':med,'k':k})
            json.dump(results,open('results/beta_routing_full_validation.json','w'),indent=2,default=str)
        except Exception as e:
            print(f'{name:25} ERROR: {e}')
    print(f'\nDone. {len(results)} instances. Results in results/beta_routing_full_validation.json')

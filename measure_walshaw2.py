import numpy as np, networkx as nx, json

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

BETA_STAR=0.05; ASYM_STAR=0.5

for name, path in [('data', 'data/walshaw/data/data.mtx'),
                   ('3elt', 'data/walshaw/3elt/3elt.mtx')]:
    G = load_mtx(path)
    rng=np.random.default_rng(42)
    for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
    b, lam2, asym = beta_e_asym(G)
    n=G.number_of_nodes(); m=G.number_of_edges()
    route='Fiedler' if (b<BETA_STAR and asym<ASYM_STAR) else 'FConn'
    print(f'{name}: n={n}, m={m}, lam2={lam2:.4f}, beta_e={b:.4f}, asym={asym:.4f} -> {route}')

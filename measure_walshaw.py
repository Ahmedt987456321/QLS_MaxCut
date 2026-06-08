import numpy as np, networkx as nx

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

G = load_mtx('data/walshaw/data/data.mtx')
n=G.number_of_nodes(); m=G.number_of_edges()
print(f'data FEM mesh: n={n}, m={m}, d_avg={2*m/n:.1f}')

v=nx.fiedler_vector(G, method='lanczos')
lam2=nx.algebraic_connectivity(G, method='lanczos')
k=n//2
nodes=list(G.nodes())
S=set(nodes[i] for i in np.argsort(v)[-k:])
cut=sum(1 for u,w in G.edges() if (u in S)!=(w in S))
beta=cut/m
asym=abs(abs(v.max())-abs(v.min()))/max(abs(v.max()),abs(v.min()))
route='Fiedler' if (beta<0.05 and asym<0.5) else 'FConn'

print(f'lam2={lam2:.4f}')
print(f'beta_e={beta:.4f}')
print(f'asymmetry={asym:.4f}')
print(f'routing: {route}')
print()
print('Expected: FEM mesh -> spatially embedded -> beta_e < 0.05 -> Fiedler')

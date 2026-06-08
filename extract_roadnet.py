import numpy as np, networkx as nx, gzip, json

print('Loading roadNet-PA...')
G_full = nx.DiGraph()
with gzip.open('data/roadnetwork/roadNet-PA.txt.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts)==2:
            G_full.add_edge(int(parts[0]), int(parts[1]))

print(f'Full graph: n={G_full.number_of_nodes()}, m={G_full.number_of_edges()}')

# Convert to undirected
G_full = G_full.to_undirected()

# Extract BFS subgraph of ~2000 nodes from node 0
print('Extracting BFS subgraph n~2000...')
start = 0
visited = []
queue = [start]
seen = {start}
while queue and len(visited) < 2000:
    node = queue.pop(0)
    visited.append(node)
    for neighbor in G_full.neighbors(node):
        if neighbor not in seen:
            seen.add(neighbor)
            queue.append(neighbor)

G_sub = G_full.subgraph(visited).copy()
G_sub = nx.convert_node_labels_to_integers(G_sub)
print(f'Subgraph: n={G_sub.number_of_nodes()}, m={G_sub.number_of_edges()}')
print(f'Connected: {nx.is_connected(G_sub)}')
print(f'd_avg={2*G_sub.number_of_edges()/G_sub.number_of_nodes():.2f}')

# Add +-1 weights
rng=np.random.default_rng(42)
for u,v in G_sub.edges(): G_sub[u][v]['weight']=int(rng.choice([-1,1]))

# Measure beta_e
v2=nx.fiedler_vector(G_sub, method='lanczos')
lam2=nx.algebraic_connectivity(G_sub, method='lanczos')
n=G_sub.number_of_nodes(); m=G_sub.number_of_edges()
k=n//2
nodes=list(G_sub.nodes())
S=set(nodes[i] for i in np.argsort(v2)[-k:])
cut=sum(1 for u,w in G_sub.edges() if (u in S)!=(w in S))
beta=cut/m
asym=abs(abs(v2.max())-abs(v2.min()))/max(abs(v2.max()),abs(v2.min()))
route='Fiedler' if (beta<0.05 and asym<0.5) else 'FConn'

print(f'\nroadNet-PA subgraph (n~2000):')
print(f'  lambda2={lam2:.4f}')
print(f'  beta_e={beta:.4f}')
print(f'  asymmetry={asym:.4f}')
print(f'  routing: {route}')
print()
print('Expected: road network -> planar, spatially embedded -> beta_e < 0.05 -> Fiedler')

# Save subgraph for AQLS
nx.write_edgelist(G_sub, 'data/roadnetwork/roadnet_pa_sub2000.edgelist',
                  data=['weight'])
print('Saved subgraph to data/roadnetwork/roadnet_pa_sub2000.edgelist')

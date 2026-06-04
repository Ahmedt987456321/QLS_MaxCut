import numpy as np, networkx as nx
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache

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

def measure_alignment(G, n_starts=20, seed=42):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    v2 = nx.fiedler_vector(G, method='lanczos')
    fiedler_sign = np.sign(v2)
    alignments = []
    for trial in range(n_starts):
        x = random_cut(G, rng)
        gc = GainCache()
        x_opt, gc, _, _ = one_flip_ls(G, x, gc)
        s_star = np.array([1 if x_opt[v]==1 else -1 for v in nodes])
        alignment = abs(np.dot(fiedler_sign, s_star)) / n
        alignments.append(alignment)
    return np.mean(alignments), np.std(alignments)

header = f"{'graph':20} {'n':>5} {'mean_align':>12} {'std_align':>12} {'useful?':>15}"
print(header)
print('-'*70)

instances = [
    ('G11 toroidal', 'gset', 'G11', 'Fiedler wins'),
    ('G13 toroidal', 'gset', 'G13', 'Fiedler wins'),
    ('G1 dense',     'gset', 'G1',  'FConn wins'),
]

for label, src, name, expected in instances:
    G = load_gset(f'data/gset/{name}.txt')
    G_unw = nx.Graph()
    G_unw.add_nodes_from(G.nodes())
    G_unw.add_edges_from(G.edges())
    mean, std = measure_alignment(G_unw)
    useful = 'HIGH' if mean > 0.15 else 'LOW'
    print(f'{label:20} {G_unw.number_of_nodes():5d} {mean:12.4f} {std:12.4f} {useful:>15}')

for name in ['delaunay_n10', 'delaunay_n11']:
    G = load_mtx(f'data/delaunay/{name}.mtx')
    rng = np.random.default_rng(42)
    for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))
    mean, std = measure_alignment(G)
    useful = 'HIGH' if mean > 0.15 else 'LOW'
    print(f'{name:20} {G.number_of_nodes():5d} {mean:12.4f} {std:12.4f} {useful:>15}')

print()
print('Threshold: alignment > 0.15 -> Fiedler useful')
print('Hypothesis: G11/G13 HIGH, Delaunay LOW, G1 LOW')

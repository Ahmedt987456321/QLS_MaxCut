import networkx as nx
from src.graph import load_gset

for name in ["G11", "G12", "G13", "G14", "G22", "G1"]:
    G = load_gset(f"data/gset/{name}.txt")
    ws = [d['weight'] for _, _, d in G.edges(data=True)]
    n_neg = sum(1 for w in ws if w < 0)
    print(f"{name}: min_w={min(ws)} max_w={max(ws)} "
          f"n_neg={n_neg} bipartite={nx.is_bipartite(G)}")
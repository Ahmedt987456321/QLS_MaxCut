import networkx as nx
from src.graph import load_gset
from src.selectors import is_signed_graph_balanced

for name in ["G11", "G12", "G13", "G14", "G22", "G1"]:
    G = load_gset(f"data/gset/{name}.txt")
    is_bal, eig = is_signed_graph_balanced(G)
    print(f"{name}: balance_says={is_bal} (eig={eig:.4f})  "
          f"nx.is_bipartite={nx.is_bipartite(G)}")
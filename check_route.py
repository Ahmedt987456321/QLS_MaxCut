import networkx as nx
from src.graph import load_gset
for name in ["G11","G12","G13","G14","G22","G1"]:
    G = load_gset(f"data/gset/{name}.txt")
    pick = "fiedler" if nx.is_bipartite(G) else "fconn"
    print(f"{name}: routes to -> {pick}")
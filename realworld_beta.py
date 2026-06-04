"""Fallback: use NetworkX built-in real-world graphs + generated
scale-free and small-world networks."""
import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh

def beta_e(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    if m==0: return None
    L=nx.laplacian_matrix(G_unw,nodelist=nodes).astype(float)
    vals,vecs=eigsh(L,k=2,sigma=0,which='LM')
    v=vecs[:,np.argsort(vals)[1]]
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    return cut/m, nx.algebraic_connectivity(G_unw, method="lanczos")

BETA_STAR = 0.05

print(f"{'graph':30} {'n':>5} {'m':>7} {'lam2':>8} {'beta_e':>8} "
      f"{'prediction':>12} {'type':>15}")
print("-"*90)

graphs = [
    # Built-in NetworkX real-world graphs
    ("karate_club",        nx.karate_club_graph(),          "social"),
    ("les_miserables",     nx.les_miserables_graph(),       "social"),
    ("davis_women",        nx.davis_southern_women_graph(),  "bipartite"),
    ("florentine",         nx.florentine_families_graph(),   "social"),
    # Scale-free (Barabasi-Albert) -- common real-world structure
    ("BA_n800_m2",         nx.barabasi_albert_graph(800,2,seed=42),  "scale-free"),
    ("BA_n800_m3",         nx.barabasi_albert_graph(800,3,seed=42),  "scale-free"),
    ("BA_n1600_m2",        nx.barabasi_albert_graph(1600,2,seed=42), "scale-free"),
    # Small-world (Watts-Strogatz)
    ("WS_n800_k4_p01",     nx.watts_strogatz_graph(800,4,0.1,seed=42),  "small-world"),
    ("WS_n800_k4_p05",     nx.watts_strogatz_graph(800,4,0.5,seed=42),  "small-world"),
    ("WS_n1200_k6_p01",    nx.watts_strogatz_graph(1200,6,0.1,seed=42), "small-world"),
    # Power-law cluster (Holme-Kim)
    ("HK_n800_m2_p05",     nx.powerlaw_cluster_graph(800,2,0.5,seed=42),  "powerlaw"),
    ("HK_n800_m3_p05",     nx.powerlaw_cluster_graph(800,3,0.5,seed=42),  "powerlaw"),
]

results = []
for name, G, gtype in graphs:
    # ensure connected
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 10 or m < 10:
        continue
    # add +-1 weights
    rng = np.random.default_rng(42)
    for u,v in G.edges():
        G[u][v]["weight"] = int(rng.choice([-1,1]))
    try:
        b, lam2 = beta_e(G)
        pred = "Fiedler" if b < BETA_STAR else "FConn"
        print(f"{name:30} {n:5d} {m:7d} {lam2:8.4f} {b:8.4f} "
              f"{pred:>12} {gtype:>15}")
        results.append({"name":name,"n":n,"m":m,"lam2":lam2,
                        "beta_e":b,"pred":pred,"type":gtype})
    except Exception as e:
        print(f"{name}: ERROR {e}")

import json
json.dump(results, open("results/realworld_beta.json","w"), indent=2)
print(f"\nbeta* = {BETA_STAR}")
print(f"Fiedler predicted: {sum(1 for r in results if r['pred']=='Fiedler')}")
print(f"FConn predicted:   {sum(1 for r in results if r['pred']=='FConn')}")

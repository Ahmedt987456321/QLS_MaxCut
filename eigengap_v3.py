"""EIGENGAP v3: FIX the k=n bug. Graphs now n=800 so k=400 < n and the
selectors actually CHOOSE different supports (the v2 ties were because n=k=400,
forcing both selectors to pick all vertices -> trivially identical)."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8; K = 400; FRAC_NEG = 0.5   # K < n now (n=800)

def bottom(G, kk=4):
    L = np.asarray(nx.laplacian_matrix(G).astype(float).todense())
    return np.sort(np.linalg.eigvalsh(L))[:kk]

def lobe(n, deg, seed):
    return nx.connected_watts_strogatz_graph(n, deg, 0.3, tries=200, seed=seed)

def two_lobe(lobe_n, deg, n_bridge, seed):       # n = 2*lobe_n
    A=lobe(lobe_n,deg,seed); B=lobe(lobe_n,deg,seed+1)
    G=nx.disjoint_union(A,B); rng=np.random.default_rng(seed)
    aN=list(range(lobe_n)); bN=list(range(lobe_n,2*lobe_n))
    for _ in range(n_bridge): G.add_edge(int(rng.choice(aN)),int(rng.choice(bN)))
    return G

def ring_lobes(lobe_n, n_lobes, n_bridge, deg, seed):  # n = lobe_n*n_lobes
    parts=[lobe(lobe_n,deg,seed+i) for i in range(n_lobes)]
    G=parts[0]; offs=[0]
    for p in parts[1:]: offs.append(G.number_of_nodes()); G=nx.disjoint_union(G,p)
    rng=np.random.default_rng(seed)
    for i in range(n_lobes):
        a0=offs[i]; b0=offs[(i+1)%n_lobes]
        aN=list(range(a0,a0+lobe_n)); bN=list(range(b0,b0+lobe_n))
        for _ in range(n_bridge): G.add_edge(int(rng.choice(aN)),int(rng.choice(bN)))
    return G

def signed(G,seed):
    G=G.copy(); rng=np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=-1 if rng.random()<FRAC_NEG else 1
    return G

def med(G,sel):
    return stats.median([adaptive_qls(G,budget_seconds=30,selector=sel,backend=neal,
        k_min=K,k_max=K,n_reads=100,seed=s,best_known=None)[1].best_cut for s in range(SEEDS)])

# n=800 graphs: 2x400 (large gap) vs 8x100 / 4x200 (small gap)
specs=[
    ("large_gap_2lobe",  two_lobe(400, 10, 6, seed=10)),   # n=800
    ("small_gap_8lobe",  ring_lobes(100, 8, 2, 10, seed=10)),  # n=800
    ("large_gap_2lobe_b",two_lobe(400, 10, 10, seed=20)),  # n=800
    ("small_gap_4lobe",  ring_lobes(200, 4, 3, 10, seed=20)),  # n=800
]
print(f"{'family':18} {'conn':>5} {'n':>5} {'lam2':>8} {'lam3':>8} {'gap':>8} "
      f"{'FConn':>7} {'Fiedler':>8} {'winner':>8} {'margin':>7}")
rows=[]
for name,G in specs:
    G=nx.convert_node_labels_to_integers(G)
    conn=nx.is_connected(G); b=bottom(G,4)
    lam2,lam3=float(b[1]),float(b[2]); gap=lam3-lam2
    Gs=signed(G,42)
    fc=med(Gs,select_frustrated_connected); fi=med(Gs,select_fiedler)
    winner="Fiedler" if fi>fc else ("FConn" if fc>fi else "tie")
    print(f"{name:18} {str(conn):>5} {G.number_of_nodes():5d} {lam2:8.4f} {lam3:8.4f} "
          f"{gap:8.4f} {fc:7.0f} {fi:8.0f} {winner:>8} {abs(fi-fc):7.0f}")
    rows.append({"family":name,"connected":conn,"n":G.number_of_nodes(),
                 "lam2":lam2,"lam3":lam3,"gap":gap,"FConn":fc,"Fiedler":fi,
                 "winner":winner,"margin":abs(fi-fc)})
print("\nNow k=400 < n=800: selectors choose DIFFERENT supports.")
print("Read: large-gap vs small-gap at comparable lam2 -> does winner track gap?")
json.dump(rows, open("results/eigengap_v3.json","w"), indent=2, default=str)

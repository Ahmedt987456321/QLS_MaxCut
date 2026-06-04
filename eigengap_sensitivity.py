"""SENSITIVITY SWEEP for the eigengap conclusion.
Hold the contrast (large-gap vs small-gap, matched lambda2). Vary the
parameters we did NOT mean to test: lobe degree and frustration.
Conclusion 'gap does not change winner' is ROBUST only if it survives all cells.
n=800, k=400 (k<n, the v2 bug fixed)."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal"); SEEDS=6; K=400

def bottom(G,kk=4):
    L=np.asarray(nx.laplacian_matrix(G).astype(float).todense())
    return np.sort(np.linalg.eigvalsh(L))[:kk]
def lobe(n,deg,seed): return nx.connected_watts_strogatz_graph(n,deg,0.3,tries=200,seed=seed)
def two_lobe(lobe_n,deg,nb,seed):   # large gap
    A=lobe(lobe_n,deg,seed); B=lobe(lobe_n,deg,seed+1); G=nx.disjoint_union(A,B)
    rng=np.random.default_rng(seed); aN=list(range(lobe_n)); bN=list(range(lobe_n,2*lobe_n))
    for _ in range(nb): G.add_edge(int(rng.choice(aN)),int(rng.choice(bN)))
    return G
def ring_lobes(lobe_n,nl,nb,deg,seed):  # small gap
    parts=[lobe(lobe_n,deg,seed+i) for i in range(nl)]; G=parts[0]; offs=[0]
    for p in parts[1:]: offs.append(G.number_of_nodes()); G=nx.disjoint_union(G,p)
    rng=np.random.default_rng(seed)
    for i in range(nl):
        a0=offs[i]; b0=offs[(i+1)%nl]; aN=list(range(a0,a0+lobe_n)); bN=list(range(b0,b0+lobe_n))
        for _ in range(nb): G.add_edge(int(rng.choice(aN)),int(rng.choice(bN)))
    return G
def signed(G,fn,seed):
    G=G.copy(); rng=np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=-1 if rng.random()<fn else 1
    return G
def med(G,sel):
    return stats.median([adaptive_qls(G,budget_seconds=20,selector=sel,backend=neal,
        k_min=K,k_max=K,n_reads=100,seed=s,best_known=None)[1].best_cut for s in range(SEEDS)])

# sweep degree x frustration; for each cell, build a matched-lambda2 large/small-gap pair
print(f"{'deg':>4} {'fn':>5} {'gaptype':>10} {'lam2':>7} {'gap':>8} "
      f"{'FConn':>7} {'Fiedler':>8} {'winner':>8}")
rows=[]
for deg in [6, 12]:
    for fn in [0.1, 0.5]:
        pair = [("large", two_lobe(400,deg,8,seed=10)),
                ("small", ring_lobes(200,4,3,deg,seed=10))]
        cell_winners={}
        for gaptype,G in pair:
            G=nx.convert_node_labels_to_integers(G)
            b=bottom(G,4); lam2,lam3=float(b[1]),float(b[2]); gap=lam3-lam2
            Gs=signed(G,fn,42)
            fc=med(Gs,select_frustrated_connected); fi=med(Gs,select_fiedler)
            w="Fiedler" if fi>fc else ("FConn" if fc>fi else "tie")
            cell_winners[gaptype]=w
            print(f"{deg:>4} {fn:>5} {gaptype:>10} {lam2:7.4f} {gap:8.4f} "
                  f"{fc:7.0f} {fi:8.0f} {w:>8}")
            rows.append({"deg":deg,"fn":fn,"gaptype":gaptype,"lam2":lam2,"gap":gap,
                         "FConn":fc,"Fiedler":fi,"winner":w})
        same = cell_winners.get("large")==cell_winners.get("small")
        print(f"     -> deg={deg} fn={fn}: gap changes winner? "
              f"{'NO (robust)' if same else 'YES (gap matters here!)'}\n")

print("=== ROBUSTNESS READ ===")
print("Conclusion 'gap does NOT change winner' is robust ONLY if every cell shows")
print("the SAME winner for large-gap and small-gap. If any cell flips, the gap")
print("(or its interaction with deg/frustration) DOES matter -> earlier conclusion biased.")
json.dump(rows, open("results/eigengap_sensitivity.json","w"), indent=2, default=str)

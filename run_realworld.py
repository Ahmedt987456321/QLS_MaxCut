import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=20

graphs = [
    ("BA_n800_m2",   nx.barabasi_albert_graph(800,2,seed=42),  "scale-free"),
    ("BA_n800_m3",   nx.barabasi_albert_graph(800,3,seed=42),  "scale-free"),
    ("WS_n800_k4_p01", nx.watts_strogatz_graph(800,4,0.1,seed=42), "small-world"),
    ("WS_n800_k4_p05", nx.watts_strogatz_graph(800,4,0.5,seed=42), "small-world"),
    ("HK_n800_m2_p05", nx.powerlaw_cluster_graph(800,2,0.5,seed=42), "powerlaw"),
]

print(f"{'graph':20} {'type':12} {'FConn':>8} {'Fiedler':>8} "
      f"{'winner':>8} {'p':>8} {'pred':>8} {'correct?':>10}")
print("-"*85)

results=[]
for name, G, gtype in graphs:
    if not nx.is_connected(G):
        largest=max(nx.connected_components(G),key=len)
        G=G.subgraph(largest).copy()
        G=nx.convert_node_labels_to_integers(G)
    rng=np.random.default_rng(42)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    k=min(400, G.number_of_nodes()//2)
    fc=[]; fi=[]
    for seed in range(TRIALS):
        s=seed*1000+42
        fc.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_frustrated_connected,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
        fi.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_fiedler,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fcm=statistics.median(fc); fim=statistics.median(fi)
    w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
    try: _,p=mannwhitneyu(fc,fi,alternative="two-sided")
    except: p=1.0
    pred="FConn"  # all predicted FConn by beta_e
    ok="YES" if w==pred else "NO"
    print(f"{name:20} {gtype:12} {fcm:8.1f} {fim:8.1f} "
          f"{w:>8} {p:8.4f} {pred:>8} {ok:>10}")
    results.append({"name":name,"type":gtype,"fc":fcm,"fi":fim,
                    "winner":w,"p":p,"correct":ok=="YES"})

json.dump(results,open("results/realworld_results.json","w"),indent=2)
correct=sum(r["correct"] for r in results)
print(f"\nRouting correct: {correct}/{len(results)}")
print("Prediction: FConn wins on all real-world/realistic graphs")

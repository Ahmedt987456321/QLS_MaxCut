import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=15; N_INST=3

print(f"{'d':>3} {'n':>6} {'FConn':>8} {'Fiedler':>8} {'winner':>8} {'p':>8} {'correct?':>10}")
print("-"*65)
summary=[]
for d in [3,4,5]:
    for n in [800,1200]:
        k={3:400,4:400,5:320}[d]
        fc=[]; fi=[]
        for seed in range(N_INST):
            G=nx.random_regular_graph(d,n,seed=seed)
            rng=np.random.default_rng(seed)
            for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
            for t in range(TRIALS):
                s=seed*1000+t
                fc.append(adaptive_qls(G,budget_seconds=BUDGET,selector=select_frustrated_connected,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
                fi.append(adaptive_qls(G,budget_seconds=BUDGET,selector=select_fiedler,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
        fcm=statistics.median(fc); fim=statistics.median(fi)
        w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
        try: _,p=mannwhitneyu(fc,fi,alternative="two-sided")
        except: p=1.0
        exp="Fiedler" if d==3 else "FConn"
        ok=w==exp
        print(f"{d:3d} {n:6d} {fcm:8.1f} {fim:8.1f} {w:>8} {p:8.4f} {'YES' if ok else 'NO':>10}")
        summary.append({"d":d,"n":n,"fc":fcm,"fi":fim,"winner":w,"p":p,"expected":exp,"correct":ok})
json.dump(summary,open("results/stage2_fast.json","w"),indent=2)
print(f"\nCorrect: {sum(s['correct'] for s in summary)}/{len(summary)}")

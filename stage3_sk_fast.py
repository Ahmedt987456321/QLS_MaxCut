import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=15; N_INST=3

print(f"{'n':>6} {'lam2':>8} {'FConn':>8} {'Fiedler':>8} {'winner':>8} {'p':>8} {'correct?':>10}")
print("-"*65)
summary=[]
for n in [100,200,400]:
    k=min(n//2,200)
    fc=[]; fi=[]; lams=[]
    for seed in range(N_INST):
        rng=np.random.default_rng(seed)
        G=nx.complete_graph(n)
        for u,v in G.edges(): G[u][v]["weight"]=float(rng.normal(0,1))
        lams.append(nx.algebraic_connectivity(G,method="lanczos"))
        for t in range(TRIALS):
            s=seed*1000+t
            fc.append(adaptive_qls(G,budget_seconds=BUDGET,selector=select_frustrated_connected,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
            fi.append(adaptive_qls(G,budget_seconds=BUDGET,selector=select_fiedler,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fcm=statistics.median(fc); fim=statistics.median(fi)
    w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
    try: _,p=mannwhitneyu(fc,fi,alternative="two-sided")
    except: p=1.0
    ok=w=="FConn"
    print(f"{n:6d} {np.mean(lams):8.2f} {fcm:8.1f} {fim:8.1f} {w:>8} {p:8.4f} {'YES' if ok else 'NO':>10}")
    summary.append({"n":n,"lam2":np.mean(lams),"fc":fcm,"fi":fim,"winner":w,"p":p,"correct":ok})
json.dump(summary,open("results/stage3_sk_fast.json","w"),indent=2)
print(f"\nCorrect: {sum(s['correct'] for s in summary)}/{len(summary)}")

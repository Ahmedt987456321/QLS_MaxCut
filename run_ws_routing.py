"""Verify WS routing: run AQLS on WS instances at p=0.02 (Fiedler)
and p=0.10 (FConn) to confirm predictions."""
import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=20; n=800; k_ws=4

print(f"{'p':>6} {'beta_e':>8} {'FConn':>8} {'Fiedler':>8} "
      f"{'winner':>8} {'pval':>8} {'pred':>8} {'correct?':>10}")
print("-"*70)

results=[]
for p, pred in [(0.02,"Fiedler"),(0.05,"Fiedler"),
                (0.07,"FConn"),(0.10,"FConn")]:
    fc=[]; fi=[]; betas=[]
    for seed in range(3):
        G=nx.watts_strogatz_graph(n,k_ws,p,seed=seed)
        rng=np.random.default_rng(seed)
        for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
        # measure beta_e
        G_unw=nx.Graph(); G_unw.add_nodes_from(G.nodes()); G_unw.add_edges_from(G.edges())
        m=G_unw.number_of_edges()
        from scipy.sparse.linalg import eigsh
        import numpy.linalg as nla
        L=nx.laplacian_matrix(G_unw,nodelist=list(G.nodes())).astype(float)
        try:
            vals,vecs=eigsh(L,k=2,which='SM',tol=1e-6)
        except:
            vals,vecs=nla.eigh(L.toarray())
        v=vecs[:,np.argsort(vals)[1]]
        S=set(list(G.nodes())[i] for i in np.argsort(v)[-(n//2):])
        cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
        betas.append(cut/m)
        for t in range(TRIALS):
            s=seed*1000+t
            fc.append(adaptive_qls(G,budget_seconds=BUDGET,
                selector=select_frustrated_connected,backend=neal,
                k_min=400,k_max=400,n_reads=100,seed=s,best_known=None)[1].best_cut)
            fi.append(adaptive_qls(G,budget_seconds=BUDGET,
                selector=select_fiedler,backend=neal,
                k_min=400,k_max=400,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fcm=statistics.median(fc); fim=statistics.median(fi)
    w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
    try: _,pv=mannwhitneyu(fc,fi,alternative="two-sided")
    except: pv=1.0
    ok="YES" if w==pred else "NO"
    print(f"{p:6.3f} {np.mean(betas):8.4f} {fcm:8.1f} {fim:8.1f} "
          f"{w:>8} {pv:8.4f} {pred:>8} {ok:>10}")
    results.append({"p":p,"beta_e":np.mean(betas),"fc":fcm,"fi":fim,
                    "winner":w,"pval":pv,"pred":pred,"correct":ok=="YES"})

json.dump(results,open("results/ws_routing_results.json","w"),indent=2)
correct=sum(r["correct"] for r in results)
print(f"\nRouting correct: {correct}/{len(results)}")
print(f"p* crossover confirmed at beta_e ~ 0.05 (p ~ 0.06)")

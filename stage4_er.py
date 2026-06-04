"""Stage 4: Sparse Erdos-Renyi G(n, 2/n) instances.
Near-critical random graphs, lambda2 -> 0, often below safe band.
Prediction: Fiedler should win (very low lambda2).
BUT: some instances may be disconnected (lambda2=0) -- handle explicitly."""
import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=20; N_INST=3

print(f"{'n':>6} {'p_edge':>8} {'lam2':>8} {'beta_e':>8} "
      f"{'FConn':>8} {'Fiedler':>8} {'winner':>8} {'p':>8} {'correct?':>10}")
print("-"*80)

def fiedler_beta_e(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    if m==0: return None
    from scipy.sparse.linalg import eigsh
    L=nx.laplacian_matrix(G_unw,nodelist=nodes).astype(float)
    vals,vecs=eigsh(L,k=2,sigma=0,which='LM')
    v=vecs[:,np.argsort(vals)[1]]
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    lam2=nx.algebraic_connectivity(G_unw,method="lanczos")
    return {"lam2":lam2,"beta_e":cut/m,"m":m}

summary=[]
for n in [800, 1200, 1600]:
    p = 2.0/n   # sparse: average degree ~2, near critical
    fc=[]; fi=[]; lam2s=[]; betas=[]
    skipped=0
    for seed in range(N_INST):
        rng=np.random.default_rng(seed)
        G=nx.erdos_renyi_graph(n, p, seed=seed)
        # add +-1 weights
        for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
        # check connectivity
        if not nx.is_connected(G):
            # use largest connected component
            largest=max(nx.connected_components(G),key=len)
            G=G.subgraph(largest).copy()
            G=nx.convert_node_labels_to_integers(G)
        r=fiedler_beta_e(G)
        if r is None:
            skipped+=1; continue
        lam2s.append(r['lam2']); betas.append(r['beta_e'])
        for t in range(TRIALS):
            s=seed*1000+t
            k_val=min(400,G.number_of_nodes()//2)
            fc.append(adaptive_qls(G,budget_seconds=BUDGET,
                selector=select_frustrated_connected,backend=neal,
                k_min=k_val,k_max=k_val,n_reads=100,seed=s,best_known=None)[1].best_cut)
            fi.append(adaptive_qls(G,budget_seconds=BUDGET,
                selector=select_fiedler,backend=neal,
                k_min=k_val,k_max=k_val,n_reads=100,seed=s,best_known=None)[1].best_cut)
    if fc and fi:
        fcm=statistics.median(fc); fim=statistics.median(fi)
        w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
        try: _,pv=mannwhitneyu(fc,fi,alternative="two-sided")
        except: pv=1.0
        mean_lam2=np.mean(lam2s); mean_beta=np.mean(betas)
        # prediction: lambda2 very small -> Fiedler
        # BUT beta_e may be large on sparse disconnected-ish graphs
        pred_lam2="Fiedler" if mean_lam2<0.2 else "FConn"
        pred_beta="Fiedler" if mean_beta<0.10 else "FConn"
        print(f"{n:6d} {p:8.5f} {mean_lam2:8.4f} {mean_beta:8.4f} "
              f"{fcm:8.1f} {fim:8.1f} {w:>8} {pv:8.4f} "
              f"lam2={pred_lam2} beta={pred_beta}")
        summary.append({"n":n,"p":p,"lam2":mean_lam2,"beta_e":mean_beta,
            "fc":fcm,"fi":fim,"winner":w,"pval":pv,
            "pred_lam2":pred_lam2,"pred_beta":pred_beta,
            "lam2_correct":w==pred_lam2,"beta_correct":w==pred_beta})

json.dump(summary,open("results/stage4_er_results.json","w"),indent=2)
print(f"\nNote: G(n,2/n) is near-critical -- many instances are disconnected")
print(f"or have tree-like components. Lambda2 may be very small or 0.")
print(f"This tests the LOWER bound of the safe band (lambda2 < 0.004).")

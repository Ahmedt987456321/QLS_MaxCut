import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh

def beta_e(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    if m==0: return None
    L=nx.laplacian_matrix(G_unw,nodelist=nodes).astype(float)
    # use which='SM' to avoid singularity issues at p=0
    try:
        vals,vecs=eigsh(L,k=2,which='SM',tol=1e-6)
    except Exception:
        # fallback: dense eigen for small graphs
        import numpy.linalg as nla
        vals,vecs=nla.eigh(L.toarray())
    v=vecs[:,np.argsort(vals)[1]]
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    return cut/m

BETA_STAR=0.05; n=800; k_ws=4

print(f"{'p':>8} {'beta_e':>8} {'prediction':>12} {'note':>20}")
print("-"*55)

for p in [0.0,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.20,0.50]:
    betas=[]
    for seed in range(5):
        G=nx.watts_strogatz_graph(n,k_ws,p,seed=seed)
        rng=np.random.default_rng(seed)
        for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
        b=beta_e(G)
        if b is not None: betas.append(b)
    if betas:
        mean_b=np.mean(betas)
        pred="Fiedler" if mean_b<BETA_STAR else "FConn"
        note="<-- crossover?" if abs(mean_b-BETA_STAR)<0.02 else ""
        print(f"{p:8.3f} {mean_b:8.4f} {pred:>12} {note:>20}")

print(f"\nbeta*={BETA_STAR}, theory crossover p*~0.03-0.07")
print(f"p=0 ring lattice: expected beta_e~0.004")

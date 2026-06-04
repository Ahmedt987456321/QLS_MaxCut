from src.graph import load_gset
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# best-known cut values for ratio context
best = {"G11":564,"G12":556,"G13":582,"G32":1410,"G33":1382,
        "G34":1384,"G48":6000,"G49":6000,"G50":5880}

print(f'{"inst":5} {"l1":>8} {"l2":>8} {"gap":>8} {"eff_n":>8}')
print("-"*44)
for name in ["G11","G12","G13","G32","G33","G34","G48","G49","G50"]:
    G = load_gset(f"data/gset/{name}.txt")
    nodes = list(G.nodes()); n = len(nodes)
    idx = {v:i for i,v in enumerate(nodes)}
    rows,cols,vals=[],[],[]; deg=np.zeros(n)
    for u,v,data in G.edges(data=True):
        w=data.get("weight",1.0); s=np.sign(w) if w!=0 else 1.0; wa=abs(w)
        i,j=idx[u],idx[v]
        rows+=[i,j];cols+=[j,i];vals+=[-s*wa,-s*wa]; deg[i]+=wa;deg[j]+=wa
    L=sp.diags(deg)+sp.csr_matrix((vals,(rows,cols)),shape=(n,n))
    ev,evec=spla.eigsh(L,k=2,which="SM",tol=1e-7,maxiter=10000)
    o=np.argsort(ev); ev=ev[o]; v0=evec[:,o[0]]; v0=v0/np.linalg.norm(v0)
    eff_n=1.0/float(np.sum(v0**4))
    print(f"{name:5} {ev[0]:8.4f} {ev[1]:8.4f} {ev[1]-ev[0]:8.4f} {eff_n:8.1f}")

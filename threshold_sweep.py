import numpy as np, networkx as nx
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def beta_e(G):
    nodes=list(G.nodes()); n=len(nodes)
    G_unw=nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges()
    L=nx.laplacian_matrix(G_unw,nodelist=nodes).astype(float)
    vals,vecs=eigsh(L,k=2,sigma=0,which='LM')
    v=vecs[:,np.argsort(vals)[1]]
    k=n//2
    S=set(nodes[i] for i in np.argsort(v)[-k:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    return cut/m

def load_dimacs(path):
    G=nx.Graph()
    with open(path) as f: lines=f.readlines()
    for line in lines[1:]:
        parts=line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

print(f"{'instance':25} {'beta_e':>8} {'actual':>10} "
      f"{'route@0.10':>12} {'route@0.05':>12} {'route@0.03':>12}")
print("-"*80)

instances = [
    ("G11","Fiedler","gset"),("G12","Fiedler","gset"),
    ("G13","Fiedler","gset"),("G32","Fiedler","gset"),
    ("G33","Fiedler","gset"),("G34","Fiedler","gset"),
    ("G1","FConn","gset"),("G22","FConn","gset"),
]
for name,actual,_ in instances:
    try:
        G=load_gset(f"data/gset/{name}.txt")
        b=beta_e(G)
        r10="Fiedler" if b<0.10 else "FConn"
        r05="Fiedler" if b<0.05 else "FConn"
        r03="Fiedler" if b<0.03 else "FConn"
        ok10="?" if r10==actual else "?"
        ok05="?" if r05==actual else "?"
        ok03="?" if r03==actual else "?"
        print(f"{name:25} {b:8.4f} {actual:>10} "
              f"{r10+ok10:>12} {r05+ok05:>12} {r03+ok03:>12}")
    except: pass

dimacs=[
    ("torusg3-8","FConn","data/dimacs/torusg3-8.dat"),
    ("toruspm3-8-50","FConn","data/dimacs/toruspm3-8-50.dat"),
    ("torusg3-15","FConn","data/dimacs/torusg3-15.dat"),
    ("toruspm3-15-50","FConn","data/dimacs/toruspm3-15-50.dat"),
]
for name,actual,path in dimacs:
    G=load_dimacs(path)
    b=beta_e(G)
    r10="Fiedler" if b<0.10 else "FConn"
    r05="Fiedler" if b<0.05 else "FConn"
    r03="Fiedler" if b<0.03 else "FConn"
    ok10="?" if r10==actual else "?"
    ok05="?" if r05==actual else "?"
    ok03="?" if r03==actual else "?"
    print(f"{name:25} {b:8.4f} {actual:>10} "
          f"{r10+ok10:>12} {r05+ok05:>12} {r03+ok03:>12}")

# d=3 regular
for seed in range(2):
    G=nx.random_regular_graph(3,800,seed=seed)
    rng=np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
    b=beta_e(G)
    r10="Fiedler" if b<0.10 else "FConn"
    r05="Fiedler" if b<0.05 else "FConn"
    r03="Fiedler" if b<0.03 else "FConn"
    actual="FConn"
    ok10="?" if r10==actual else "?"
    ok05="?" if r05==actual else "?"
    ok03="?" if r03==actual else "?"
    print(f"{'d3-reg-s'+str(seed):25} {b:8.4f} {actual:>10} "
          f"{r10+ok10:>12} {r05+ok05:>12} {r03+ok03:>12}")

print(f"\nLooking for a threshold that separates:")
print(f"  G-set toroidal (Fiedler wins): beta_e = 0.008-0.031")
print(f"  DIMACS 3D torus (FConn wins):  beta_e = 0.075-0.151")
print(f"  d=3 regular (FConn wins):      beta_e = 0.112-0.125")
print(f"\nThreshold 0.05 would separate G-set toroidal from everything else.")
print(f"Threshold 0.03 might be too tight (could exclude G13 at 0.031).")

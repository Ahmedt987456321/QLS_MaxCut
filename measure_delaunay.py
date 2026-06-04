import numpy as np, networkx as nx
import numpy.linalg as nla

def load_mtx(path):
    G = nx.Graph()
    with open(path) as f:
        lines = f.readlines()
    data_lines = [l.strip() for l in lines
                  if not l.startswith("%") and l.strip()]
    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            u,v = int(parts[0]),int(parts[1])
            if u != v: G.add_edge(u,v)
    return G

def beta_e(G):
    # relabel to 0..n-1 to ensure contiguous indices
    G = nx.convert_node_labels_to_integers(G)
    nodes = list(range(G.number_of_nodes()))
    n = len(nodes); m = G.number_of_edges()
    if m == 0: return None, None
    # use sparse eigsh with shift-invert for stability
    from scipy.sparse.linalg import eigsh
    L = nx.laplacian_matrix(G, nodelist=nodes).astype(float)
    # get 2 smallest eigenvalues
    try:
        vals, vecs = eigsh(L, k=2, which='SM', tol=1e-8,
                          maxiter=10000)
    except Exception:
        # fallback dense
        Ld = L.toarray()
        vals, vecs = nla.eigh(Ld)
    idx = np.argsort(vals)
    lam2 = float(vals[idx[1]])
    v = vecs[:, idx[1]]
    k = n // 2
    S = set(i for i in np.argsort(v)[-k:])
    cut = sum(1 for u,w in G.edges() if (u in S) != (w in S))
    return cut/m, lam2

BETA_STAR = 0.05
print(f"{'instance':20} {'n':>5} {'m':>6} {'lam2':>8} "
      f"{'beta_e':>8} {'prediction':>12}")
print("-"*65)

for name in ["delaunay_n10","delaunay_n11"]:
    G = load_mtx(f"data/delaunay/{name}.mtx")
    rng = np.random.default_rng(42)
    for u,v in G.edges(): G[u][v]["weight"] = int(rng.choice([-1,1]))
    b, lam2 = beta_e(G)
    n=G.number_of_nodes(); m=G.number_of_edges()
    pred = "Fiedler" if b < BETA_STAR else "FConn"
    print(f"{name:20} {n:5d} {m:6d} {lam2:8.4f} "
          f"{b:8.4f} {pred:>12}")

print(f"\nbeta*={BETA_STAR}")
print(f"Theory: n10 beta_e~0.062->FConn, n11 beta_e~0.042->Fiedler")

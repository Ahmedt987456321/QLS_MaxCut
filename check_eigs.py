from src.graph import load_gset
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

for name in ['G11', 'G12', 'G13']:
    G = load_gset(f'data/gset/{name}.txt')
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        s = np.sign(w) if w != 0 else 1.0
        wa = abs(w)
        i, j = idx[u], idx[v]
        rows += [i, j]; cols += [j, i]; vals += [-s * wa, -s * wa]
        deg[i] += wa; deg[j] += wa
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    ev = sorted(spla.eigsh(L, k=4, which='SM', return_eigenvectors=False, tol=1e-7, maxiter=8000))
    print(f'{name}: eigs={[f"{e:.4f}" for e in ev]}  gap(2-1)={ev[1]-ev[0]:.4f}')

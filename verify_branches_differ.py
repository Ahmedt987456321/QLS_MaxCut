import numpy as np
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls
import scipy.sparse as sp, scipy.sparse.linalg as spla

G = load_gset("data/gset/G11.txt"); nodes = list(G.nodes())
idx = {v:i for i,v in enumerate(nodes)}
rng = np.random.default_rng(0)
x = random_cut(G, rng); gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc)

def fied(signed):
    rows, cols, vals = [], [], []; deg = np.zeros(len(nodes))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        if signed:
            ew = abs(d.get("weight", 1.0)) * (1.0 if x[u] != x[v] else -1.0)
        else:
            ew = abs(d.get("weight", 1.0))
        rows += [i, j]; cols += [j, i]; vals += [-ew, -ew]
        deg[i] += abs(ew); deg[j] += abs(ew)
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes),)*2)
    _, vecs = spla.eigsh(L, k=2, which="SM", tol=1e-3, maxiter=1000)
    return vecs[:, 1]

fs, fu = fied(True), fied(False)
if np.dot(fs, fu) < 0: fu = -fu          # eigenvectors defined up to sign

corr = np.corrcoef(fs, fu)[0, 1]
mean_diff = np.abs(fs - fu).mean()
identical = np.allclose(fs, fu, atol=1e-6)
rank_fs = np.argsort(np.argsort(np.abs(fs)))
rank_fu = np.argsort(np.argsort(np.abs(fu)))
rank_corr = np.corrcoef(rank_fs, rank_fu)[0, 1]

# how different are the edge weightings themselves?
n_unsat = sum(1 for u, v in G.edges() if x[u] == x[v])
print(f"assignment: {n_unsat}/{G.number_of_edges()} non-cut edges "
      f"(these flip sign between the two operators)")
print(f"signed vs unsigned Fiedler vector:")
print(f"  vector correlation = {corr:+.4f}")
print(f"  mean |difference|  = {mean_diff:.4f}")
print(f"  identical?         = {identical}")
print(f"  rank correlation   = {rank_corr:+.4f}  (this is what the selector scores on)")

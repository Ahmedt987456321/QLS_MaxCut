import numpy as np
import networkx as nx
import scipy.sparse as sp, scipy.sparse.linalg as spla
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K, BUDGET = 400, 30
G = load_gset("data/gset/G11.txt"); BKS = 564
NODES = list(G.nodes()); IDX = {v: i for i, v in enumerate(NODES)}

def fiedler_score(G, gc, x):
    rows, cols, vals = [], [], []; deg = np.zeros(len(NODES))
    for u, v, d in G.edges(data=True):
        i, j = IDX[u], IDX[v]; ew = abs(d.get("weight", 1.0))
        rows += [i, j]; cols += [j, i]; vals += [-ew, -ew]; deg[i] += ew; deg[j] += ew
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(len(NODES),)*2)
    _, vecs = spla.eigsh(L, k=2, which="SM", tol=1e-3, maxiter=1000)
    f = vecs[:, 1]
    return {v: 1.0/(0.005 + abs(f[IDX[v]])) for v in NODES}

def gain_score(G, gc, x):
    return {v: 1.0/(1.0 + abs(gc.gain[v])) for v in NODES}

def grow(G, score, k):
    nodes = list(G.nodes())
    seed = max(nodes, key=lambda v: score[v]); S = [seed]
    frontier = set(G.neighbors(seed))
    while len(S) < k and frontier:
        b = max(frontier, key=lambda v: score[v]); S.append(b)
        for u in G.neighbors(b):
            if u not in S: frontier.add(u)
        frontier.discard(b)
    if len(S) < k:
        rest = sorted([v for v in nodes if v not in set(S)], key=lambda v: -score[v])
        S.extend(rest[:k - len(S)])
    return seed, S[:k]

def make_selector(score_fn, log):
    def sel(G, gc, k, pool=None, rng=None, x=None):
        sc = score_fn(G, gc, x)
        seed, S = grow(G, sc, k)
        log["seed"].append(seed); log["supp"].append(frozenset(S))
        return S
    sel.__name__ = score_fn.__name__
    return sel

def analyze(name, log):
    seeds = log["seed"]; sups = log["supp"]
    seed_moves = sum(1 for a, b in zip(seeds[:-1], seeds[1:]) if a != b)
    seed_distinct = len(set(seeds))
    stab = np.mean([len(a & b)/len(a | b) for a, b in zip(sups[:-1], sups[1:]) if a | b])
    dists = []
    for a, b in zip(seeds[:-1], seeds[1:]):
        try:
            dists.append(nx.shortest_path_length(G, a, b))
        except Exception:
            pass
    print("\n=== " + name + " (" + str(len(seeds)) + " calls) ===")
    print("  distinct seed vertices : " + str(seed_distinct) + "/" + str(len(seeds)))
    print("  seed changed each call : " + str(seed_moves) + "/" + str(len(seeds)-1))
    print("  mean dist between seeds: " + format(np.mean(dists) if dists else 0, ".1f") +
          "  (max seen " + str(max(dists) if dists else 0) + ")")
    print("  support stability      : " + format(stab, ".2f"))

for score_fn in [fiedler_score, gain_score]:
    log = {"seed": [], "supp": []}
    sel = make_selector(score_fn, log)
    xb, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=sel, backend=NEAL,
                        k_min=K, k_max=K, n_reads=100, best_known=BKS,
                        seed=0, acceptance="lookahead")
    print("\n[" + score_fn.__name__ + "] final cut = " + format(compute_cut_value(G, xb), ".0f"))
    analyze(score_fn.__name__, log)

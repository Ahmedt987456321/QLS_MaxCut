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

K = 400
CONFIG = {"G11": dict(bks=564, bud=30), "G14": dict(bks=3064, bud=60)}

def build(G):
    nodes = list(G.nodes()); idx = {v:i for i,v in enumerate(nodes)}
    return nodes, idx

def fiedler_score(G, gc, x, nodes, idx):
    rows, cols, vals = [], [], []; deg = np.zeros(len(nodes))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]; ew = abs(d.get("weight", 1.0))
        rows += [i, j]; cols += [j, i]; vals += [-ew, -ew]; deg[i] += ew; deg[j] += ew
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes),)*2)
    _, vecs = spla.eigsh(L, k=2, which="SM", tol=1e-3, maxiter=1000)
    f = vecs[:, 1]
    return {v: 1.0/(0.005 + abs(f[idx[v]])) for v in nodes}

def gain_score(G, gc, x, nodes, idx):
    return {v: 1.0/(1.0 + abs(gc.gain[v])) for v in nodes}

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

def make_selector(score_fn, nodes, idx, log):
    def sel(G, gc, k, pool=None, rng=None, x=None):
        sc = score_fn(G, gc, x, nodes, idx)
        seed, S = grow(G, sc, k)
        log["seed"].append(seed); log["supp"].append(frozenset(S))
        return S
    sel.__name__ = score_fn.__name__
    return sel

for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt"); nodes, idx = build(G)
    # estimate diameter cheaply from a few BFS sources (exact diameter is costly)
    samp = [nodes[i] for i in np.linspace(0, len(nodes)-1, 8).astype(int)]
    diam = max(max(nx.single_source_shortest_path_length(G, s).values()) for s in samp)
    print(f"\n######## {inst}  (approx diameter {diam}) ########")
    for score_fn in [fiedler_score, gain_score]:
        log = {"seed": [], "supp": []}
        sel = make_selector(score_fn, nodes, idx, log)
        xb, _ = adaptive_qls(G, budget_seconds=cfg["bud"], selector=sel, backend=NEAL,
                            k_min=K, k_max=K, n_reads=100, best_known=cfg["bks"],
                            seed=0, acceptance="lookahead")
        seeds = log["seed"]; sups = log["supp"]
        dists = []
        for a, b in zip(seeds[:-1], seeds[1:]):
            try: dists.append(nx.shortest_path_length(G, a, b))
            except Exception: pass
        md = np.mean(dists) if dists else 0
        stab = np.mean([len(a&b)/len(a|b) for a,b in zip(sups[:-1],sups[1:]) if a|b])
        print(f"  [{score_fn.__name__:13s}] final={compute_cut_value(G,xb):.0f}/{cfg['bks']}  "
              f"distinct_seeds={len(set(seeds))}/{len(seeds)}  "
              f"seed_dist={md:.1f} (={md/diam:.2f} x diam)  stability={stab:.2f}")

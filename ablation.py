import numpy as np, networkx as nx
import scipy.sparse as sp, scipy.sparse.linalg as spla
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K, BUDGET, SEEDS = 400, 30, list(range(8))
G = load_gset("data/gset/G11.txt"); BKS = 564
NODES = list(G.nodes()); IDX = {v:i for i,v in enumerate(NODES)}

def fiedler_vec(G, x, signed):
    rows, cols, vals = [], [], []; deg = np.zeros(len(NODES))
    for u, v, d in G.edges(data=True):
        i, j = IDX[u], IDX[v]
        if signed and x is not None:
            ew = abs(d.get("weight",1.0)) * (1.0 if x[u]!=x[v] else -1.0)
        else:
            ew = abs(d.get("weight",1.0))
        rows += [i,j]; cols += [j,i]; vals += [-ew,-ew]
        deg[i] += abs(ew); deg[j] += abs(ew)
    L = sp.diags(deg) + sp.csr_matrix((vals,(rows,cols)), shape=(len(NODES),)*2)
    _, vecs = spla.eigsh(L, k=2, which="SM", tol=1e-3, maxiter=1000)
    return vecs[:,1]

def minmax(d):
    lo, hi = min(d.values()), max(d.values())
    if hi-lo < 1e-12: return {k:0.5 for k in d}
    return {k:(v-lo)/(hi-lo) for k,v in d.items()}

def make_ablation(alpha, signed=False):
    """alpha=1.0 -> pure gain; alpha=0.0 -> pure Fiedler; normalised blend."""
    def sel(G, gc, k, pool=None, rng=None, x=None):
        nodes = list(G.nodes())
        if len(nodes) <= k: return nodes
        gain_raw = {v: 1.0/(1.0+abs(gc.gain[v])) for v in nodes}
        if alpha >= 1.0:
            score = minmax(gain_raw)
        else:
            f = fiedler_vec(G, x, signed)
            fied_raw = {v: 1.0/(0.005+abs(f[IDX[v]])) for v in nodes}
            g, fn = minmax(gain_raw), minmax(fied_raw)
            score = {v: alpha*g[v] + (1-alpha)*fn[v] for v in nodes}
        seed = max(nodes, key=lambda v: score[v])
        S = [seed]; frontier = set(G.neighbors(seed))
        while len(S) < k and frontier:
            best = max(frontier, key=lambda v: score[v]); S.append(best)
            for u in G.neighbors(best):
                if u not in S: frontier.add(u)
            frontier.discard(best)
        if len(S) < k:
            rest = sorted([v for v in nodes if v not in set(S)],
                          key=lambda v:-score[v]); S.extend(rest[:k-len(S)])
        return S[:k]
    sel.__name__ = f"ablate_a{alpha}"
    return sel

def stability(sups):
    js=[len(a&b)/len(a|b) for a,b in zip(sups[:-1],sups[1:]) if (a|b)]
    return float(np.mean(js)) if js else float("nan")

VARIANTS = {"PURE_GAIN (a=1.0)": make_ablation(1.0),
            "PURE_FIEDLER (a=0.0)": make_ablation(0.0, signed=False),
            "BLEND_NORM (a=0.6)": make_ablation(0.6, signed=False)}

agg = {k:[] for k in VARIANTS}
for s in SEEDS:
    line = f"seed {s}: "
    for name, base in VARIANTS.items():
        log = {"supp": []}
        def wrapped(G, gc, k, pool=None, rng=None, x=None, _b=base, _l=log):
            S = _b(G, gc, k, pool=pool, rng=rng, x=x); _l["supp"].append(frozenset(S)); return S
        wrapped.__name__ = base.__name__
        xb, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=wrapped, backend=NEAL,
                            k_min=K, k_max=K, n_reads=100, best_known=BKS,
                            seed=s, acceptance="lookahead")
        fin = compute_cut_value(G, xb); st = stability(log["supp"])
        agg[name].append((fin, st)); line += f"{name.split()[0]}={fin:.0f}/{st:.2f}  "
    print(line)

print("\n=== distribution over seeds ===")
for name in VARIANTS:
    a = np.array(agg[name], float); f = a[:,0]
    print(f"{name:22s}: hit564={int((f>=564).sum())}/{len(f)}  median={np.median(f):.0f}  "
          f"min={f.min():.0f}  stability={a[:,1].mean():.2f}")

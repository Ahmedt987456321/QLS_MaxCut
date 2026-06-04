import numpy as np, networkx as nx
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls, compute_cut_value
from src.selectors import select_fiedler, select_frustrated_connected
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K = 400
CONFIG = {"G11": dict(bks=564, sel=select_fiedler, bud=30),
          "G14": dict(bks=3064, sel=select_frustrated_connected, bud=60)}

def align(xref, x, S, nodes):
    """spin-flip-align x to xref, return (best Jaccard of S vs a disagreement
    component, # components S touches, total # components)."""
    if sum(1 for v in nodes if x[v]==xref[v]) < len(nodes)/2:
        x = {v: 1-x[v] for v in nodes}
    D = [v for v in nodes if x[v] != xref[v]]
    comps = [set(c) for c in nx.connected_components(G.subgraph(D))]
    Sset = set(S)
    if not comps: return 0.0, 0, 0
    best = max(len(Sset & C)/len(Sset | C) for C in comps)
    touched = sum(1 for C in comps if Sset & C)
    return best, touched, len(comps)

def geom(G, S):
    Sset = set(S); k = len(Sset)
    Ein = sum(1 for u,v in G.edges() if u in Sset and v in Sset)
    bnd = sum(1 for u,v in G.edges() if (u in Sset) ^ (v in Sset))
    ncc = nx.number_connected_components(G.subgraph(list(Sset)))
    return Ein, bnd, ncc

for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt"); nodes = list(G.nodes())
    # reference = best solution from a real run (shared target for both selectors)
    xref, _ = adaptive_qls(G, budget_seconds=cfg["bud"], selector=cfg["sel"],
                          backend=NEAL, k_min=K, k_max=K, n_reads=100,
                          best_known=cfg["bks"], seed=0, acceptance="lookahead")
    print(f"\n=== {inst} (ref cut {compute_cut_value(G,xref):.0f}/{cfg['bks']}) ===")

    # incumbents: spread of local optima from random starts (shared by both selectors)
    rng = np.random.default_rng(7)
    incumbents = []
    for s in range(12):
        x = random_cut(G, rng); gc = GainCache()
        x, gc, _, _ = one_flip_ls(G, x, gc)
        incumbents.append(x)

    rows = {"fiedler": [], "fconn": []}
    for x in incumbents:
        gc = GainCache(); gc.update(G, x)            # mirror the loop exactly
        for name, sel in [("fiedler", select_fiedler),
                          ("fconn", select_frustrated_connected)]:
            S = sel(G, gc, K, pool=None, rng=rng, x=x)   # REAL signed operator
            bestJ, touched, ncomp = align(xref, x, S, nodes)
            Ein, bnd, ncc = geom(G, S)
            rows[name].append((bestJ, touched, ncomp, bnd, ncc))

    for name in ("fiedler", "fconn"):
        a = np.array(rows[name], float)
        print(f"  {name:8s}: bestJaccard={a[:,0].mean():.3f}  "
              f"comps_touched={a[:,1].mean():.1f}/{a[:,2].mean():.0f}  "
              f"|boundary|={a[:,3].mean():.0f}  support_CCs={a[:,4].mean():.1f}")

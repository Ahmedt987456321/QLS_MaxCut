import numpy as np, networkx as nx
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls, compute_cut_value
from src.selectors import select_fiedler, select_frustrated_connected
from src.qubo import build_local_qubo, merge_proposal
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K = 400
CONFIG = {"G11": dict(bks=564, sel=select_fiedler, bud=30),
          "G14": dict(bks=3064, sel=select_frustrated_connected, bud=60)}

def boundary(G, S):
    Sset = set(S)
    return sum(1 for u,v in G.edges() if (u in Sset) ^ (v in Sset))

def subqubo_delta(G, x, S):
    """Solve the local QUBO on S, merge, lookahead-descend; return cut improvement."""
    Q = build_local_qubo(G, x, S)
    if not Q or all(abs(v) < 1e-10 for v in Q.values()):
        return 0.0
    x_local = NEAL(Q, S, n_reads=100)
    x_prop = merge_proposal(x, x_local, S)
    gc2 = GainCache(); x_prop, gc2, _, _ = one_flip_ls(G, x_prop, gc2)
    return compute_cut_value(G, x_prop) - compute_cut_value(G, x)

for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt"); nodes = list(G.nodes())
    xref, _ = adaptive_qls(G, budget_seconds=cfg["bud"], selector=cfg["sel"],
                          backend=NEAL, k_min=K, k_max=K, n_reads=100,
                          best_known=cfg["bks"], seed=0, acceptance="lookahead")
    print(f"\n=== {inst} (ref {compute_cut_value(G,xref):.0f}/{cfg['bks']}) ===")
    rng = np.random.default_rng(7)

    # tier 'low' = random local optima; tier 'near' = perturbed-ref local optima
    incs = {"low": [], "near": []}
    for _ in range(8):
        x = random_cut(G, rng); gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc)
        incs["low"].append(x)
        xp = dict(xref)
        for v in rng.choice(nodes, size=len(nodes)//10, replace=False):  # flip ~10%
            xp[v] = 1 - xp[v]
        gc = GainCache(); xp, gc, _, _ = one_flip_ls(G, xp, gc)
        incs["near"].append(xp)

    store = {"fiedler": {"b": [], "d": []}, "fconn": {"b": [], "d": []}}
    for tier in ("low", "near"):
        for name, sel in [("fiedler", select_fiedler), ("fconn", select_frustrated_connected)]:
            bs, ds = [], []
            for x in incs[tier]:
                gc = GainCache(); gc.update(G, x)
                S = sel(G, gc, K, pool=None, rng=rng, x=x)
                b = boundary(G, S); d = subqubo_delta(G, x, S)
                bs.append(b); ds.append(d)
                store[name]["b"].append(b); store[name]["d"].append(d)
            bs, ds = np.array(bs), np.array(ds)
            print(f"  [{tier:4s}] {name:8s}: |bnd|={bs.mean():5.0f}  "
                  f"delta={ds.mean():6.1f}  improving={np.mean(ds>0):.2f}")

    print("  --- boundary->improvement coupling (pooled) ---")
    for name in ("fiedler", "fconn"):
        b = np.array(store[name]["b"]); d = np.array(store[name]["d"])
        r = np.corrcoef(b, d)[0,1] if b.std()>0 and d.std()>0 else float("nan")
        print(f"    {name:8s}: corr(|bnd|, delta) = {r:+.2f}  "
              f"(negative = lower boundary -> bigger gain)")

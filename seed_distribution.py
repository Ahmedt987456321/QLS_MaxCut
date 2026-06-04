import numpy as np
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.selectors import select_fiedler, select_frustrated_connected
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K, BUDGET, SEEDS = 400, 30, list(range(10))
G = load_gset("data/gset/G11.txt"); BKS = 564
NODES = list(G.nodes()); IDX = {v:i for i,v in enumerate(NODES)}

def make_logger(real_sel):
    log = {"cut": [], "supp": []}
    def wrapped(G, gc, k, pool=None, rng=None, x=None):
        S = real_sel(G, gc, k, pool=pool, rng=rng, x=x)
        if x is not None: log["cut"].append(compute_cut_value(G, x))
        log["supp"].append(frozenset(S))
        return S
    wrapped.__name__ = real_sel.__name__
    return wrapped, log

def stability(sups):
    js = [len(a&b)/len(a|b) for a,b in zip(sups[:-1],sups[1:]) if (a|b)]
    return float(np.mean(js)) if js else float("nan")

def plateau_call(traj):
    if len(traj)==0: return -1
    run = np.maximum.accumulate(traj)
    return int(np.argmax(run == run[-1]))   # first call reaching final plateau

agg = {"fiedler": [], "fconn": []}
print(f"{'seed':>5} | {'fiedler final/plat/stab':>26} | {'fconn final/plat/stab':>26}")
print("-"*64)
for s in SEEDS:
    row = {}
    for name, real in [("fiedler", select_fiedler), ("fconn", select_frustrated_connected)]:
        wrapped, log = make_logger(real)
        x_best, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=wrapped, backend=NEAL,
                                k_min=K, k_max=K, n_reads=100, best_known=BKS,
                                seed=s, acceptance="lookahead")
        fin = compute_cut_value(G, x_best)
        traj = np.array(log["cut"]); pc = plateau_call(traj); st = stability(log["supp"])
        agg[name].append((fin, pc, st, len(log["supp"])))
        row[name] = (fin, pc, st)
    f, c = row["fiedler"], row["fconn"]
    print(f"{s:>5} | {f'{f[0]:.0f} / call {f[1]} / {f[2]:.2f}':>26} | "
          f"{f'{c[0]:.0f} / call {c[1]} / {c[2]:.2f}':>26}")

print("\n=== distribution over seeds ===")
for name in ("fiedler","fconn"):
    a = np.array(agg[name], float)
    fins = a[:,0]
    print(f"{name:8s}: final cut  median={np.median(fins):.0f}  "
          f"min={fins.min():.0f}  max={fins.max():.0f}  hit564={int((fins>=564).sum())}/{len(fins)}")
    print(f"{'':8s}  plateau call median={np.median(a[:,1]):.0f}   "
          f"support stability mean={a[:,2].mean():.2f}   calls/run median={np.median(a[:,3]):.0f}")

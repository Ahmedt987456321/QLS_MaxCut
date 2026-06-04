import numpy as np, networkx as nx
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.selectors import select_fiedler, select_frustrated_connected
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K, BUDGET, SEED = 400, 30, 0
G = load_gset("data/gset/G11.txt")
BKS = 564
NODES = list(G.nodes())
IDX = {v: i for i, v in enumerate(NODES)}      # map node id -> 0-based position

def make_logger(real_sel):
    log = {"cut": [], "supp": []}
    def wrapped(G, gc, k, pool=None, rng=None, x=None):
        S = real_sel(G, gc, k, pool=pool, rng=rng, x=x)
        if x is not None:
            log["cut"].append(compute_cut_value(G, x))
        log["supp"].append(frozenset(S))
        return S
    wrapped.__name__ = real_sel.__name__
    return wrapped, log

def coverage_entropy(supports):
    c = np.zeros(len(NODES))
    for S in supports:
        for v in S: c[IDX[v]] += 1          # FIX: index by position, not raw id
    tot = c.sum()
    if tot == 0: return 0.0
    p = c[c > 0] / tot
    return -(p*np.log(p)).sum() / np.log(len(NODES))

def stability(supports):
    js = [len(a & b)/len(a | b) for a,b in zip(supports[:-1], supports[1:]) if (a|b)]
    return float(np.mean(js)) if js else float("nan")

results = {}
for name, real in [("fiedler", select_fiedler), ("fconn", select_frustrated_connected)]:
    wrapped, log = make_logger(real)
    x_best, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=wrapped, backend=NEAL,
                            k_min=K, k_max=K, n_reads=100, best_known=BKS,
                            seed=SEED, acceptance="lookahead")
    traj = np.array(log["cut"])
    running = np.maximum.accumulate(traj) if len(traj) else traj
    results[name] = dict(final=compute_cut_value(G, x_best), calls=len(log["supp"]),
                        running=running,
                        H=coverage_entropy(log["supp"]),
                        stab=stability(log["supp"]),
                        nverts=len(set().union(*log["supp"])) if log["supp"] else 0)

print(f"{'metric':<22}{'fiedler':>12}{'fconn':>12}")
for key, lab in [("final","final cut"), ("calls","# QLS calls"),
                 ("H","coverage entropy"), ("stab","support stability"),
                 ("nverts","distinct verts used")]:
    f, c = results["fiedler"][key], results["fconn"][key]
    print(f"{lab:<22}{f:>12.3f}{c:>12.3f}" if isinstance(f,float)
          else f"{lab:<22}{f:>12}{c:>12}")

print("\nrunning-best cut at call checkpoints:")
print(f"{'call':>8}{'fiedler':>12}{'fconn':>12}")
rf, rc = results["fiedler"]["running"], results["fconn"]["running"]
for q in [0.1,0.25,0.5,0.75,0.9,1.0]:
    vf = rf[min(int(q*len(rf)), len(rf)-1)] if len(rf) else float("nan")
    vc = rc[min(int(q*len(rc)), len(rc)-1)] if len(rc) else float("nan")
    print(f"{int(q*100):>6}%{vf:>12.0f}{vc:>12.0f}")

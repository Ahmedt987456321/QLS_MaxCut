import numpy as np
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.selectors import select_fiedler
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

K, BUDGET, SEEDS = 400, 30, list(range(8))
G = load_gset("data/gset/G11.txt"); BKS = 564

def stability(sups):
    js = [len(a&b)/len(a|b) for a,b in zip(sups[:-1],sups[1:]) if (a|b)]
    return float(np.mean(js)) if js else float("nan")

# signed = forward real x (assignment-dependent); unsigned = force x=None (fixed strip)
def make_variant(signed):
    def factory():
        log = {"supp": []}
        def wrapped(G, gc, k, pool=None, rng=None, x=None):
            S = select_fiedler(G, gc, k, pool=pool, rng=rng,
                               x=(x if signed else None))
            log["supp"].append(frozenset(S))
            return S
        wrapped.__name__ = "fiedler_signed" if signed else "fiedler_unsigned"
        return wrapped, log
    return factory

variants = {"fiedler_SIGNED": make_variant(True),
            "fiedler_UNSIGNED": make_variant(False)}

agg = {k: [] for k in variants}
print(f"{'seed':>5} | {'SIGNED final/stab':>20} | {'UNSIGNED final/stab':>20}")
print("-"*52)
for s in SEEDS:
    row = {}
    for name, factory in variants.items():
        wrapped, log = factory()
        x_best, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=wrapped, backend=NEAL,
                                k_min=K, k_max=K, n_reads=100, best_known=BKS,
                                seed=s, acceptance="lookahead")
        fin = compute_cut_value(G, x_best); st = stability(log["supp"])
        agg[name].append((fin, st)); row[name] = (fin, st)
    a, b = row["fiedler_SIGNED"], row["fiedler_UNSIGNED"]
    print(f"{s:>5} | {f'{a[0]:.0f} / {a[1]:.2f}':>20} | {f'{b[0]:.0f} / {b[1]:.2f}':>20}")

print("\n=== distribution over seeds ===")
for name in variants:
    arr = np.array(agg[name], float); fins = arr[:,0]
    print(f"{name:16s}: hit564={int((fins>=564).sum())}/{len(fins)}  "
          f"median={np.median(fins):.0f}  min={fins.min():.0f}  "
          f"support_stability_mean={arr[:,1].mean():.2f}")
print("\n(reference from last run: FConn = 0/10 at 564, stability 0.80)")

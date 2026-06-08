import numpy as np, networkx as nx
from scipy.stats import mannwhitneyu
from src.adaptive_qls import adaptive_qls
from src.local_search import compute_cut_value
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
neal = get_backend("neal")

K = 400; SEEDS = list(range(6)); BUD = 25
SIGN_SEEDS = [42, 7, 123]          # disorder PATTERNS at each density

base = nx.grid_2d_graph(20, 40, periodic=True)
base = nx.convert_node_labels_to_integers(base)
degs = [d for _, d in base.degree()]
assert base.number_of_nodes()==800 and base.number_of_edges()==1600
assert min(degs)==4 and max(degs)==4
lam2 = nx.algebraic_connectivity(base, method="lanczos")
print(f"torus OK: n=800 m=1600 deg=4 lambda2={lam2:.4f} (PINNED)\n")

def make_signed(frac_neg, sign_seed):
    G = base.copy()
    rng = np.random.default_rng(sign_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = -1 if rng.random() < frac_neg else 1
    return G

def run(G, sel):
    cuts, stabs = [], []
    for s in SEEDS:
        log = []
        def w(G, gc, kk, pool=None, rng=None, x=None):
            S = sel(G, gc, kk, pool=pool, rng=rng, x=x); log.append(frozenset(S)); return S
        w.__name__ = sel.__name__
        xb, _ = adaptive_qls(G, budget_seconds=BUD, selector=w, backend=neal,
                             k_min=K, k_max=K, n_reads=100, best_known=None,
                             seed=s, acceptance="lookahead")
        cuts.append(compute_cut_value(G, xb))
        js = [len(a&b)/len(a|b) for a,b in zip(log[:-1], log[1:]) if a|b]
        stabs.append(np.mean(js) if js else float("nan"))
    return np.array(cuts, float), np.array(stabs, float)

PS = [0.0, 0.05, 0.08, 0.103, 0.15, 0.25, 0.50]
print(f"{'frac_neg':>9}{'pattern':>8}{'F_mob':>7}{'winner':>9}{'p':>8}   (mob across patterns)")
print("-"*64)
for p in PS:
    fmobs, winners = [], []
    rowtxt = []
    for ss in SIGN_SEEDS:
        G = make_signed(p, sign_seed=ss)
        fc, fs = run(G, select_fiedler)
        gc_, gs = run(G, select_frustrated_connected)
        fmed, cmed = float(np.median(fc)), float(np.median(gc_))
        if np.array_equal(fc, gc_):
            pv = 1.0
        else:
            try:
                _, pv = mannwhitneyu(fc, gc_, alternative="two-sided")
            except ValueError:
                pv = 1.0
        win = ("Fiedler" if fmed > cmed else "FConn") if pv < 0.05 else "tie"
        fmobs.append(fs.mean()); winners.append(win)
        rowtxt.append(f"{p:>9.3f}{ss:>8}{fs.mean():>7.2f}{win:>9}"
                      f"{('<1e-4' if pv<1e-4 else format(pv,'.3f')):>8}")
    for r in rowtxt: print(r)
    fm = np.array(fmobs)
    agree = len(set(winners))==1
    print(f"  -> p={p:.3f}: mobility {fm.mean():.2f} +/- {fm.std():.3f}  |  "
          f"winners {winners}  {'(CONSISTENT)' if agree else '(VARIES BY PATTERN)'}")
    print()

print(f"fallback_count: {getattr(select_fiedler,'_fallback_count',0)}")
print("READ: (1) does winner-flip reproduce across patterns?  "
      "(2) is mobility std ~0 at every p (pinned by structure)?")

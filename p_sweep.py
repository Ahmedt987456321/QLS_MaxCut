import numpy as np, networkx as nx
from scipy.stats import mannwhitneyu
from src.adaptive_qls import adaptive_qls
from src.local_search import compute_cut_value
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
neal = get_backend("neal")

K = 400; SEEDS = list(range(8)); BUD = 25

base = nx.grid_2d_graph(20, 40, periodic=True)
base = nx.convert_node_labels_to_integers(base)

degs = [d for _, d in base.degree()]
assert base.number_of_nodes() == 800, "n != 800"
assert base.number_of_edges() == 1600, f"edges {base.number_of_edges()} != 1600"
assert min(degs) == 4 and max(degs) == 4, f"not 4-regular: {min(degs)}-{max(degs)}"
lam2 = nx.algebraic_connectivity(base, method="lanczos")
print(f"torus OK: n=800 m=1600 deg=4 lambda2={lam2:.4f} (PINNED across all p)\n")

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

PS = [0.0, 0.02, 0.05, 0.08, 0.103, 0.13, 0.16, 0.20, 0.30, 0.50]
print(f"{'frac_neg':>9}{'F_mob':>8}{'C_mob':>8}{'F_med':>8}{'C_med':>8}{'winner':>10}{'p':>9}")
print("-"*60)
for p in PS:
    G = make_signed(p, sign_seed=42)
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
    if pv < 0.05:
        win = "Fiedler" if fmed > cmed else "FConn"
    else:
        win = "tie"
    tag = "<1e-4" if pv < 1e-4 else format(pv, ".3f")
    print(f"{p:>9.3f}{fs.mean():>8.2f}{gs.mean():>8.2f}{fmed:>8.0f}{cmed:>8.0f}{win:>10}{tag:>9}")

print(f"\nfallback_count: {getattr(select_fiedler,'_fallback_count',0)}")
print("READ: does winner flip + Fiedler mobility move as frac_neg rises, while lambda2 PINNED?")

import numpy as np, networkx as nx
from scipy.stats import mannwhitneyu
from src.adaptive_qls import adaptive_qls
from src.local_search import compute_cut_value
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
neal = get_backend("neal")

K = 400; SEEDS = list(range(6)); BUD = 25
SIGN_SEEDS = [42, 7, 123]          # 3 disorder patterns per density

# 30x100 torus = the G49 topology (n=3000, lambda2=0.00395, which TIED in the full sweep)
base = nx.grid_2d_graph(30, 100, periodic=True)
base = nx.convert_node_labels_to_integers(base)
degs = [d for _, d in base.degree()]
assert base.number_of_nodes()==3000 and base.number_of_edges()==6000
assert min(degs)==4 and max(degs)==4
lam2 = nx.algebraic_connectivity(base, method="lanczos")
print(f"30x100 torus OK: n=3000 m=6000 deg=4 lambda2={lam2:.5f} (G49 geometry, which TIED)\n")

def make_signed(frac_neg, sign_seed):
    G = base.copy()
    rng = np.random.default_rng(sign_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = -1 if rng.random() < frac_neg else 1
    return G

def run(G, sel):
    cuts = []
    for s in SEEDS:
        xb, _ = adaptive_qls(G, budget_seconds=BUD, selector=sel, backend=neal,
                             k_min=K, k_max=K, n_reads=100, best_known=None,
                             seed=s, acceptance="lookahead")
        cuts.append(compute_cut_value(G, xb))
    return np.array(cuts, float)

PS = [0.0, 0.05, 0.10, 0.25, 0.50]
print(f"{'frac_neg':>9}{'pattern':>8}{'F_med':>8}{'C_med':>8}{'winner':>9}{'p':>8}")
print("-"*52)
for p in PS:
    winners = []
    for ss in SIGN_SEEDS:
        G = make_signed(p, ss)
        fc = run(G, select_fiedler); gc_ = run(G, select_frustrated_connected)
        fmed, cmed = float(np.median(fc)), float(np.median(gc_))
        if np.array_equal(fc, gc_):
            pv = 1.0
        else:
            try:
                _, pv = mannwhitneyu(fc, gc_, alternative="two-sided")
            except ValueError:
                pv = 1.0
        win = ("Fiedler" if fmed > cmed else "FConn") if pv < 0.05 else "tie"
        winners.append(win)
        print(f"{p:>9.3f}{ss:>8}{fmed:>8.0f}{cmed:>8.0f}{win:>9}{('<1e-4' if pv<1e-4 else format(pv,'.3f')):>8}")
    from collections import Counter
    print(f"  -> p={p:.2f}: {dict(Counter(winners))}")

print(f"\nfallback_count: {getattr(select_fiedler,'_fallback_count',0)}")
print("READ: does the 30x100 (n=3000, G49 geometry) ever give Fiedler-wins as disorder rises,")
print("      or does it TIE regardless? Tie-regardless => disorder is NOT sufficient at this scale.")

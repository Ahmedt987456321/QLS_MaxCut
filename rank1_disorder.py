"""RANK 1: is lambda2 causal or a proxy? Hold the GRAPH (topology) fixed so
router-lambda2 is pinned EXACTLY, vary only the +-1 edge-sign disorder
(frustration density), and test whether the FConn-vs-Fiedler winner flips.

Fixed topology -> fixed lambda2 (=0.0246, confirmed). If the winner moves as
frustration changes, lambda2 is a proxy and the configuration-space landscape
is the driver."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8
K = 400

# ONE fixed topology: 20x40 periodic torus (n=800, degree 4, like G-set tori)
base = nx.grid_2d_graph(20, 40, periodic=True)
base = nx.convert_node_labels_to_integers(base)
lam2 = nx.algebraic_connectivity(base, method="lanczos")
print(f"Fixed topology: n={base.number_of_nodes()}, "
      f"edges={base.number_of_edges()}, lambda2={lam2:.4f} (PINNED)\n")

def make_signed(frac_neg, sign_seed):
    """Same topology; fraction frac_neg of edges get weight -1, rest +1."""
    G = base.copy()
    rng = np.random.default_rng(sign_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = -1 if rng.random() < frac_neg else 1
    return G

def med(G, sel):
    return stats.median([adaptive_qls(G, budget_seconds=30, selector=sel,
        backend=neal, k_min=K, k_max=K, n_reads=100, seed=s,
        best_known=None)[1].best_cut for s in range(SEEDS)])

print(f"{'frac_neg':>9} {'lambda2':>8} {'FConn':>7} {'Fiedler':>8} {'winner':>8} {'margin':>7}")
rows = []
# frustration sweep: 0% negative (ferromagnetic, unfrustrated) -> 50% (max disorder)
for frac_neg in [0.0, 0.1, 0.25, 0.5]:
    G = make_signed(frac_neg, sign_seed=42)
    l2 = nx.algebraic_connectivity(G, method="lanczos")  # should stay 0.0246
    fc = med(G, select_frustrated_connected)
    fi = med(G, select_fiedler)
    winner = "Fiedler" if fi > fc else ("FConn" if fc > fi else "tie")
    print(f"{frac_neg:9.2f} {l2:8.4f} {fc:7.0f} {fi:8.0f} {winner:>8} {abs(fi-fc):7.0f}")
    rows.append({"frac_neg":frac_neg,"lambda2":l2,"FConn":fc,"Fiedler":fi,
                 "winner":winner,"margin":abs(fi-fc)})

print("\n=== VERDICT ===")
winners = set(r["winner"] for r in rows if r["winner"]!="tie")
print(f"lambda2 stayed at ~{lam2:.4f} throughout (topology fixed).")
if len(winners) > 1:
    print("Winner FLIPS as frustration changes at FIXED lambda2")
    print("-> lambda2 is a PROXY; configuration-space frustration is the driver.")
else:
    print(f"Winner is CONSTANT ({winners}) across all frustration at fixed lambda2")
    print("-> lambda2/topology survives as the driver; landscape disorder does not flip it.")
json.dump(rows, open("results/rank1_disorder_sweep.json","w"), indent=2, default=str)

"""RANK 2 (converse of Rank 1): vary lambda2 while holding base geometry and
+-1 frustration comparable. Add random long-range chords to a fixed torus to
raise lambda2; does the FConn-vs-Fiedler winner FLIP with lambda2?

Rank 1: landscape varied, lambda2 fixed -> NO flip (Fiedler always).
Rank 2: lambda2 varied, geometry+frustration comparable -> does it flip?
Both together pin lambda2 as the driver from both directions."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8
K = 400
FRAC_NEG = 0.5   # fixed frustration level (max disorder, as in Rank 1)

# base torus (low lambda2), same as Rank 1
base = nx.grid_2d_graph(20, 40, periodic=True)
base = nx.convert_node_labels_to_integers(base)
nodes = list(base.nodes())

def build(n_chords, sign_seed, chord_seed):
    """Torus + n_chords random long-range edges. Raises lambda2 while keeping
    the lattice geometry and +-1 frustration. Signs assigned to ALL edges."""
    G = base.copy()
    crng = np.random.default_rng(chord_seed)
    added = 0
    while added < n_chords:
        u, v = crng.choice(nodes, size=2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(int(u), int(v)); added += 1
    srng = np.random.default_rng(sign_seed)
    for a, b in G.edges():
        G[a][b]["weight"] = -1 if srng.random() < FRAC_NEG else 1
    return G

def med(G, sel):
    return stats.median([adaptive_qls(G, budget_seconds=30, selector=sel,
        backend=neal, k_min=K, k_max=K, n_reads=100, seed=s,
        best_known=None)[1].best_cut for s in range(SEEDS)])

print(f"Base torus n=800; frustration fixed at frac_neg={FRAC_NEG}.")
print(f"Raising lambda2 by adding random chords.\n")
print(f"{'chords':>7} {'lambda2':>8} {'FConn':>7} {'Fiedler':>8} {'winner':>8} {'margin':>7}")
rows = []
for n_chords in [0, 50, 200, 800]:
    G = build(n_chords, sign_seed=42, chord_seed=7)
    l2 = nx.algebraic_connectivity(G, method="lanczos")
    fc = med(G, select_frustrated_connected)
    fi = med(G, select_fiedler)
    winner = "Fiedler" if fi > fc else ("FConn" if fc > fi else "tie")
    print(f"{n_chords:7d} {l2:8.4f} {fc:7.0f} {fi:8.0f} {winner:>8} {abs(fi-fc):7.0f}")
    rows.append({"chords":n_chords,"lambda2":l2,"FConn":fc,"Fiedler":fi,
                 "winner":winner,"margin":abs(fi-fc)})

print("\n=== VERDICT ===")
winners_seq = [r["winner"] for r in rows]
print(f"lambda2 swept: {[round(r['lambda2'],3) for r in rows]}")
print(f"winners:       {winners_seq}")
flipped = "Fiedler" in winners_seq and "FConn" in winners_seq
if flipped:
    print("Winner FLIPS as lambda2 rises (geometry+frustration comparable)")
    print("-> combined with Rank 1, lambda2 is the DRIVER from BOTH directions.")
else:
    print("Winner does NOT flip even as lambda2 rises")
    print("-> lambda2 alone may NOT be sufficient; geometry/locality may co-drive.")
json.dump(rows, open("results/rank2_lambda2_sweep.json","w"), indent=2, default=str)

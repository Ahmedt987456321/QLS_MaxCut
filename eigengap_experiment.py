"""EIGENGAP experiment: matched lambda2, varied lambda3-lambda2 gap.
Does the FConn-vs-Fiedler winner track the GAP (bottleneck isolation) or
lambda2's magnitude?

Large gap  = clean single bottleneck (2 lobes, thin cut) -> bimodal Fiedler.
Small gap  = no clean bottleneck (many lobes) -> lambda2~lambda3, ambiguous.
Both tuned toward a comparable lambda2; we REPORT the achieved spectra so the
construction is verified, not assumed."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8
K = 400
FRAC_NEG = 0.5

def laplacian_spectrum_bottom(G, kk=4):
    """Return the kk smallest Laplacian eigenvalues (unweighted topology)."""
    L = nx.laplacian_matrix(G).astype(float).todense()
    ev = np.linalg.eigvalsh(np.asarray(L))
    return np.sort(ev)[:kk]

def two_lobe(lobe_n, n_bridge, intra_p, seed):
    """Two random lobes joined by n_bridge edges -> isolated small lambda2,
    large gap (clean bottleneck)."""
    rng = np.random.default_rng(seed)
    A = nx.gnp_random_graph(lobe_n, intra_p, seed=seed)
    B = nx.gnp_random_graph(lobe_n, intra_p, seed=seed+1)
    G = nx.disjoint_union(A, B)
    aN = list(range(lobe_n)); bN = list(range(lobe_n, 2*lobe_n))
    for _ in range(n_bridge):
        G.add_edge(int(rng.choice(aN)), int(rng.choice(bN)))
    return G

def multi_lobe(lobe_n, n_lobes, n_bridge, intra_p, seed):
    """n_lobes lobes in a ring of thin bridges -> several small eigenvalues,
    small lambda3-lambda2 gap (no single clean bottleneck)."""
    rng = np.random.default_rng(seed)
    parts = [nx.gnp_random_graph(lobe_n, intra_p, seed=seed+i) for i in range(n_lobes)]
    G = parts[0]
    offsets = [0]
    for p in parts[1:]:
        off = G.number_of_nodes()
        offsets.append(off)
        G = nx.disjoint_union(G, p)
    for i in range(n_lobes):
        a0 = offsets[i]; b0 = offsets[(i+1) % n_lobes]
        aN = list(range(a0, a0+lobe_n)); bN = list(range(b0, b0+lobe_n))
        for _ in range(n_bridge):
            G.add_edge(int(rng.choice(aN)), int(rng.choice(bN)))
    return G

def signed(G, seed):
    G = G.copy(); rng = np.random.default_rng(seed)
    for u,v in G.edges():
        G[u][v]["weight"] = -1 if rng.random() < FRAC_NEG else 1
    return G

def med(G, sel):
    return stats.median([adaptive_qls(G, budget_seconds=30, selector=sel,
        backend=neal, k_min=K, k_max=K, n_reads=100, seed=s,
        best_known=None)[1].best_cut for s in range(SEEDS)])

# build families; tune intra_p / bridges so lambda2 is comparable across the two
print("Constructing families (reporting achieved spectra)...\n")
specs = [
    ("large_gap_2lobe", two_lobe(200, 2, 0.05, seed=10)),
    ("small_gap_4lobe", multi_lobe(100, 4, 1, 0.05, seed=10)),
    ("large_gap_2lobe_b", two_lobe(200, 3, 0.05, seed=20)),
    ("small_gap_5lobe", multi_lobe(80, 5, 1, 0.05, seed=20)),
]

print(f"{'family':18} {'n':>5} {'lam2':>8} {'lam3':>8} {'gap=lam3-lam2':>14} "
      f"{'FConn':>7} {'Fiedler':>8} {'winner':>8}")
rows = []
for name, G in specs:
    G = nx.convert_node_labels_to_integers(G)
    bot = laplacian_spectrum_bottom(G, 4)
    lam2, lam3 = float(bot[1]), float(bot[2])
    gap = lam3 - lam2
    Gs = signed(G, seed=42)
    fc = med(Gs, select_frustrated_connected)
    fi = med(Gs, select_fiedler)
    winner = "Fiedler" if fi > fc else ("FConn" if fc > fi else "tie")
    print(f"{name:18} {G.number_of_nodes():5d} {lam2:8.4f} {lam3:8.4f} "
          f"{gap:14.4f} {fc:7.0f} {fi:8.0f} {winner:>8}")
    rows.append({"family":name,"n":G.number_of_nodes(),"lam2":lam2,"lam3":lam3,
                 "gap":gap,"FConn":fc,"Fiedler":fi,"winner":winner})

print("\n=== READ ===")
print("Compare families with SIMILAR lambda2 but DIFFERENT gap:")
print(" - if Fiedler wins on LARGE gap and loses on SMALL gap (at matched lambda2)")
print("   -> bottleneck ISOLATION (the gap) is the mechanism, not lambda2 magnitude.")
print(" - if winner tracks lambda2 regardless of gap -> lambda2 magnitude survives.")
print("NOTE: verify lambda2 is actually comparable across the contrasted pairs;")
print("if not, the construction needs retuning before drawing conclusions.")
json.dump(rows, open("results/eigengap_experiment.json","w"), indent=2, default=str)

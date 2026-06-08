import networkx as nx, statistics, json, numpy as np
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler, select_beta_routed
from src.backends import get_backend
from scipy.stats import mannwhitneyu

def make_torus(rows, cols, p_neg, seed):
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(rows, cols, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        w = -1 if rng.random() < p_neg else 1
        G[u][v]["weight"] = w
    return G

def measure_beta_e(G):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    G_unw = nx.Graph()
    G_unw.add_nodes_from(G.nodes())
    G_unw.add_edges_from(G.edges())
    try:
        v2 = nx.fiedler_vector(G_unw, method="lanczos")
        n = G_unw.number_of_nodes()
        m = G_unw.number_of_edges()
        half = n // 2
        nodes = list(G_unw.nodes())
        S = set(nodes[i] for i in np.argsort(v2)[-half:])
        cut = sum(1 for u, v in G_unw.edges() if (u in S) != (v in S))
        beta_e = cut / m
        lam2 = nx.algebraic_connectivity(G_unw, method="lanczos")
        return beta_e, lam2
    except Exception as e:
        return -1, -1

neal = get_backend("neal")
TRIALS = 10
BUDGET = 20
ROWS, COLS = 100, 8
k = 400
SEEDS = [42, 123, 999]

p_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

all_results = {}
for p in p_values:
    p_results = []
    for seed in SEEDS:
        G = make_torus(ROWS, COLS, p_neg=p, seed=seed)
        beta_e, lam2 = measure_beta_e(G)
        cuts = {}
        for sel_name, sel in [("FConn", select_frustrated_connected),
                               ("Fiedler", select_fiedler)]:
            c = [adaptive_qls(G, BUDGET, selector=sel, backend=neal,
                              k_min=k, k_max=k, n_reads=100,
                              seed=s*1000, best_known=None)[1].best_cut
                 for s in range(TRIALS)]
            cuts[sel_name] = c
        fconn_med = statistics.median(cuts["FConn"])
        fiedler_med = statistics.median(cuts["Fiedler"])
        _, p_val = mannwhitneyu(cuts["FConn"], cuts["Fiedler"], alternative="two-sided")
        winner = "Fiedler" if fiedler_med > fconn_med else "FConn"
        if p_val >= 0.05:
            winner = "tie"
        print("p=" + str(p) + " seed=" + str(seed) +
              " beta_e=" + str(round(beta_e, 4)) +
              " lam2=" + str(round(lam2, 4)) +
              " FConn=" + str(fconn_med) +
              " Fiedler=" + str(fiedler_med) +
              " winner=" + winner +
              " p_val=" + str(round(p_val, 4)))
        p_results.append({
            "p_neg": p, "seed": seed, "beta_e": beta_e, "lam2": lam2,
            "FConn": cuts["FConn"], "Fiedler": cuts["Fiedler"],
            "winner": winner, "p_val": p_val
        })
    all_results[str(p)] = p_results
    json.dump(all_results, open("results/psweep.json", "w"), indent=2, default=str)

print("Done. Results in results/psweep.json")
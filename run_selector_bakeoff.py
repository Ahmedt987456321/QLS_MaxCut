import networkx as nx, statistics, json, numpy as np
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler, select_energy_impact_bfs, select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from scipy.stats import mannwhitneyu

def load_mtx(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    data_lines=[l.strip() for l in lines if not l.startswith("%") and l.strip()]
    for line in data_lines[1:]:
        parts=line.split()
        if len(parts)>=2:
            u,v=int(parts[0]),int(parts[1])
            if u!=v: G.add_edge(u,v)
    return nx.convert_node_labels_to_integers(G)

def add_weights(G, seed=42):
    rng=np.random.default_rng(seed)
    for u,v in G.edges():
        if "weight" not in G[u][v]:
            G[u][v]["weight"]=int(rng.choice([-1,1]))
    return G

neal = get_backend("neal")
TRIALS = 10
BUDGET = 20

instances = [
    ("G11",  load_gset("data/gset/G11.txt"),  400),
    ("G12",  load_gset("data/gset/G12.txt"),  400),
    ("G13",  load_gset("data/gset/G13.txt"),  400),
    ("G32",  load_gset("data/gset/G32.txt"),  640),
    ("G33",  load_gset("data/gset/G33.txt"),  640),
    ("G34",  load_gset("data/gset/G34.txt"),  640),
    ("G48",  load_gset("data/gset/G48.txt"),  640),
    ("G49",  load_gset("data/gset/G49.txt"),  640),
    ("G50",  load_gset("data/gset/G50.txt"),  640),
    ("G1",   load_gset("data/gset/G1.txt"),   160),
    ("G14",  load_gset("data/gset/G14.txt"),  400),
    ("G22",  load_gset("data/gset/G22.txt"),  640),
    ("3elt", add_weights(load_mtx("data/walshaw/3elt/3elt.mtx")), 400),
    ("power",add_weights(load_mtx("data/roadnetwork/power/power.mtx")), 400),
]

selectors = [
    ("FConn",   select_frustrated_connected),
    ("Fiedler", select_fiedler),
    ("EID-BFS", select_energy_impact_bfs),
    ("beta_e",  select_beta_routed),
]

all_results = {}
for name, G, k in instances:
    print("Running " + name + "...")
    cuts = {}
    for sel_name, sel in selectors:
        c = [adaptive_qls(G, BUDGET, selector=sel, backend=neal,
                          k_min=k, k_max=k, n_reads=100,
                          seed=s*1000, best_known=None)[1].best_cut
             for s in range(TRIALS)]
        cuts[sel_name] = c
        print("  " + sel_name + ": median=" + str(statistics.median(c)) + " best=" + str(max(c)))
    route = G.graph.get("_beta_routing_cache", {}).get("route", "?")
    beta_e = G.graph.get("_beta_routing_cache", {}).get("beta_e", -1)
    print("  beta_e=" + str(round(beta_e, 4)) + " route=" + route)
    for sel_name in ["FConn", "Fiedler", "EID-BFS"]:
        _, p = mannwhitneyu(cuts["beta_e"], cuts[sel_name], alternative="two-sided")
        print("  p(beta_e vs " + sel_name + ")=" + str(round(p, 4)))
    print()
    all_results[name] = cuts
    json.dump(all_results, open("results/selector_bakeoff.json","w"), indent=2, default=str)

print("Done. Results in results/selector_bakeoff.json")
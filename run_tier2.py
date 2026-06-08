import networkx as nx, statistics, json, time, numpy as np
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from src.baselines import simulated_annealing, tabu_search
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
TRIALS = 30
BUDGET = 20

instances = [
    ("G12",  load_gset("data/gset/G12.txt"),  400, 100),
    ("G13",  load_gset("data/gset/G13.txt"),  400, 100),
    ("G32",  load_gset("data/gset/G32.txt"),  640, 100),
    ("G33",  load_gset("data/gset/G33.txt"),  640, 100),
    ("G34",  load_gset("data/gset/G34.txt"),  640, 100),
    ("G48",  load_gset("data/gset/G48.txt"),  640, 100),
    ("G49",  load_gset("data/gset/G49.txt"),  640, 100),
    ("G50",  load_gset("data/gset/G50.txt"),  640, 100),
    ("3elt", add_weights(load_mtx("data/walshaw/3elt/3elt.mtx")), 400, 100),
    ("power",add_weights(load_mtx("data/roadnetwork/power/power.mtx")), 400, 100),
]

all_results = {}
for name, G, k, tenure in instances:
    print("Running " + name + "...")
    cuts = {}
    for algo, fn in [
        ("SA",   lambda s,G=G: simulated_annealing(G,BUDGET,seed=s)[1].best_cut),
        ("Tabu", lambda s,G=G,t=tenure: tabu_search(G,BUDGET,tabu_tenure=t,seed=s)[1].best_cut),
        ("AQLS", lambda s,G=G,k=k: adaptive_qls(G,BUDGET,selector=select_beta_routed,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s*1000,best_known=None)[1].best_cut),
    ]:
        cuts[algo] = [fn(s) for s in range(TRIALS)]
        med=statistics.median(cuts[algo]); best=max(cuts[algo])
        print("  " + algo + ": median=" + str(med) + " best=" + str(best))
    _,p1=mannwhitneyu(cuts["SA"],cuts["Tabu"],alternative="two-sided")
    _,p2=mannwhitneyu(cuts["SA"],cuts["AQLS"],alternative="two-sided")
    _,p3=mannwhitneyu(cuts["Tabu"],cuts["AQLS"],alternative="two-sided")
    route=G.graph.get("_beta_routing_cache",{}).get("route","?")
    beta_e=G.graph.get("_beta_routing_cache",{}).get("beta_e",-1)
    print("  beta_e=" + str(round(beta_e,4)) + " route=" + route)
    print("  p(Tabu vs SA)=" + str(round(p1,4)) + " p(AQLS vs SA)=" + str(round(p2,4)) + " p(AQLS vs Tabu)=" + str(round(p3,4)))
    print()
    all_results[name] = {"SA": cuts["SA"], "Tabu": cuts["Tabu"], "AQLS": cuts["AQLS"],
                         "beta_e": beta_e, "route": route}
    json.dump(all_results, open("results/tier2_comparison.json","w"), indent=2, default=str)

print("Done. Results in results/tier2_comparison.json")

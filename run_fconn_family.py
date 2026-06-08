import networkx as nx, statistics, json, numpy as np
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from src.baselines import simulated_annealing, tabu_search
from scipy.stats import mannwhitneyu

def make_weighted(G, seed=42):
    rng=np.random.default_rng(seed)
    for u,v in G.edges():
        if "weight" not in G[u][v]:
            G[u][v]["weight"]=int(rng.choice([-1,1]))
    return G

neal = get_backend("neal")
TRIALS = 30
BUDGET = 20

instances = [
    ("G15", load_gset("data/gset/G15.txt"), 400, 60),
    ("G16", load_gset("data/gset/G16.txt"), 400, 60),
    ("G17", load_gset("data/gset/G17.txt"), 400, 60),
    ("G18", load_gset("data/gset/G18.txt"), 400, 60),
    ("d3_n800", make_weighted(nx.random_regular_graph(3,800,seed=42)), 400, 60),
    ("d4_n800", make_weighted(nx.random_regular_graph(4,800,seed=42)), 400, 60),
    ("d5_n800", make_weighted(nx.random_regular_graph(5,800,seed=42)), 320, 60),
    ("BA_m2", make_weighted(nx.barabasi_albert_graph(800,2,seed=42)), 400, 60),
    ("BA_m3", make_weighted(nx.barabasi_albert_graph(800,3,seed=42)), 400, 60),
    ("WS_p01", make_weighted(nx.watts_strogatz_graph(800,4,0.1,seed=42)), 400, 60),
    ("WS_p05", make_weighted(nx.watts_strogatz_graph(800,4,0.5,seed=42)), 400, 60),
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
    json.dump(all_results, open("results/fconn_family_comparison.json","w"), indent=2, default=str)

print("Done. Results in results/fconn_family_comparison.json")

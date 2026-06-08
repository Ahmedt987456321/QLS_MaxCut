import statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from src.baselines import simulated_annealing, tabu_search
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS = 30
configs = {
    "G11": {"k":400,"budget":30,"bks":564,"tenure":100},
    "G1":  {"k":160,"budget":30,"bks":11624,"tenure":60},
    "G14": {"k":400,"budget":60,"bks":3064,"tenure":100},
    "G22": {"k":640,"budget":30,"bks":13359,"tenure":100},
}
all_results = {}
for name, cfg in configs.items():
    G = load_gset("data/gset/" + name + ".txt")
    k=cfg["k"]; budget=cfg["budget"]; bks=cfg["bks"]; tenure=cfg["tenure"]
    cuts = {}
    for algo, fn in [
        ("SA",   lambda s,G=G,b=budget: simulated_annealing(G,b,seed=s)[1].best_cut),
        ("Tabu", lambda s,G=G,b=budget,t=tenure: tabu_search(G,b,tabu_tenure=t,seed=s)[1].best_cut),
        ("AQLS", lambda s,G=G,b=budget,k=k: adaptive_qls(G,b,selector=select_beta_routed,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s*1000,best_known=None)[1].best_cut),
    ]:
        cuts[algo] = [fn(s) for s in range(TRIALS)]
        med = statistics.median(cuts[algo])
        best = max(cuts[algo])
        hits = sum(1 for c in cuts[algo] if c >= bks)
        print(name + " " + algo + ": median=" + str(med) + " best=" + str(best) + " hits=" + str(hits) + "/" + str(TRIALS))
    _,p1=mannwhitneyu(cuts["SA"],cuts["Tabu"],alternative="two-sided")
    _,p2=mannwhitneyu(cuts["SA"],cuts["AQLS"],alternative="two-sided")
    _,p3=mannwhitneyu(cuts["Tabu"],cuts["AQLS"],alternative="two-sided")
    print("  p(Tabu vs SA)=" + str(round(p1,4)) + " p(AQLS vs SA)=" + str(round(p2,4)) + " p(AQLS vs Tabu)=" + str(round(p3,4)))
    print()
    all_results[name] = cuts
    json.dump(all_results, open("results/comparison_tuned_tabu.json","w"), indent=2, default=str)
print("Done.")

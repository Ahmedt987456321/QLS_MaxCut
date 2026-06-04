import json
import numpy as np
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_fiedler
from src.backends import backend_neal
from src.local_search import compute_cut_value

best_known = {"G11":564,"G12":556,"G13":582,"G32":1410,"G33":1382,
              "G34":1384,"G48":6000,"G49":6000,"G50":5880}
config = {  # (k, budget_seconds, n_trials)
    "G11":(400,30,10),"G12":(400,30,10),"G13":(400,30,10),
    "G32":(640,30,10),"G33":(640,30,10),"G34":(640,30,10),
    "G48":(900,30,10),"G49":(900,30,10),"G50":(900,30,10),
}

results = {}
for name in ["G11","G12","G13","G32","G33","G34","G48","G49","G50"]:
    G = load_gset(f"data/gset/{name}.txt")
    bk = best_known[name]
    k, budget, trials = config[name]
    nodes = sorted(G.nodes())
    cuts, partitions, exact_hits = [], [], 0
    for t in range(trials):
        x_best, m = adaptive_qls(
            G=G, budget_seconds=budget, selector=select_fiedler,
            backend=backend_neal, k_min=k, k_max=k, n_reads=200,
            best_known=bk, acceptance="lookahead", seed=1000+t)
        c = compute_cut_value(G, x_best)
        cuts.append(c)
        # store partition as compact bitstring keyed on sorted nodes
        bits = "".join(str(int(x_best[v])) for v in nodes)
        partitions.append(bits)
        if abs(c - bk) < 1e-9:
            exact_hits += 1
    # canonicalise partitions (a cut and its complement are the same cut)
    def canon(b):
        comp = "".join("1" if ch=="0" else "0" for ch in b)
        return min(b, comp)
    distinct = len(set(canon(b) for b in partitions if abs(cuts[partitions.index(b)]-max(cuts))<1e-9))
    ratio = max(cuts)/bk
    results[name] = {
        "best_cut": max(cuts), "best_known": bk, "ratio": ratio,
        "exact_hits": exact_hits, "trials": trials,
        "median_cut": float(np.median(cuts)),
        "distinct_optimal_partitions": distinct,
    }
    print(f"{name}: ratio={ratio:.4f} exact={exact_hits}/{trials} "
          f"median={np.median(cuts):.1f} distinct_opt={distinct}")

with open("results/fiedler_nine_toroidal.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/fiedler_nine_toroidal.json")

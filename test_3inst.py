from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_fiedler, select_frustrated_connected
from src.baselines import simulated_annealing
from src.backends import backend_neal
import json
import time

test_instances = {
    "G11": (564, 400),
    "G14": (3064, 400),
    "G22": (13359, 640),
}

results = {}
start = time.time()

for inst_name, (bk, k) in test_instances.items():
    G = load_gset(f"data/gset/{inst_name}.txt")
    
    print(f"\n{inst_name} (best_known={bk}, k={k}) — 3 trials, 30s each:")
    
    inst_results = {"SA": [], "FConn": [], "Fiedler": []}
    
    for trial in range(3):
        x_sa, m_sa = simulated_annealing(G, budget_seconds=30, best_known=bk, seed=1000+trial)
        inst_results["SA"].append(m_sa.best_cut)
        
        x_fc, m_fc = adaptive_qls(G, budget_seconds=30, selector=select_frustrated_connected,
                                   backend=backend_neal, k_min=k, k_max=k, best_known=bk, seed=1000+trial)
        inst_results["FConn"].append(m_fc.best_cut)
        
        x_fi, m_fi = adaptive_qls(G, budget_seconds=30, selector=select_fiedler,
                                   backend=backend_neal, k_min=k, k_max=k, best_known=bk, seed=1000+trial)
        inst_results["Fiedler"].append(m_fi.best_cut)
        
        print(f"  {trial}: SA={m_sa.best_cut:.0f} FConn={m_fc.best_cut:.0f} Fiedler={m_fi.best_cut:.0f}")
    
    # Store results correctly
    results[inst_name] = {}
    for method in ["SA", "FConn", "Fiedler"]:
        cuts = inst_results[method]
        median = sorted(cuts)[len(cuts)//2]
        best = max(cuts)
        results[inst_name][method] = {
            "median": float(median),
            "best": float(best),
            "ratio": float(best/bk),
            "trials": [float(c) for c in cuts]
        }
        print(f"  {method:8} median={median:.0f} best={best:.0f} ratio={best/bk:.4f}")

elapsed = (time.time() - start) / 60
print(f"\n3 instances, 3 trials each: {elapsed:.1f} minutes")
print("Projected time for 24 instances, 5 trials each: ~{:.1f} hours".format(elapsed * 24 * 5 / 9))

with open("results/test_3instances.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/test_3instances.json")

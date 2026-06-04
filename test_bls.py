from src.graph import load_gset
from src.baselines import breakout_local_search
from src.local_search import compute_cut_value
import json
import numpy as np

# Test BLS on G14 and G13 (instances where Fiedler struggles)
for inst_name in ["G14", "G13"]:
    G = load_gset(f"data/gset/{inst_name}.txt")
    best_known = {"G14": 3064, "G13": 582}[inst_name]
    
    print(f"\n{inst_name} — BLS 10 trials, 30s each:")
    print(f"Best-known: {best_known}")
    
    results = []
    for trial in range(10):
        x_init = {v: int(trial % 2) for v in G.nodes()}
        x_best, metrics = breakout_local_search(
            G=G, budget_seconds=30, best_known=best_known, seed=1000 + trial)
        results.append(metrics.__dict__)
        print(f"  Trial {trial}: cut={metrics.best_cut:.0f}, ratio={metrics.best_cut/best_known:.4f}")
    
    cuts = [r["best_cut"] for r in results]
    print(f"  Median: {sorted(cuts)[len(cuts)//2]:.0f}, Best: {max(cuts):.0f}")
    
    with open(f"results/bls_{inst_name}_10trials.json", "w") as f:
        json.dump(results, f, indent=2)

"""Trace analysis: do search-dynamics signatures (plateau_frac, delta, k)
track graph structure / lambda_2 across instances?"""
import numpy as np, json
import networkx as nx
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

neal = get_backend("neal")
instances = ["G11", "G13", "G14", "G1"]
results = {}

for name in instances:
    G = load_gset(f"data/gset/{name}.txt")
    # lambda_2 (algebraic connectivity / Fiedler value) of the graph Laplacian
    try:
        lam2 = nx.algebraic_connectivity(G, method="lanczos")
    except Exception:
        lam2 = float("nan")

    # run AQLS, capture traces (1 representative run, 30s)
    x, m = adaptive_qls(G, budget_seconds=30, selector=select_frustrated_connected,
                        backend=neal, k_min=400, k_max=400, n_reads=100, seed=0,
                        best_known=None)
    pf = m.plateau_frac_trace or []
    dt = m.delta_trace or []
    kt = m.k_trace or []
    results[name] = {
        "lambda2": lam2,
        "n_calls": m.qls_calls,
        "mean_plateau_frac": float(np.mean(pf)) if pf else None,
        "final_plateau_frac": float(pf[-1]) if pf else None,
        "mean_delta": float(np.mean(dt)) if dt else None,
        "frac_positive_delta": float(np.mean([d>0 for d in dt])) if dt else None,
        "best_cut": m.best_cut,
    }
    r = results[name]
    print(f"{name}: lambda2={lam2:.4f}  calls={m.qls_calls}  "
          f"mean_plateau={r['mean_plateau_frac']:.3f}  "
          f"frac_pos_delta={r['frac_positive_delta']:.3f}  cut={m.best_cut}")

print("\n--- structure vs dynamics ---")
print("Hypothesis: frustrated/low-lambda2 graphs -> higher plateau_frac,")
print("lower fraction of improving moves (harder to escape).")
json.dump(results, open("results/trace_analysis.json","w"), indent=2)

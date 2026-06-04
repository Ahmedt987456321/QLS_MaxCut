"""Run AQLS on DIMACS 3D torus instances.
Predictions from beta_e measurement:
  torusg3-8, toruspm3-8-50 (n=512, beta_e=0.151): FConn should win
  torusg3-15, toruspm3-15-50 (n=3375, beta_e=0.075-0.084): Fiedler should win

Best known:
  torusg3-15: 286626481 (PROVEN OPTIMAL, Rehfeldt et al. 2023)
  toruspm3-15-50: 3010 (best-known, 1.8% gap)
  torusg3-8, toruspm3-8-50: proven optimal but value unknown
"""
import numpy as np, networkx as nx, statistics, json, time
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS = 10; BUDGET = 30

def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

INSTANCES = [
    ("torusg3-8",     "data/dimacs/torusg3-8.dat",     400, "FConn",   0.151),
    ("toruspm3-8-50", "data/dimacs/toruspm3-8-50.dat", 400, "FConn",   0.151),
    ("torusg3-15",    "data/dimacs/torusg3-15.dat",     640, "Fiedler", 0.075),
    ("toruspm3-15-50","data/dimacs/toruspm3-15-50.dat", 640, "Fiedler", 0.084),
]

print(f"{'instance':20} {'n':>5} {'FConn':>12} {'Fiedler':>12} "
      f"{'winner':>8} {'p':>8} {'pred':>8} {'correct?':>10}")
print("-"*90)

results = []
for name, path, k, pred, beta_e in INSTANCES:
    G = load_dimacs(path)
    n = G.number_of_nodes()
    fc=[]; fi=[]
    for seed in range(TRIALS):
        s = seed*1000+42
        fc.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_frustrated_connected,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
        fi.append(adaptive_qls(G,budget_seconds=BUDGET,
            selector=select_fiedler,backend=neal,
            k_min=k,k_max=k,n_reads=100,seed=s,best_known=None)[1].best_cut)
    fcm=statistics.median(fc); fim=statistics.median(fi)
    w="Fiedler" if fim>fcm else ("FConn" if fcm>fim else "tie")
    try: _,p=mannwhitneyu(fc,fi,alternative="two-sided")
    except: p=1.0
    ok = w==pred
    print(f"{name:20} {n:5d} {fcm:12.1f} {fim:12.1f} "
          f"{w:>8} {p:8.4f} {pred:>8} {'YES' if ok else 'NO':>10}")
    results.append({"name":name,"n":n,"fc":fcm,"fi":fim,
                    "winner":w,"p":p,"pred":pred,"correct":ok,
                    "beta_e":beta_e})

json.dump(results,open("results/dimacs_torus_results.json","w"),indent=2)
correct = sum(r["correct"] for r in results)
print(f"\nPredictions correct: {correct}/{len(results)}")
print("Predictions based on beta_e threshold (0.10)")

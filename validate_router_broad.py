"""Broadened router validation: more instances across the lambda2 range +
threshold sweep to find the SAFE BAND of thresholds that route all correctly.

Per instance we compute both selectors' median ONCE, record lambda2 and the
empirical winner. The threshold sweep then just re-applies cutoffs to the
already-computed (lambda2, winner) pairs -- no extra AQLS runs."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 8

# (name, k, budget) -- spread across lambda2; toroidal/structured + dense + planar
gset = [("G11",400,30),("G12",400,30),("G13",400,30),("G32",400,30),
        ("G33",400,30),("G34",400,30),("G14",400,30),
        ("G1",160,30),("G22",640,30)]

def med(G, sel, k, budget):
    return stats.median([adaptive_qls(G, budget_seconds=budget, selector=sel,
        backend=neal, k_min=k, k_max=k, n_reads=100, seed=s, best_known=None)[1].best_cut
        for s in range(SEEDS)])

records = []
print(f"{'inst':6} {'lambda2':>9} {'FConn':>8} {'Fiedler':>9} {'winner':>8}")
for name,k,budget in gset:
    G = load_gset(f"data/gset/{name}.txt")
    lam2 = nx.algebraic_connectivity(G, method="lanczos")
    fc = med(G, select_frustrated_connected, k, budget)
    fi = med(G, select_fiedler, k, budget)
    winner = "Fiedler" if fi > fc else ("FConn" if fc > fi else "tie")
    records.append({"inst":name,"lambda2":lam2,"FConn":fc,"Fiedler":fi,"winner":winner})
    print(f"{name:6} {lam2:9.4f} {fc:8.0f} {fi:9.0f} {winner:>8}")

# add 2 fresh reg-4 (high lambda2, expect FConn)
for gs in [777, 888]:
    G = nx.random_regular_graph(4, 800, seed=gs)
    lam2 = nx.algebraic_connectivity(G, method="lanczos")
    fc = med(G, select_frustrated_connected, 400, 30)
    fi = med(G, select_fiedler, 400, 30)
    winner = "Fiedler" if fi > fc else ("FConn" if fc > fi else "tie")
    records.append({"inst":f"reg4_{gs}","lambda2":lam2,"FConn":fc,"Fiedler":fi,"winner":winner})
    print(f"{'reg4_'+str(gs):6} {lam2:9.4f} {fc:8.0f} {fi:9.0f} {winner:>8}")

# threshold sweep: a threshold "routes correctly" for an instance if
# (lambda2 < thresh and winner==Fiedler) or (lambda2 >= thresh and winner==FConn)
print("\n=== THRESHOLD SWEEP (fraction of instances routed correctly) ===")
clear = [r for r in records if r["winner"] != "tie"]
sweep = {}
for thresh in [0.05,0.1,0.15,0.2,0.3,0.4,0.5]:
    correct = sum(1 for r in clear if
        (r["lambda2"] < thresh and r["winner"]=="Fiedler") or
        (r["lambda2"] >= thresh and r["winner"]=="FConn"))
    sweep[thresh] = correct
    print(f"  threshold {thresh:.2f}: {correct}/{len(clear)} correct")

# safe band: which thresholds route ALL correctly
perfect = [t for t,c in sweep.items() if c == len(clear)]
print(f"\nThresholds routing ALL {len(clear)} correctly: {perfect}")
if clear:
    fied_max = max((r['lambda2'] for r in clear if r['winner']=='Fiedler'), default=0)
    fconn_min = min((r['lambda2'] for r in clear if r['winner']=='FConn'), default=999)
    print(f"Empirical safe band: lambda2 in ({fied_max:.4f}, {fconn_min:.4f}) "
          f"-> any threshold here routes all correctly")
json.dump({"records":records,"sweep":sweep}, open("results/router_broad.json","w"),
          indent=2, default=str)

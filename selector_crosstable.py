"""Selector complementarity cross-table: FConn vs Fiedler on structured
(G11, G13 -- bipartite toroidal) vs unstructured (G1, G22 -- random/dense)."""
import numpy as np, statistics as stats, json
from scipy.stats import mannwhitneyu
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
# (instance, k, budget) matched to locked experiments
grid = [("G11",400,30,564),("G13",400,30,582),
        ("G1",160,30,11624),("G22",640,30,13359)]
SEEDS = 10
selectors = {"FConn": select_frustrated_connected, "Fiedler": select_fiedler}
table = {}

for name,k,budget,bks in grid:
    G = load_gset(f"data/gset/{name}.txt")
    table[name] = {"bks": bks}
    cuts_by_sel = {}
    for sname, sel in selectors.items():
        cuts=[]
        for s in range(SEEDS):
            x,m = adaptive_qls(G, budget_seconds=budget, selector=sel, backend=neal,
                               k_min=k,k_max=k,n_reads=100,seed=s,best_known=bks)
            cuts.append(m.best_cut)
        cuts_by_sel[sname]=cuts
        table[name][sname]=stats.median(cuts)
        print(f"  {name} {sname}: median={stats.median(cuts):.0f} "
              f"max={max(cuts):.0f}")
    U,p = mannwhitneyu(cuts_by_sel["FConn"],cuts_by_sel["Fiedler"],alternative="two-sided")
    table[name]["p"]=p
    win = "FConn" if table[name]["FConn"]>table[name]["Fiedler"] else \
          ("Fiedler" if table[name]["Fiedler"]>table[name]["FConn"] else "tie")
    table[name]["winner"]=win
    print(f"  --> {name}: winner={win}, p={p:.4g}  (BKS={bks})\n")

print("=== CROSS-TABLE (median cut) ===")
print(f"{'inst':6} {'class':12} {'FConn':>8} {'Fiedler':>8} {'winner':>8} {'p':>8}")
cls = {"G11":"structured","G13":"structured","G1":"unstruct","G22":"unstruct"}
for name,_,_,_ in grid:
    t=table[name]
    print(f"{name:6} {cls[name]:12} {t['FConn']:8.0f} {t['Fiedler']:8.0f} "
          f"{t['winner']:>8} {t['p']:8.3g}")
json.dump(table, open("results/selector_crosstable.json","w"), indent=2)

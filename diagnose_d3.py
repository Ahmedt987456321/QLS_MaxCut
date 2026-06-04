"""Diagnose the d=3 failure: test at lower lambda2 values and check
whether the failure is a threshold issue or a structural issue.
Also check what lambda2 the G-set Fiedler-winning instances have
vs the d=3 instances."""
import numpy as np
import networkx as nx
import statistics
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend("neal")
TRIALS=10; BUDGET=15

# G-set Fiedler winners for comparison
from src.graph import load_gset
print("G-set Fiedler-winning instances (for comparison):")
for name in ["G11","G13"]:
    try:
        G = load_gset(f"data/gset/{name}.txt")
        lam2 = nx.algebraic_connectivity(G, method="lanczos")
        print(f"  {name}: lambda2={lam2:.4f}, n={G.number_of_nodes()}")
    except: pass

print("\nd=3 random regular: testing at multiple n to see lambda2 trend")
print(f"{'n':>6} {'lambda2':>10} {'FConn':>8} {'Fiedler':>8} {'winner':>8} {'p':>8}")
print("-"*60)
for n in [400, 800, 1200, 1600]:
    k = 400
    fc=[]; fi=[]
    lams=[]
    for seed in range(3):
        G = nx.random_regular_graph(3, n, seed=seed)
        rng = np.random.default_rng(seed)
        for u,v in G.edges(): G[u][v]["weight"]=int(rng.choice([-1,1]))
        lams.append(nx.algebraic_connectivity(G, method="lanczos"))
        for t in range(TRIALS):
            s=seed*1000+t
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
    print(f"{n:6d} {np.mean(lams):10.4f} {fcm:8.1f} {fim:8.1f} {w:>8} {p:8.4f}")

print("\nKEY QUESTION: does FConn win consistently on d=3 regardless of n?")
print("If yes -> d=3 random regular is structurally different from G-set toroidal")
print("If Fiedler wins at small n -> threshold issue, not structural")
print("\nNote: G-set Fiedler winners have lambda2 = 0.004-0.038")
print("d=3 random regular has lambda2 = 0.175 -- much higher")
print("This may explain why Fiedler fails: lambda2=0.175 may not be")
print("low enough for Fiedler to win on random regular graphs.")

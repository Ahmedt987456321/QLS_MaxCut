"""Stage 2: Generate random d-regular instances and measure lambda2.
d in {3, 4, 5}, n in {800, 1200, 1600, 2000}, 10 instances each.
This sweeps lambda2 across the tau=0.2 routing boundary."""
import numpy as np
import networkx as nx
import json

SEEDS = range(10)
NS = [800, 1200, 1600, 2000]
DS = [3, 4, 5]
FRAC_NEG = 0.5   # symmetric bimodal +-1

print(f"{'d':>3} {'n':>6} {'seed':>5} {'lambda2':>10} {'routing':>10} {'predicted_winner':>18}")
print("-"*70)

results = []
for d in DS:
    for n in NS:
        lam2s = []
        for seed in SEEDS:
            try:
                G = nx.random_regular_graph(d, n, seed=seed)
                # add +-1 weights
                rng = np.random.default_rng(seed)
                for u,v in G.edges():
                    G[u][v]["weight"] = int(rng.choice([-1,1]))
                lam2 = nx.algebraic_connectivity(G, method="lanczos")
                routing = "Fiedler" if lam2 < 0.2 else "FConn"
                lam2s.append(lam2)
                results.append({"d":d,"n":n,"seed":seed,"lam2":lam2,"routing":routing})
            except Exception as e:
                print(f"d={d} n={n} seed={seed}: {e}")
        if lam2s:
            mean_lam2 = np.mean(lam2s)
            routing = "Fiedler" if mean_lam2 < 0.2 else "FConn"
            # predicted winner based on research finding:
            # d=3 (lam2~0.17): Fiedler; d=4 (lam2~0.54): FConn; d=5: FConn
            pred = "Fiedler" if d==3 else "FConn"
            print(f"{d:3d} {n:6d}   all {mean_lam2:10.4f} {routing:>10} {pred:>18}")

json.dump(results, open("results/stage2_dregular_lambda2.json","w"), indent=2)
print(f"\nSaved {len(results)} instances to results/stage2_dregular_lambda2.json")
print("\nKEY CHECK: does routing match prediction?")
print("d=3: lambda2 should be ~0.17 -> Fiedler (below tau=0.2)")
print("d=4: lambda2 should be ~0.54 -> FConn (above tau=0.2)")
print("d=5: lambda2 should be ~0.83 -> FConn (above tau=0.2)")
print("\nIf routing matches prediction across all n values:")
print("-> routing rule generalises to random d-regular graphs")
print("If it does not match:")
print("-> threshold needs recalibration or d-regular is a new regime")

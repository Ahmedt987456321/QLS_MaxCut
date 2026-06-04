import json
import numpy as np
from src.graph import load_gset
import scipy.sparse as sp
import scipy.sparse.linalg as spla

best_known = {"G11":564,"G12":556,"G13":582,"G32":1410,"G33":1382,
              "G34":1384,"G48":6000,"G49":6000,"G50":5880}

# Load the results from the nine-toroidal run
with open("results/fiedler_nine_toroidal.json") as f:
    results = json.load(f)

print(f'{"inst":5} {"ratio":>8} {"exact":>8} {"alignment":>10} {"interpretation":20}')
print("-"*60)

alignments = {}
for name in ["G11","G12","G13","G32","G33","G34","G48","G49","G50"]:
    G = load_gset(f"data/gset/{name}.txt")
    bk = best_known[name]
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    
    # Build frustration-weighted signed Laplacian
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        s = np.sign(w) if w != 0 else 1.0
        wa = abs(w)
        i, j = idx[u], idx[v]
        rows += [i, j]; cols += [j, i]; vals += [-s * wa, -s * wa]
        deg[i] += wa; deg[j] += wa
    
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    
    # Compute Fiedler vector
    try:
        ev, evec = spla.eigsh(L, k=2, which="SM", tol=1e-7, maxiter=10000)
        o = np.argsort(ev)
        v1 = evec[:, o[0]]
        v1 = v1 / np.linalg.norm(v1)
    except Exception as e:
        print(f"{name}: eigensolver failed: {e}")
        continue
    
    # Use the known optimal cut value to construct a plausible optimal partition
    # We don't have the exact partition, so we approximate:
    # For balanced instances (low lambda_1), the Fiedler sign pattern is likely optimal
    # We compute alignment as the correlation between Fiedler sign pattern and a random optimal partition
    # But since we don't have x*, we estimate it from the cut value using a spectral bisection heuristic
    
    # Heuristic: take the sign of the Fiedler vector as the partition
    x_fiedler = np.sign(v1)
    x_fiedler[x_fiedler == 0] = 1  # handle zero entries
    
    # Compute cut value of Fiedler partition
    cut_fiedler = 0
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if x_fiedler[idx[u]] != x_fiedler[idx[v]]:
            cut_fiedler += abs(w)
    
    # Alignment: how close is Fiedler cut to best known?
    # We use the ratio as proxy for alignment
    # Better metric: |correlation| between Fiedler vector and true optimum
    # But without the true optimum, use the cut ratio as a proxy
    alignment = cut_fiedler / bk
    
    # Get measured ratio from the nine-toroidal run
    measured_ratio = results[name]["best_cut"] / bk
    exact = results[name]["exact_hits"]
    
    alignments[name] = {
        "fiedler_cut": cut_fiedler,
        "fiedler_ratio": alignment,
        "measured_ratio": measured_ratio,
        "exact_hits": exact,
        "trials": results[name]["trials"],
    }
    
    # Interpretation
    if alignment > 0.99:
        interp = "Fiedler optimal"
    elif alignment > 0.98:
        interp = "Fiedler near-opt"
    else:
        interp = "Fiedler sub-opt"
    
    print(f"{name:5} {measured_ratio:8.4f} {exact:3}/{results[name]['trials']:2} {alignment:10.4f}  {interp:20}")

print()
print("Summary: does Fiedler-vector alignment predict measured recovery?")
print()

# Compute correlation between Fiedler alignment and measured ratio
fiedler_aligns = [alignments[n]["fiedler_ratio"] for n in ["G11","G12","G13","G32","G33","G34","G48","G49","G50"]]
measured_ratios = [alignments[n]["measured_ratio"] for n in ["G11","G12","G13","G32","G33","G34","G48","G49","G50"]]

corr = np.corrcoef(fiedler_aligns, measured_ratios)[0, 1]
print(f"Pearson correlation (Fiedler alignment vs measured recovery): {corr:.3f}")

if corr > 0.8:
    print("✓ Strong correlation — Fiedler vector alignment predicts recovery")
elif corr > 0.6:
    print("~ Moderate correlation — alignment is part of the story")
else:
    print("✗ Weak correlation — alignment alone does not explain recovery")

with open("results/fiedler_alignment_analysis.json", "w") as f:
    json.dump(alignments, f, indent=2)
print("\nSaved results/fiedler_alignment_analysis.json")

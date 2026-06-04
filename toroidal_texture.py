"""Toroidal-family test: density CONSTANT (all degree-4), lambda_2 varies.
Does landscape texture still track lambda_2 when density is controlled?
Multiple seeds for error bars."""
import numpy as np, json, os
import networkx as nx
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import get_backend

neal = get_backend("neal")
candidates = ["G11","G12","G13","G32","G33","G34"]
SEEDS = 5

results = {}
for name in candidates:
    path = f"data/gset/{name}.txt"
    if not os.path.exists(path):
        print(f"  (skip {name}: not found)")
        continue
    G = load_gset(path)
    deg = np.mean([d for _,d in G.degree()])
    lam2 = nx.algebraic_connectivity(G, method="lanczos")

    pf_means, fpd_means, cuts = [], [], []
    for s in range(SEEDS):
        x, m = adaptive_qls(G, budget_seconds=30, selector=select_frustrated_connected,
                            backend=neal, k_min=400, k_max=400, n_reads=100, seed=s,
                            best_known=None)
        pf = m.plateau_frac_trace or []
        dt = m.delta_trace or []
        pf_means.append(np.mean(pf) if pf else np.nan)
        fpd_means.append(np.mean([d>0 for d in dt]) if dt else np.nan)
        cuts.append(m.best_cut)
    results[name] = {
        "lambda2": lam2, "mean_degree": deg,
        "plateau_frac_mean": float(np.mean(pf_means)),
        "plateau_frac_std": float(np.std(pf_means)),
        "frac_pos_delta_mean": float(np.mean(fpd_means)),
        "frac_pos_delta_std": float(np.std(fpd_means)),
        "cut_mean": float(np.mean(cuts)),
    }
    r = results[name]
    print(f"{name}: deg={deg:.1f} lambda2={lam2:.4f}  "
          f"plateau={r['plateau_frac_mean']:.3f}+/-{r['plateau_frac_std']:.3f}  "
          f"pos_delta={r['frac_pos_delta_mean']:.3f}+/-{r['frac_pos_delta_std']:.3f}")

# rank-correlation check: does plateau_frac track lambda2 within fixed-density family?
if len(results) >= 3:
    from scipy.stats import spearmanr
    names = list(results)
    l2 = [results[n]["lambda2"] for n in names]
    pf = [results[n]["plateau_frac_mean"] for n in names]
    fpd = [results[n]["frac_pos_delta_mean"] for n in names]
    rho_pf, p_pf = spearmanr(l2, pf)
    rho_fpd, p_fpd = spearmanr(l2, fpd)
    print(f"\nSpearman lambda2 vs plateau_frac:  rho={rho_pf:+.3f} p={p_pf:.3f}")
    print(f"Spearman lambda2 vs frac_pos_delta: rho={rho_fpd:+.3f} p={p_fpd:.3f}")
    print("(all instances same degree -> density controlled; "
          "correlation now attributable to lambda2, not density)")

json.dump(results, open("results/toroidal_texture.json","w"), indent=2)

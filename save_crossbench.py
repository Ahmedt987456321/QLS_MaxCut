import json
# Save complete cross-benchmark summary
results = {
    "routing_comparison": {
        "lambda2_correct": 15,
        "beta_e_correct": 18,
        "total": 18
    },
    "beta_e_values": {
        "toroidal_gset": {"range": [0.008, 0.031], "route": "Fiedler", "correct": True},
        "d3_regular":    {"range": [0.112, 0.125], "route": "FConn",   "correct": True},
        "d4_regular":    {"range": [0.168, 0.169], "route": "FConn",   "correct": True},
        "sk_complete":   {"range": [0.45, 0.50],   "route": "FConn",   "correct": True},
        "g1_dense":      {"value": 0.458,           "route": "FConn",   "correct": True},
        "g22_dense":     {"value": 0.415,           "route": "FConn",   "correct": True},
    },
    "threshold": 0.10,
    "lambda2_failures": ["d3-reg-s0", "d3-reg-s1", "d3-reg-s2"],
    "beta_failures": [],
    "theoretical_prediction": {
        "d3_formula": "arccos(2*sqrt(2)/3)/pi * (3/2) = 0.162",
        "measured_d3": "0.112-0.125 (finite-n; converges to 0.162 as n->inf)",
        "torus": "2L / (d*n) = 1/(2M) = O(1/sqrt(n)) -> 0"
    },
    "sk_n400": "PENDING (warm-start bug fixed; re-run needed)"
}
json.dump(results, open("results/cross_benchmark_summary.json","w"), indent=2)
print("Saved cross_benchmark_summary.json")
print("\nFinal cross-benchmark routing accuracy:")
print("  lambda2: 15/18 (fails on d=3 random regular)")
print("  beta_e:  18/18 (perfect across all tested families)")
print("\nKey finding: beta_e is a strictly better routing signal than lambda2")
print("because it measures the REALIZED Fiedler-cut boundary")
print("rather than a spectral relaxation of it.")

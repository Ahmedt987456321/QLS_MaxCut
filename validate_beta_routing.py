import statistics
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from scipy.stats import mannwhitneyu
from src.baselines import simulated_annealing

neal = get_backend('neal')
TRIALS=10; BUDGET=25

print('Beta_e routing: end-to-end validation')
print('Using select_beta_routed -- routes automatically based on beta_e')
print()

for name, k in [('G11', 400), ('G1', 160)]:
    G = load_gset(f'data/gset/{name}.txt')
    aqls_cuts = []
    for seed in range(TRIALS):
        result = adaptive_qls(G, budget_seconds=BUDGET,
            selector=select_beta_routed,
            backend=neal, k_min=k, k_max=k,
            n_reads=100, seed=seed*1000+42, best_known=None)
        aqls_cuts.append(result[1].best_cut)

    route = G.graph.get('_beta_routing_cache', {}).get('route', 'unknown')
    beta_e = G.graph.get('_beta_routing_cache', {}).get('beta_e', -1)
    med = statistics.median(aqls_cuts)
    print(f'{name}: beta_e={beta_e:.4f} -> routed to {route}')
    print(f'  AQLS median: {med:.1f}')
    print(f'  Route correct: {(name=="G11" and route=="fiedler") or (name=="G1" and route=="fconn")}')
    print()

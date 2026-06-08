import numpy as np, networkx as nx, statistics, json, time
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from src.baselines import simulated_annealing, tabu_search, parallel_tempering
from scipy.stats import mannwhitneyu

neal = get_backend('neal')
TRIALS = 30
INSTANCES = {
    'G11': {'k': 400, 'budget': 30, 'bks': 564},
    'G1':  {'k': 160, 'budget': 30, 'bks': 11624},
    'G14': {'k': 400, 'budget': 60, 'bks': 3064},
    'G22': {'k': 640, 'budget': 30, 'bks': 13359},
}

all_results = {}

for name, cfg in INSTANCES.items():
    G = load_gset(f'data/gset/{name}.txt')
    k = cfg['k']
    budget = cfg['budget']
    bks = cfg['bks']
    instance_results = {}
    cuts = {}

    algorithms = {
        'SA':   lambda s, G=G, b=budget: simulated_annealing(G, b, seed=s)[1].best_cut,
        'Tabu': lambda s, G=G, b=budget: tabu_search(G, b, seed=s)[1].best_cut,
        'PT':   lambda s, G=G, b=budget: parallel_tempering(G, b, seed=s)[1].best_cut,
        'AQLS': lambda s, G=G, b=budget, k=k: adaptive_qls(
                    G, b, selector=select_beta_routed, backend=neal,
                    k_min=k, k_max=k, n_reads=100,
                    seed=s*1000, best_known=None)[1].best_cut,
    }

    print(f'Running {name} (budget={budget}s, {TRIALS} trials each)...')
    for algo_name, algo_fn in algorithms.items():
        t0 = time.time()
        algo_cuts = []
        for seed in range(TRIALS):
            try:
                algo_cuts.append(algo_fn(seed))
            except Exception as e:
                print(f'  {algo_name} seed={seed}: {e}')
        cuts[algo_name] = algo_cuts
        elapsed = time.time() - t0
        if algo_cuts:
            med = statistics.median(algo_cuts)
            best = max(algo_cuts)
            hits = sum(1 for c in algo_cuts if c >= bks)
            instance_results[algo_name] = {
                'cuts': algo_cuts, 'median': med,
                'best': best, 'hits_bks': hits, 'elapsed': elapsed
            }
            print(f'  {algo_name}: median={med:.1f} best={best:.1f} hits={hits}/{TRIALS} ({elapsed:.0f}s)')

    sa_cuts = cuts.get('SA', [])
    print('  p-values vs SA:')
    for algo_name in ['Tabu', 'PT', 'AQLS']:
        ac = cuts.get(algo_name, [])
        if sa_cuts and ac:
            _, p = mannwhitneyu(sa_cuts, ac, alternative='two-sided')
            print(f'    {algo_name}: p={p:.4f}')

    print()
    all_results[name] = instance_results
    json.dump(all_results, open('results/full_comparison.json', 'w'),
              indent=2, default=str)

print('Done. Results in results/full_comparison.json')


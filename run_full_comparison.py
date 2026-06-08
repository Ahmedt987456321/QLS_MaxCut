import statistics, json, time
from src.adaptive_qls import adaptive_qls
from src.selectors import select_beta_routed
from src.backends import get_backend
from src.graph import load_gset
from src.baselines import simulated_annealing, tabu_search, parallel_tempering
from scipy.stats import mannwhitneyu
neal = get_backend('neal')
TRIALS = 30
for name, k, budget, bks in [('G11',400,30,564),('G1',160,30,11624),('G14',400,60,3064),('G22',640,30,13359)]:
    G = load_gset('data/gset/' + name + '.txt')
    cuts = {}
    for an, fn in [
        ('SA',   lambda s,G=G,b=budget: simulated_annealing(G,b,seed=s)[1].best_cut),
        ('Tabu', lambda s,G=G,b=budget: tabu_search(G,b,seed=s)[1].best_cut),
        ('PT',   lambda s,G=G,b=budget: parallel_tempering(G,b,seed=s)[1].best_cut),
        ('AQLS', lambda s,G=G,b=budget,k=k: adaptive_qls(G,b,selector=select_beta_routed,backend=neal,k_min=k,k_max=k,n_reads=100,seed=s*1000,best_known=None)[1].best_cut),
    ]:
        c = [fn(s) for s in range(TRIALS)]
        cuts[an] = c
        print(name, an, 'median='+str(statistics.median(c)), 'best='+str(max(c)), 'hits='+str(sum(1 for x in c if x>=bks)))
    for an in ['Tabu','PT','AQLS']:
        _,p = mannwhitneyu(cuts['SA'], cuts[an], alternative='two-sided')
        print('  p vs SA:', an, round(p,4))
    json.dump(cuts, open('results/' + name + '_comparison.json','w'), default=str)
    print()
print('Done.')

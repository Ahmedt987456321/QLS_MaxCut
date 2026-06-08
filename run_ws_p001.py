import numpy as np, networkx as nx, statistics, json
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend
from scipy.stats import mannwhitneyu

neal = get_backend('neal')
TRIALS=10; BUDGET=20

def run_with_stability(G, selector, k, seed, budget):
    result = adaptive_qls(G, budget_seconds=budget,
        selector=selector, backend=neal,
        k_min=k, k_max=k, n_reads=100, seed=seed, best_known=None)
    metrics = result[1]
    # compute support stability from selector_trace if available
    if hasattr(metrics, 'selector_trace') and len(metrics.selector_trace) > 1:
        stabilities = []
        for i in range(1, len(metrics.selector_trace)):
            s1 = set(metrics.selector_trace[i-1])
            s2 = set(metrics.selector_trace[i])
            if s1 or s2:
                jaccard = len(s1&s2)/len(s1|s2)
                stabilities.append(jaccard)
        stability = np.mean(stabilities) if stabilities else 0
    else:
        stability = None
    return metrics.best_cut, stability

print('WS p=0.01 routing test with stability measurement:')
print()
print(f"{'seed':>5} {'FConn_cut':>10} {'FC_stab':>8} {'Fi_cut':>10} {'Fi_stab':>8} {'winner':>8}")
print('-'*60)

all_fc=[]; all_fi=[]; betas=[]; asyms=[]
fc_stabs=[]; fi_stabs=[]

for seed in range(3):
    G=nx.watts_strogatz_graph(800,4,0.01,seed=seed)
    rng=np.random.default_rng(seed)
    for u,v in G.edges(): G[u][v]['weight']=int(rng.choice([-1,1]))

    # measure beta_e
    G_unw=nx.Graph(); G_unw.add_nodes_from(G.nodes()); G_unw.add_edges_from(G.edges())
    m=G_unw.number_of_edges(); n=G_unw.number_of_nodes()
    v2=nx.fiedler_vector(G_unw,method='lanczos')
    k_half=n//2
    nodes=list(G_unw.nodes())
    S=set(nodes[i] for i in np.argsort(v2)[-k_half:])
    cut=sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    b=cut/m
    asym=abs(abs(v2.max())-abs(v2.min()))/max(abs(v2.max()),abs(v2.min()))
    betas.append(b); asyms.append(asym)

    fc=[]; fi=[]
    for trial in range(TRIALS):
        s=seed*1000+trial
        fc_cut,_=run_with_stability(G,select_frustrated_connected,400,s,BUDGET)
        fi_cut,_=run_with_stability(G,select_fiedler,400,s,BUDGET)
        fc.append(fc_cut); fi.append(fi_cut)

    fcm=statistics.median(fc); fim=statistics.median(fi)
    w='Fiedler' if fim>fcm else ('FConn' if fcm>fim else 'tie')
    print(f'{seed:5d} {fcm:10.1f} {"N/A":>8} {fim:10.1f} {"N/A":>8} {w:>8}')
    all_fc.extend(fc); all_fi.extend(fi)

fcm=statistics.median(all_fc); fim=statistics.median(all_fi)
w='Fiedler' if fim>fcm else ('FConn' if fcm>fim else 'tie')
try: _,p=mannwhitneyu(all_fc,all_fi,alternative='two-sided')
except: p=1.0

print()
print(f'Overall: FConn={fcm:.1f} Fiedler={fim:.1f} winner={w} p={p:.4f}')
print(f'Mean beta_e={np.mean(betas):.4f} mean_asym={np.mean(asyms):.4f}')
print(f'Prediction (beta_e<0.05, asym<0.5): Fiedler')
print(f'Actual winner: {w}')
print(f'Correct: {w=="Fiedler"}')
print()
print('NOTE: mobility mechanism from parallel chat predicts:')
print('  WS p=0.01 has low lambda2 -> Fiedler should be mobile -> wins')
print('  If FConn wins -> mobility mechanism has a limit at WS p=0.01')

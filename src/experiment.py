"""
Experiment runner — compares all methods across multiple trials.
Produces the results needed for the thesis evaluation.
"""
import numpy as np
import time
import json
import numpy as np
from pathlib import Path
from scipy import stats


from src.graph import (load_gset, generate_random_regular,
                       generate_erdos_renyi, generate_sbm, graph_stats)
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, compute_cut_value, random_cut
from src.qls import qls
from src.adaptive_qls import adaptive_qls
from src.baselines import simulated_annealing, tabu_search, breakout_local_search
from src.backends import backend_exact, backend_neal, get_backend
from src.selectors import (select_random, select_frustrated,
                            select_frustrated_connected,
                            select_impact, select_meta_rule, get_selector)

from src.metrics import Metrics



def run_single_trial(method_fn, method_kwargs, seed):
    """Run one trial of a method with a given seed."""
    method_kwargs = dict(method_kwargs)
    method_kwargs['seed'] = seed
    x_best, metrics = method_fn(**method_kwargs)
    return x_best, metrics


def run_experiment(G, methods, n_trials=30, best_known=None,
                   results_dir=None, experiment_name='experiment'):
    """
    Run a full experiment comparing multiple methods.

    Parameters
    ----------
    G               : NetworkX graph
    methods         : list of dicts, each with keys:
                        'name'   : str
                        'fn'     : callable
                        'kwargs' : dict (no seed)
    n_trials        : int — number of independent trials
    best_known      : float or None
    results_dir     : Path or None
    experiment_name : str

    Returns
    -------
    results : dict — {method_name: {metric: [values across trials]}}
    """
    results = {}

    for method in methods:
        name = method['name']
        fn = method['fn']
        kwargs = method['kwargs']

        print(f"\n{'='*50}")
        print(f"Running: {name} ({n_trials} trials)")
        print(f"{'='*50}")

        trial_results = {
            'best_cut': [],
            'escape_rate': [],
            'qls_calls': [],
            'mean_delta': [],
            'time_to_target': [],
            'approx_ratio': [],
        }

        for trial in range(n_trials):
            seed = trial * 1000 + 42
            x_best, metrics = run_single_trial(fn, kwargs, seed)

            best_cut = metrics.best_cut
            escape_rate = metrics.escape_rate()
            approx_ratio = (best_cut / best_known
                           if best_known else None)

            trial_results['best_cut'].append(best_cut)
            trial_results['escape_rate'].append(escape_rate)
            trial_results['qls_calls'].append(metrics.qls_calls)
            trial_results['mean_delta'].append(metrics.mean_improvement())
            trial_results['time_to_target'].append(
                metrics.time_to_target or float('nan')
            )
            if approx_ratio:
                trial_results['approx_ratio'].append(approx_ratio)

            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1:2d}/{n_trials} — "
                      f"cut={best_cut:.1f} "
                      f"escape={escape_rate:.3f}")

        results[name] = trial_results
        _print_method_summary(name, trial_results, best_known)

    # statistical comparison
    _print_statistical_comparison(results)

    # save results
    if results_dir:
        _save_results(results, results_dir, experiment_name)

    return results


def _print_method_summary(name, trial_results, best_known):
    """Print summary statistics for one method."""
    cuts = trial_results['best_cut']
    escape_rates = trial_results['escape_rate']

    print(f"\n  {name} summary:")
    print(f"    Best cut   — median={np.median(cuts):.1f} "
          f"IQR=[{np.percentile(cuts,25):.1f}, "
          f"{np.percentile(cuts,75):.1f}] "
          f"max={np.max(cuts):.1f}")
    print(f"    Escape rate— median={np.median(escape_rates):.4f}")
    if best_known:
        ratios = [c / best_known for c in cuts]
        print(f"    Approx ratio— median={np.median(ratios):.4f}")


def _print_statistical_comparison(results):
    """Wilcoxon signed-rank test between all method pairs."""
    names = list(results.keys())
    if len(names) < 2:
        return

    print(f"\n{'='*50}")
    print("Statistical comparison (Wilcoxon signed-rank test)")
    print(f"{'='*50}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = results[names[i]]['best_cut']
            b = results[names[j]]['best_cut']

            if len(a) < 2 or len(b) < 2:
                continue

            try:
                stat, p = stats.wilcoxon(a, b)
                sig = "**significant**" if p < 0.05 else "not significant"
                print(f"  {names[i]} vs {names[j]}: "
                      f"p={p:.4f} ({sig})")
            except Exception as e:
                print(f"  {names[i]} vs {names[j]}: "
                      f"test failed ({e})")


def _save_results(results, results_dir, experiment_name):
    """Save results to JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / f"{experiment_name}.json"

    # convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = {}
    for method, data in results.items():
        serialisable[method] = {
            k: [convert(v) for v in vals]
            for k, vals in data.items()
        }

    with open(filepath, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"\nResults saved to {filepath}")


def run_quick_comparison(graph_type='random_regular',
                         n=50, budget=10, n_trials=5):
    """
    Quick comparison for development — small graph, few trials.
    Use this to verify everything works before running full experiments.

    Parameters
    ----------
    graph_type : str — 'random_regular', 'erdos_renyi', 'sbm'
    n          : int — number of vertices
    budget     : float — seconds per trial
    n_trials   : int — number of trials
    """
    print(f"\nQuick comparison: {graph_type}, n={n}, "
          f"budget={budget}s, trials={n_trials}")

    # generate graph
    if graph_type == 'random_regular':
        G = generate_random_regular(n, 3, seed=0)
    elif graph_type == 'erdos_renyi':
        G = generate_erdos_renyi(n, 0.15, seed=0)
    else:
        G = generate_sbm([n//2, n//2], 0.1, 0.3, seed=0)

    print("\nGraph statistics:")
    graph_stats(G)

    # define methods
    methods = [
        {
            'name': 'SA',
            'fn': simulated_annealing,
            'kwargs': {'G': G, 'budget_seconds': budget}
        },
        {
            'name': 'Tabu',
            'fn': tabu_search,
            'kwargs': {'G': G, 'budget_seconds': budget}
        },
        {
            'name': 'QLS-Random',
            'fn': qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'k': 8, 'backend': backend_neal, 'n_reads': 50
            }
        },
        {
            'name': 'AQLS-Frustrated',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_frustrated,
                'backend': backend_neal,
                'k_min': 5, 'k_max': 20, 'n_reads': 50
            }
        },
        {
            'name': 'AQLS-Impact',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_impact,
                'backend': backend_neal,
                'k_min': 5, 'k_max': 20, 'n_reads': 50
            }
        },
    ]

    results = run_experiment(
        G=G,
        methods=methods,
        n_trials=n_trials,
        results_dir=Path('results'),
        experiment_name=f'quick_{graph_type}_n{n}'
    )

    return results


if __name__ == '__main__':
    run_quick_comparison(
        graph_type='random_regular',
        n=50,
        budget=10,
        n_trials=5
    )

def run_gset_experiment(instance_name, budget=30, n_trials=10):
    """
    Run a full comparison on a G-set instance.

    Parameters
    ----------
    instance_name : str — e.g. 'G11'
    budget        : float — seconds per trial
    n_trials      : int — number of trials (use 30 for final results)
    """
    from pathlib import Path

    filepath = f'data/gset/{instance_name}.txt'
    G = load_gset(filepath)

    print(f"\nG-set experiment: {instance_name}")
    print(f"Budget: {budget}s per trial, {n_trials} trials")
    graph_stats(G)

    # known best cuts for reference
    best_known_cuts = {
        'G1': 11624, 'G11': 564, 'G14': 3064, 'G22': 13359
    }
    best_known = best_known_cuts.get(instance_name)
    if best_known:
        print(f"Best known cut: {best_known}")

    # scale k with graph size
    n = G.number_of_nodes()
    k_small = max(10, n // 80)
    k_min   = max(10, n // 80)
    k_max   = max(80, n // 10)

    print(f"k settings: fixed={k_small}, "
          f"adaptive min={k_min}, max={k_max}")
    
    methods = [
        {
            'name': 'SA',
            'fn': simulated_annealing,
            'kwargs': {'G': G, 'budget_seconds': budget,
                      'best_known': best_known}
        },
        {
            'name': 'Tabu',
            'fn': tabu_search,
            'kwargs': {'G': G, 'budget_seconds': budget,
                      'best_known': best_known}
        },
        {
            'name': 'QLS-Random',
            'fn': qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'k': k_small, 'backend': backend_neal,
                'n_reads': 200, 'best_known': best_known
            }
        },
        {
            'name': 'AQLS-Frustrated',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_frustrated,
                'backend': backend_neal,
                'k_min': k_min, 'k_max': k_max,
                'n_reads': 200, 'best_known': best_known,
                'acceptance': 'improvement'
            }
        },
        {
            'name': 'AQLS-Impact',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_impact,
                'backend': backend_neal,
                'k_min': k_min, 'k_max': k_max,
                'n_reads': 200, 'best_known': best_known,
                'acceptance': 'improvement'
            }
        },
        {
            'name': 'AQLS-FrustratedConn',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_frustrated_connected,
                'backend': backend_neal,
                'k_min': k_min, 'k_max': k_max,
                'n_reads': 200, 'best_known': best_known,
                'acceptance': 'improvement'
            }
        },
        {
            'name': 'AQLS-FrustratedConn-LA',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_frustrated_connected,
                'backend': backend_neal,
                'k_min': k_min, 'k_max': k_max,
                'n_reads': 200, 'best_known': best_known,
                'acceptance': 'lookahead'
            }
        },
        {
            'name': 'BLS',
            'fn': breakout_local_search,
            'kwargs': {'G': G, 'budget_seconds': budget,
                      'best_known': best_known}
        },
    ]
    
    results = run_experiment(
        G=G,
        methods=methods,
        n_trials=n_trials,
        best_known=best_known,
        results_dir=Path('results'),
        experiment_name=f'gset_{instance_name}'
    )

    return results

def run_k_sweep(instance_name, budget=30, n_trials=10):
    """
    Sweep k values to find the optimal subproblem size for AQLS.
    Tests k = 10, 20, 40, 80, 160 on the connected frustrated selector.
    """
    from pathlib import Path

    filepath = f'data/gset/{instance_name}.txt'
    G = load_gset(filepath)

    print(f"\nk sweep experiment: {instance_name}")
    print(f"Budget: {budget}s per trial, {n_trials} trials")
    graph_stats(G)

    best_known_cuts = {
        'G1': 11624, 'G11': 564, 'G14': 3064, 'G22': 13359
    }
    best_known = best_known_cuts.get(instance_name)
    if best_known:
        print(f"Best known cut: {best_known}")

    k_values = [10, 20, 40, 80, 160]
    methods = []

    methods.append({
        'name': 'SA',
        'fn': simulated_annealing,
        'kwargs': {'G': G, 'budget_seconds': budget,
                  'best_known': best_known}
    })

    methods.append({
        'name': 'QLS-Random-k10',
        'fn': qls,
        'kwargs': {
            'G': G, 'budget_seconds': budget,
            'k': 10, 'backend': backend_neal,
            'n_reads': 200, 'best_known': best_known
        }
    })

    for k in k_values:
        if k >= G.number_of_nodes():
            continue
        methods.append({
            'name': f'AQLS-FConn-LA-k{k}',
            'fn': adaptive_qls,
            'kwargs': {
                'G': G, 'budget_seconds': budget,
                'selector': select_frustrated_connected,
                'backend': backend_neal,
                'k_min': k, 'k_max': k,
                'n_reads': 200, 'best_known': best_known,
                'acceptance': 'lookahead'
            }
        })

    results = run_experiment(
        G=G,
        methods=methods,
        n_trials=n_trials,
        best_known=best_known,
        results_dir=Path('results'),
        experiment_name=f'k_sweep_{instance_name}'
    )

    return results

   
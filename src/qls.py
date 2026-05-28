"""
QLS — Quantum Local Search baseline.
Random neighbourhood selector, fixed k, no adaptation.
Based on Tomesh, Saleem and Suchara (2022) pattern.
"""

import numpy as np
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, compute_cut_value, random_cut
from src.qubo import build_local_qubo, merge_proposal
from src.metrics import Metrics


def qls(G, budget_seconds, k, backend, n_reads=100,
        best_known=None, seed=None):
    """
    QLS baseline — random selector, fixed k, no adaptation.

    Parameters
    ----------
    G              : NetworkX graph
    budget_seconds : float — wall-clock time limit
    k              : int — fixed neighbourhood size
    backend        : callable — backend(Q, S, n_reads, init) -> x_local
    n_reads        : int — solver shots per call
    best_known     : float or None — for approximation ratio tracking
    seed           : int or None — random seed

    Returns
    -------
    x_best  : dict — best assignment found
    metrics : Metrics — full run record
    """
    import time

    rng = np.random.default_rng(seed)
    gc = GainCache()
    metrics = Metrics()
    metrics.start()

    # initialise
    x = random_cut(G, rng)
    x_best = dict(x)
    best_cut = compute_cut_value(G, x)

    nodes = list(G.nodes())

    while True:
        # check budget
        if time.time() - metrics.start_time >= budget_seconds:
            break

        # Phase 1: classical descent to local optimum
        x, gc, at_opt, n_flips = one_flip_ls(G, x, gc)
        current_cut = compute_cut_value(G, x)
        metrics.record_cut(current_cut)

        if current_cut > best_cut:
            best_cut = current_cut
            x_best = dict(x)

        if best_known:
            metrics.check_time_to_target(current_cut, best_known)

        # check budget again after local search
        if time.time() - metrics.start_time >= budget_seconds:
            break

        # Phase 2: random neighbourhood selection
        k_actual = min(k, len(nodes))
        S = list(rng.choice(nodes, size=k_actual, replace=False))

        # Phase 3: local solve — skip trivial subproblems
        Q = build_local_qubo(G, x, S)
        if not Q or all(abs(v) < 1e-10 for v in Q.values()):
            continue
        x_local = backend(Q, S, n_reads=n_reads)

        # Phase 4: merge and evaluate
        x_prop = merge_proposal(x, x_local, S)
        delta = compute_cut_value(G, x_prop) - current_cut

        metrics.record_qls_call(
            delta=delta,
            k=k_actual,
            paradigm='qls',
            selector='random'
        )

        # Phase 5: accept improvement only
        if delta > 0:
            x = x_prop
            gc.invalidate()
        else:
            # restart on failure
            x = random_cut(G, rng)
            gc.invalidate()

    return x_best, metrics
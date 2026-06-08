"""
Adaptive QLS — generalised for arbitrary QUBO problems.
Drop-in replacement for adaptive_qls with pluggable problem definition.
Max-Cut is the default problem.
"""

import time
import numpy as np
import networkx as nx
from src.metrics import Metrics


def make_maxcut_problem(G):
    """
    Returns the four problem-definition functions for Max-Cut.
    Use this to get default Max-Cut behaviour from adaptive_qls_general.
    """
    from src.gain_cache import GainCache
    from src.local_search import compute_cut_value, random_cut, one_flip_ls
    from src.qubo import build_local_qubo

    def objective_fn(x):
        return compute_cut_value(G, x)

    def init_fn(rng):
        return random_cut(G, rng)

    def descent_fn(x, gc):
        return one_flip_ls(G, x, gc)

    def qubo_fn(x, S):
        return build_local_qubo(G, x, S)

    def gain_cache_fn():
        return GainCache()

    return objective_fn, init_fn, descent_fn, qubo_fn, gain_cache_fn


def make_maxsat_problem(clauses, n_vars):
    """
    Returns problem-definition functions for Max-SAT.
    clauses: list of lists, each clause is a list of literals
             (positive integer = variable, negative = negated)
    n_vars: number of boolean variables (1-indexed)

    This is a SKELETON — not tested. Flagged for verification.
    """
    from src.gain_cache import GainCache

    variables = list(range(1, n_vars + 1))

    def objective_fn(x):
        satisfied = 0
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                val = x[v]
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    satisfied += 1
                    break
        return float(satisfied)

    def init_fn(rng):
        return {v: int(rng.integers(0, 2)) for v in variables}

    def descent_fn(x, gc):
        # one-flip local search using gain cache
        gc.update_general(x, objective_fn, variables)
        n_flips = 0
        while True:
            best_v, best_gain = gc.best_flip()
            if best_gain <= 0:
                return x, gc, True, n_flips
            x[best_v] = 1 - x[best_v]
            gc.invalidate()
            gc.update_general(x, objective_fn, variables)
            n_flips += 1
        return x, gc, True, n_flips

    def qubo_fn(x, S):
        # QUBO for Max-SAT subproblem over variables in S
        # This is a placeholder -- proper QUBO encoding needed
        # [INFERENCE -- not verified, flag before use]
        Q = {}
        S_set = set(S)
        for clause in clauses:
            clause_vars = [abs(lit) for lit in clause if abs(lit) in S_set]
            if not clause_vars:
                continue
            # penalty: add coupling terms for unsatisfied clause
            for i, vi in enumerate(clause_vars):
                for vj in clause_vars[i+1:]:
                    Q[(vi, vj)] = Q.get((vi, vj), 0) - 1.0
        return Q

    def gain_cache_fn():
        return GainCache()

    return objective_fn, init_fn, descent_fn, qubo_fn, gain_cache_fn


def adaptive_qls_general(
        variables,
        objective_fn,
        init_fn,
        descent_fn,
        qubo_fn,
        gain_cache_fn,
        selector,
        backend,
        G=None,
        budget_seconds=30,
        k_min=5,
        k_max=30,
        n_reads=100,
        best_known=None,
        seed=None,
        acceptance='improvement',
        T_initial=2.0,
        T_min=0.001,
        cooling=0.995):
    """
    Generalised Adaptive QLS for arbitrary QUBO problems.

    Parameters
    ----------
    variables     : list — variable names (any hashable)
    objective_fn  : callable(x) -> float — objective to MAXIMISE
    init_fn       : callable(rng) -> x dict — random initialisation
    descent_fn    : callable(x, gc) -> (x, gc, at_opt, n_flips)
    qubo_fn       : callable(x, S) -> Q dict — build sub-QUBO
    gain_cache_fn : callable() -> GainCache — create fresh gain cache
    selector      : callable(G, gc, k, pool, rng, x) -> S
    backend       : callable(Q, S, n_reads, seed) -> x_local dict
    G             : NetworkX graph or None (needed for spectral selectors)
    budget_seconds: float
    k_min, k_max  : int — neighbourhood size bounds
    n_reads       : int — solver shots
    best_known    : float or None
    seed          : int or None
    acceptance    : 'improvement', 'sa_cooling', 'fixed_temp', 'lookahead'

    Returns
    -------
    x_best  : dict
    metrics : Metrics
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics()
    metrics.start()

    nodes = variables

    # ── pool warm-start ───────────────────────────────────────────
    pool = []
    warmstart_deadline = time.time() + 0.2 * budget_seconds
    for i in range(5):
        if time.time() > warmstart_deadline:
            break
        x_init = init_fn(rng)
        gc_temp = gain_cache_fn()
        x_opt, gc_temp, _, _ = descent_fn(x_init, gc_temp)
        pool.append(dict(x_opt))
    if not pool:
        pool.append(dict(init_fn(rng)))

    x = max(pool, key=objective_fn)
    x = dict(x)
    x_best = dict(x)
    best_val = objective_fn(x)

    # ── adaptive state ────────────────────────────────────────────
    gc = gain_cache_fn()
    k = k_min
    T = T_initial
    ema_esc = 0.5
    alpha = 0.3
    k_window = []
    k_cooldown = 0

    while time.time() - metrics.start_time < budget_seconds:

        # Phase 1: descent
        x, gc, at_opt, n_flips = descent_fn(x, gc)
        current_val = objective_fn(x)
        metrics.record_cut(current_val)

        if current_val > best_val:
            best_val = current_val
            x_best = dict(x)
            if best_known:
                metrics.check_time_to_target(best_val, best_known)

        if time.time() - metrics.start_time >= budget_seconds:
            break

        # Phase 2: EMA trigger
        if not at_opt and ema_esc >= 0.2:
            continue

        # Phase 3: plateau detection
        eps = 1e-6
        frac_zero = sum(
            1 for v in nodes if abs(gc.gain.get(v, 0)) < eps
        ) / len(nodes)

        # Phase 4: selector
        k_actual = min(k, len(nodes))
        try:
            S = selector(G, gc, k_actual, pool=pool, rng=rng, x=x)
        except Exception as e:
            import warnings
            warnings.warn("selector failed: " + str(e) + " -- using random")
            S = list(rng.choice(nodes, size=k_actual, replace=False))

        # Phase 5: sub-QUBO solve
        Q = qubo_fn(x, S)
        if not Q or all(abs(v) < 1e-10 for v in Q.values()):
            continue
        call_seed = int(rng.integers(0, 2**31 - 1))
        x_local = backend(Q, S, n_reads=n_reads, seed=call_seed)
        x_prop = dict(x)
        x_prop.update({v: x_local[v] for v in S if v in x_local})

        # Phase 5b: look-ahead
        if acceptance in ('lookahead', 'walk'):
            gc_temp = gain_cache_fn()
            x_prop, gc_temp, _, _ = descent_fn(x_prop, gc_temp)

        delta = objective_fn(x_prop) - current_val

        sel_name = getattr(selector, '__name__', 'unknown')
        metrics.record_qls_call(
            delta=delta,
            k=k_actual,
            paradigm='adaptive_qls_general',
            plateau_frac=frac_zero,
            selector=sel_name
        )

        # Phase 6: acceptance
        accepted = False
        if delta > 0:
            accepted = True
        elif acceptance == 'sa_cooling':
            if T > T_min:
                prob = float(np.exp(delta / T))
                if rng.random() < prob:
                    accepted = True
        elif acceptance == 'fixed_temp':
            gamma = 1.0 / T_initial
            prob = float(np.exp(-gamma * abs(delta)))
            if rng.random() < prob:
                accepted = True

        if accepted:
            x = x_prop
            gc.invalidate()
            pool.append(dict(x))
            if len(pool) > 20:
                pool.pop(0)
        else:
            if ema_esc < 0.05:
                top5 = sorted(pool, key=objective_fn, reverse=True)[:5]
                pool = list(top5)
                x = init_fn(rng)
                gc.invalidate()

        if acceptance == 'sa_cooling':
            T = max(T_min, T * cooling)

        # Phase 7: EMA update
        ema_esc = alpha * float(delta > 0) + (1 - alpha) * ema_esc

        # Phase 8: k update
        k_window.append(delta)
        k_cooldown -= 1
        if len(k_window) >= 10 and k_cooldown <= 0:
            w = k_window[-9:]
            t0 = np.mean(w[0:3])
            t1 = np.mean(w[3:6])
            t2 = np.mean(w[6:9])
            if t0 < t1 < t2:
                k = max(k_min, k - 2)
                k_cooldown = 5
            elif t0 > t1 > t2:
                k = min(k_max, k + 2)
                k_cooldown = 5
            k_window = []

    return x_best, metrics
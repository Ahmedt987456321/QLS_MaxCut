"""
Classical baselines for fair comparison against QLS.
- Simulated Annealing (improved: auto-calibrated T, linear-in-beta,
  gain-weighted selection, reheating)
- Tabu Search for Max-Cut
"""

import time
import numpy as np
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, compute_cut_value, random_cut
from src.metrics import Metrics


# ─────────────────────────────────────────────────────────────────
# SA helper functions
# ─────────────────────────────────────────────────────────────────

def _calibrate_T_initial(G, x, gc, chi0=0.80, n_samples=200):
    """
    Ben-Ameur 2004 method — set T_initial so that chi0 fraction
    of worsening moves are accepted at the start.

    Parameters
    ----------
    G        : NetworkX graph
    x        : dict — current assignment
    gc       : GainCache (must be valid)
    chi0     : float — target acceptance probability (0.80 recommended)
    n_samples: int — number of random moves to sample

    Returns
    -------
    T_initial : float
    """
    nodes = list(G.nodes())
    rng = np.random.default_rng()

    # collect worsening move deltas
    bad_deltas = []
    for _ in range(n_samples):
        v = nodes[int(rng.integers(0, len(nodes)))]
        delta = gc.gain[v]
        if delta < 0:
            bad_deltas.append(abs(delta))

    if not bad_deltas:
        return 1.0

    # T such that mean acceptance of bad moves = chi0
    # chi0 = exp(-mean_bad_delta / T) => T = -mean_bad / ln(chi0)
    mean_bad = np.mean(bad_deltas)
    T = -mean_bad / np.log(chi0) if chi0 > 0 else mean_bad
    return max(T, 0.01)


def _gain_weighted_flip(G, x, gc, rng, T):
    """
    Gain-weighted vertex selection — considers best-gain vertex
    and a random vertex, picks better one with SA acceptance.
    More efficient than pure random selection.

    Returns delta (change in cut value) and vertex flipped.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # mix: 70% best-gain, 30% random (prevents pure greedy trapping)
    if rng.random() < 0.70:
        # pick from top-10% gain vertices
        top_k = max(1, n // 10)
        candidates = sorted(nodes,
                            key=lambda v: gc.gain[v],
                            reverse=True)[:top_k]
        v = candidates[int(rng.integers(0, len(candidates)))]
    else:
        v = nodes[int(rng.integers(0, n))]

    delta = gc.gain[v]
    return v, delta


# ─────────────────────────────────────────────────────────────────
# Improved Simulated Annealing
# ─────────────────────────────────────────────────────────────────

def simulated_annealing(G, budget_seconds, best_known=None, seed=None,
                        chi0=0.80, beta_final_multiplier=5.0,
                        reheat_threshold=10, max_reheats=3):
    """
    Improved simulated annealing for Max-Cut.

    Improvements over basic SA:
    1. Auto-calibrated T_initial (Ben-Ameur 2004)
    2. Linear-in-beta schedule across full budget
    3. Gain-weighted vertex selection
    4. Reheating when stagnated

    Parameters
    ----------
    G                    : NetworkX graph
    budget_seconds       : float
    best_known           : float or None
    seed                 : int or None
    chi0                 : float — target initial acceptance (0.80)
    beta_final_multiplier: float — beta_final = beta_initial * multiplier
    reheat_threshold     : int — reheat after n*N non-improving flips
    max_reheats          : int — maximum reheats per run

    Returns
    -------
    x_best  : dict
    metrics : Metrics
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics()
    metrics.start()

    nodes = list(G.nodes())
    n = len(nodes)

    # initialise
    x = random_cut(G, rng)
    gc = GainCache()
    gc.update(G, x)

    x_best = dict(x)
    best_cut = compute_cut_value(G, x)
    current_cut = best_cut

    # ── Step 1: calibrate T_initial ───────────────────────────────
    T_initial = _calibrate_T_initial(G, x, gc, chi0=chi0)
    T = T_initial
    beta = 1.0 / T

    # ── Step 2: set beta schedule ─────────────────────────────────
    beta_initial = beta
    beta_final = beta_initial * beta_final_multiplier
    # will interpolate linearly across budget

    # reheating state
    non_improving_flips = 0
    reheat_count = 0
    reheat_trigger = reheat_threshold * n

    metrics.record_cut(current_cut)

    while time.time() - metrics.start_time < budget_seconds:
        # ── linear-in-beta schedule ───────────────────────────────
        elapsed_frac = (time.time() - metrics.start_time) / budget_seconds
        beta = beta_initial + (beta_final - beta_initial) * elapsed_frac
        T = 1.0 / beta if beta > 0 else T_initial

        # ── gain-weighted flip ────────────────────────────────────
        v, delta = _gain_weighted_flip(G, x, gc, rng, T)

        # ── Metropolis acceptance ─────────────────────────────────
        if delta > 0:
            # always accept improvements
            x[v] = 1 - x[v]
            gc.incremental_update(G, x, v)
            current_cut += delta
            non_improving_flips = 0

            if current_cut > best_cut:
                best_cut = current_cut
                x_best = dict(x)
                if best_known:
                    metrics.check_time_to_target(best_cut, best_known)

        elif T > 1e-10:
            # probabilistic acceptance of worsening moves
            prob = np.exp(delta / T)
            if rng.random() < prob:
                x[v] = 1 - x[v]
                gc.incremental_update(G, x, v)
                current_cut += delta
            non_improving_flips += 1
        else:
            non_improving_flips += 1

        # ── reheating ─────────────────────────────────────────────
        if (non_improving_flips >= reheat_trigger and
                reheat_count < max_reheats):
            T = min(T * 10, T_initial * 0.5)
            beta = 1.0 / T
            # reset beta_initial for remainder of run
            remaining = 1.0 - elapsed_frac
            if remaining > 0.05:
                beta_initial = beta
                beta_final = beta * beta_final_multiplier
            non_improving_flips = 0
            reheat_count += 1

        metrics.record_cut(current_cut)

    return x_best, metrics


# ─────────────────────────────────────────────────────────────────
# Tabu Search
# ─────────────────────────────────────────────────────────────────

def tabu_search(G, budget_seconds, tabu_tenure=None,
                best_known=None, seed=None):
    """
    Tabu search for Max-Cut.
    Uses short-term memory to avoid cycling.

    Parameters
    ----------
    G              : NetworkX graph
    budget_seconds : float
    tabu_tenure    : int or None — scales with n if not specified
    best_known     : float or None
    seed           : int or None

    Returns
    -------
    x_best  : dict
    metrics : Metrics
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics()
    metrics.start()

    x = random_cut(G, rng)
    gc = GainCache()
    gc.update(G, x)

    x_best = dict(x)
    best_cut = compute_cut_value(G, x)
    current_cut = best_cut

    nodes = list(G.nodes())
    n = len(nodes)

    # scale tabu tenure with graph size
    if tabu_tenure is None:
        tabu_tenure = max(10, n // 20)

    # tabu list
    tabu = {v: 0 for v in nodes}
    iteration = 0

    metrics.record_cut(current_cut)

    while time.time() - metrics.start_time < budget_seconds:
        iteration += 1

        # find best non-tabu move (with aspiration)
        best_v = None
        best_gain = float('-inf')

        for v in nodes:
            gain = gc.gain[v]
            is_tabu = tabu[v] > iteration

            # aspiration: override tabu if improves best known
            if is_tabu and (current_cut + gain) <= best_cut:
                continue

            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v is None:
            best_v = min(nodes, key=lambda v: tabu[v])
            best_gain = gc.gain[best_v]

        # apply move
        x[best_v] = 1 - x[best_v]
        current_cut += best_gain
        gc.incremental_update(G, x, best_v)

        # update tabu list
        tabu[best_v] = iteration + tabu_tenure

        # update best
        if current_cut > best_cut:
            best_cut = current_cut
            x_best = dict(x)
            if best_known:
                metrics.check_time_to_target(best_cut, best_known)

        metrics.record_cut(current_cut)

    return x_best, metrics
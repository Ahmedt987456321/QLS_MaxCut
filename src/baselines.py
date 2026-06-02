"""
Classical baselines for fair comparison against QLS.
- Simulated Annealing (improved: auto-calibrated T, linear-in-beta,
  gain-weighted selection, reheating)
- Tabu Search for Max-Cut
- Breakout Local Search (Benlic & Hao 2013) — FIXED
"""

import time
import numpy as np
from src.gain_cache import GainCache
from src.metrics import Metrics
from src.local_search import one_flip_ls, compute_cut_value, random_cut, one_flip_ls_with_plateau
 
# ─────────────────────────────────────────────────────────────────
# SA helper functions
# ─────────────────────────────────────────────────────────────────
 
def _calibrate_T_initial(G, x, gc, chi0=0.80, n_samples=200):
    """
    Ben-Ameur 2004 method — set T_initial so that chi0 fraction
    of worsening moves are accepted at the start.
    """
    nodes = list(G.nodes())
    rng = np.random.default_rng()
 
    bad_deltas = []
    for _ in range(n_samples):
        v = nodes[int(rng.integers(0, len(nodes)))]
        delta = gc.gain[v]
        if delta < 0:
            bad_deltas.append(abs(delta))
 
    if not bad_deltas:
        return 1.0
 
    mean_bad = np.mean(bad_deltas)
    T = -mean_bad / np.log(chi0) if chi0 > 0 else mean_bad
    return max(T, 0.01)
 
 
def _gain_weighted_flip(G, x, gc, rng, T):
    """
    Gain-weighted vertex selection.
    """
    nodes = list(G.nodes())
    n = len(nodes)
 
    if rng.random() < 0.70:
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
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics()
    metrics.start()
 
    nodes = list(G.nodes())
    n = len(nodes)
 
    x = random_cut(G, rng)
    gc = GainCache()
    gc.update(G, x)
 
    x_best = dict(x)
    best_cut = compute_cut_value(G, x)
    current_cut = best_cut
 
    T_initial = _calibrate_T_initial(G, x, gc, chi0=chi0)
    T = T_initial
    beta = 1.0 / T
 
    beta_initial = beta
    beta_final = beta_initial * beta_final_multiplier
 
    non_improving_flips = 0
    reheat_count = 0
    reheat_trigger = reheat_threshold * n
 
    metrics.record_cut(current_cut)
 
    while time.time() - metrics.start_time < budget_seconds:
        elapsed_frac = (time.time() - metrics.start_time) / budget_seconds
        beta = beta_initial + (beta_final - beta_initial) * elapsed_frac
        T = 1.0 / beta if beta > 0 else T_initial
 
        v, delta = _gain_weighted_flip(G, x, gc, rng, T)
 
        if delta > 0:
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
            prob = np.exp(delta / T)
            if rng.random() < prob:
                x[v] = 1 - x[v]
                gc.incremental_update(G, x, v)
                current_cut += delta
            non_improving_flips += 1
        else:
            non_improving_flips += 1
 
        if (non_improving_flips >= reheat_trigger and
                reheat_count < max_reheats):
            T = min(T * 10, T_initial * 0.5)
            beta = 1.0 / T
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
 
    if tabu_tenure is None:
        tabu_tenure = max(10, n // 20)
 
    tabu = {v: 0 for v in nodes}
    iteration = 0
 
    metrics.record_cut(current_cut)
 
    while time.time() - metrics.start_time < budget_seconds:
        iteration += 1
 
        best_v = None
        best_gain = float('-inf')
 
        for v in nodes:
            gain = gc.gain[v]
            is_tabu = tabu[v] > iteration
 
            if is_tabu and (current_cut + gain) <= best_cut:
                continue
 
            if gain > best_gain:
                best_gain = gain
                best_v = v
 
        if best_v is None:
            best_v = min(nodes, key=lambda v: tabu[v])
            best_gain = gc.gain[best_v]
 
        x[best_v] = 1 - x[best_v]
        current_cut += best_gain
        gc.incremental_update(G, x, best_v)
 
        tabu[best_v] = iteration + tabu_tenure
 
        if current_cut > best_cut:
            best_cut = current_cut
            x_best = dict(x)
            if best_known:
                metrics.check_time_to_target(best_cut, best_known)
 
        metrics.record_cut(current_cut)
 
    return x_best, metrics
 
 
# ─────────────────────────────────────────────────────────────────
# Breakout Local Search — Benlic & Hao 2013
# ─────────────────────────────────────────────────────────────────
 
def breakout_local_search(G, budget_seconds, best_known=None, seed=None):
    """
    Breakout Local Search for Max-Cut.
    Benlic & Hao 2013 — Engineering Applications of AI 26(3):1162-1173.
 
    FIX SUMMARY vs previous version:
    - Removed gc = GainCache() reset before local search
      (one_flip_ls already calls gc.update internally, and the
       perturbation operators maintain gc via incremental_update)
    - Fixed omega / L update logic:
        omega increments every cycle (not just on non-improvement)
        omega resets only when best_cut is improved
        L resets on new attractor (not just on improvement)
        L escalates on attractor revisit
    - Fixed T_stag warm-up: proper gc initialisation each cycle
    - Fixed attractor detection ordering: happens before L update
    - M2 now updates current_cut correctly
    - M1/M2/M3 no longer need to recompute cut — caller does it
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics()
    metrics.start()
 
   
    nodes = list(G.nodes())
    n = len(nodes)

    # detect if graph has ±1 weights (plateau search needed)
    # or +1 only weights (plain descent is faster)
    sample_weights = [d.get('weight', 1.0)
                      for _, _, d in list(G.edges(data=True))[:20]]
    has_negative_weights = any(w < 0 for w in sample_weights)
    plateau_budget = None if has_negative_weights else 0

    # ── BLS parameters (Benlic & Hao Table 1) ────────────────────
    # ── BLS parameters (Benlic & Hao Table 1) ────────────────────
    L0        = max(1, n // 100)          # 0.01 * n
    L_max     = max(L0 * 5, n // 10)      # cap: prevents budget starvation
    P0        = 0.8                        # min prob of directed perturbation
    Q_split   = 0.5                        # M1 vs M2 split
 
    # ── adaptive T_stag: measure LO/s, target 2.5s stagnation window ─
    # Warm-up OUTSIDE the budget timer.
    # one_flip_ls calls gc.update internally so we don't need to
    # pre-populate gc here.
    _t0 = time.time()
    _x_tmp = random_cut(G, rng)
    _gc_tmp = GainCache()
    for _ in range(10):
        _x_tmp, _gc_tmp, _, _ = one_flip_ls_with_plateau(G, _x_tmp, _gc_tmp, max_plateau=plateau_budget)
        _x_tmp = random_cut(G, rng)          # new random start
        # _gc_tmp will be re-populated by next one_flip_ls call
    _elapsed = time.time() - _t0
    _lo_per_sec = 10.0 / max(_elapsed, 1e-6)
    T_stag = max(10, int(np.ceil(2.5 * _lo_per_sec)))
 
    # ── initial solution ─────────────────────────────────────────
    x  = random_cut(G, rng)
    gc = GainCache()
    # descend to local optimum
    x, gc, _, _ = one_flip_ls_with_plateau(G, x, gc, max_plateau=plateau_budget)
    x_best    = dict(x)
    best_cut  = compute_cut_value(G, x)
    current_cut = best_cut
 
    # tabu list: tabu[v] = iteration at which v becomes free again
    tabu      = {v: 0 for v in nodes}
    iteration = 0
 
    # BLS adaptive state
    L      = L0
    omega  = 0           # consecutive non-improving local optima
    prev_x = dict(x)     # previous local optimum for attractor detection
 
    metrics.record_cut(current_cut)
 
    while time.time() - metrics.start_time < budget_seconds:
        iteration += 1
 
        # ── select and apply perturbation ─────────────────────────
        # The perturbation operators apply L moves and maintain gc
        # via incremental_update.  They do NOT run local search.
        p_directed = max(P0, np.exp(-omega / max(T_stag, 1)))
 
        r = rng.random()
        if r < p_directed * Q_split:
            # M1 — directed single flip, tabu-filtered
            x, gc = _bls_m1(G, x, gc, tabu, iteration,
                             best_cut, current_cut, L, nodes, rng)
        elif r < p_directed:
            # M2 — directed swap between partitions
            x, gc = _bls_m2(G, x, gc, tabu, iteration,
                             best_cut, current_cut, L, nodes, rng)
        else:
            # M3 — random perturbation
            x, gc = _bls_m3(G, x, gc, L, nodes, rng)
 
        # ── descent to next local optimum ────────────────────────
        # one_flip_ls calls gc.update(G, x) internally so the cache
        # is always valid on return regardless of perturbation.
        x, gc, _, _ = one_flip_ls_with_plateau(G, x, gc, max_plateau=plateau_budget)
        current_cut  = compute_cut_value(G, x)
 
        # ── update best ───────────────────────────────────────────
        if current_cut > best_cut:
            best_cut = current_cut
            x_best   = dict(x)
            omega    = 0
            if best_known:
                metrics.check_time_to_target(best_cut, best_known)
        else:
            omega += 1
 
        # ── attractor detection → L update ───────────────────────
        # Compare current local optimum with the previous one.
        # Same configuration  → revisiting same attractor → escalate L.
        # New configuration   → escaped to new basin → reset L.
        same_attractor = all(x[v] == prev_x[v] for v in nodes)
        if same_attractor:
            L = min(L + 1, L_max)
        else:
            L = L0
 
        prev_x = dict(x)
 
        # ── stagnation restart ────────────────────────────────────
        # If omega exceeds T_stag, force a fresh random restart.
        if omega >= T_stag:
            x  = random_cut(G, rng)
            gc = GainCache()
            x, gc, _, _ = one_flip_ls_with_plateau(G, x, gc, max_plateau=plateau_budget)
            current_cut  = compute_cut_value(G, x)
            omega = 0
            L     = L0
            tabu  = {v: 0 for v in nodes}
            prev_x = dict(x)
 
        metrics.record_cut(current_cut)
 
    return x_best, metrics
 
 
# ─────────────────────────────────────────────────────────────────
# BLS perturbation operators
# ─────────────────────────────────────────────────────────────────
 
def _bls_m1(G, x, gc, tabu, iteration, best_cut, current_cut, L, nodes, rng):
    """
    M1 — directed single flip.
    Each of the L moves selects the highest-gain non-tabu vertex
    (tabu overridden by aspiration: move accepted if it would improve
    best_cut).  Updates gc incrementally.  Returns (x, gc).
    """
    tenure_max = max(4, len(nodes) // 10)
 
    for _ in range(L):
        best_v    = None
        best_gain = float('-inf')
 
        for v in nodes:
            g       = gc.gain[v]
            is_tabu = tabu[v] > iteration
            # aspiration: allow tabu move if it improves the best cut
            if is_tabu and (current_cut + g) <= best_cut:
                continue
            if g > best_gain:
                best_gain = g
                best_v    = v
 
        if best_v is None:
            # all vertices tabu and aspiration not triggered — pick random
            best_v = nodes[int(rng.integers(0, len(nodes)))]
 
        x[best_v]  = 1 - x[best_v]
        current_cut += gc.gain[best_v]   # track for aspiration in next move
        gc.incremental_update(G, x, best_v)
        tabu[best_v] = iteration + int(rng.integers(3, tenure_max))
 
    return x, gc
 
 
def _bls_m2(G, x, gc, tabu, iteration, best_cut, current_cut, L, nodes, rng):
    """
    M2 — directed swap.
    Each of the L moves swaps the best vertex from partition 0 with
    the best vertex from partition 1 (both tabu-filtered).
    Updates gc incrementally.  Returns (x, gc).
    """
    tenure_max = max(4, len(nodes) // 10)
 
    for _ in range(L):
        # best non-tabu vertex in partition 0
        v0, g0 = None, float('-inf')
        for v in nodes:
            if x[v] == 0 and tabu[v] <= iteration and gc.gain[v] > g0:
                g0, v0 = gc.gain[v], v
 
        # best non-tabu vertex in partition 1
        v1, g1 = None, float('-inf')
        for v in nodes:
            if x[v] == 1 and tabu[v] <= iteration and gc.gain[v] > g1:
                g1, v1 = gc.gain[v], v
 
        if v0 is not None:
            current_cut += gc.gain[v0]
            x[v0] = 1 - x[v0]
            gc.incremental_update(G, x, v0)
            tabu[v0] = iteration + int(rng.integers(3, tenure_max))
 
        if v1 is not None:
            current_cut += gc.gain[v1]
            x[v1] = 1 - x[v1]
            gc.incremental_update(G, x, v1)
            tabu[v1] = iteration + int(rng.integers(3, tenure_max))
 
    return x, gc
 
 
def _bls_m3(G, x, gc, L, nodes, rng):
    """
    M3 — random perturbation.
    Flips L randomly chosen distinct vertices.
    Updates gc incrementally.  Returns (x, gc).
    """
    chosen = rng.choice(len(nodes), size=min(L, len(nodes)), replace=False)
    for idx in chosen:
        v      = nodes[idx]
        x[v]   = 1 - x[v]
        gc.incremental_update(G, x, v)
 
    return x, gc


# -------------------------------------------------------------
# Parallel Tempering (replica exchange) - classical gold standard
# -------------------------------------------------------------

def parallel_tempering(G, budget_seconds, best_known=None, seed=None,
                       n_replicas=8, beta_min=0.1, beta_max=3.0,
                       swap_interval=10):
    """
    Plain Parallel Tempering (replica exchange) for Max-Cut.
    n_replicas at a geometric beta ladder; each does gain-weighted
    Metropolis flips; adjacent replicas swap on the PT criterion.
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics(); metrics.start()
    nodes = list(G.nodes()); n = len(nodes)

    # geometric temperature ladder
    betas = np.geomspace(beta_min, beta_max, n_replicas)

    # one state per replica
    xs = [random_cut(G, rng) for _ in range(n_replicas)]
    gcs = [GainCache() for _ in range(n_replicas)]
    cuts = []
    for r in range(n_replicas):
        gcs[r].update(G, xs[r])
        cuts.append(compute_cut_value(G, xs[r]))

    x_best = dict(xs[0]); best_cut = max(cuts)
    bi = int(np.argmax(cuts)); x_best = dict(xs[bi])
    metrics.record_cut(best_cut)

    step = 0
    while time.time() - metrics.start_time < budget_seconds:
        step += 1
        # one Metropolis sweep-ish per replica (a batch of flips)
        for r in range(n_replicas):
            beta = betas[r]
            for _ in range(max(1, n // 10)):
                v = nodes[int(rng.integers(0, n))]
                delta = gcs[r].gain[v]   # gain = +cut change if flipped
                if delta >= 0 or rng.random() < np.exp(beta * delta):
                    xs[r][v] = 1 - xs[r][v]
                    gcs[r].incremental_update(G, xs[r], v)
                    cuts[r] += delta
                    if cuts[r] > best_cut:
                        best_cut = cuts[r]; x_best = dict(xs[r])
                        if best_known:
                            metrics.check_time_to_target(best_cut, best_known)

        # replica exchange on adjacent pairs
        if step % swap_interval == 0:
            for r in range(n_replicas - 1):
                d_beta = betas[r] - betas[r+1]
                d_cut = cuts[r] - cuts[r+1]
                # swap accept: exp((beta_r - beta_{r+1})(E_{r+1}-E_r));
                # for Max-Cut we maximise cut, so use d_beta * (cut_r - cut_{r+1}) sign
                arg = d_beta * (cuts[r+1] - cuts[r])
                if arg >= 0 or rng.random() < np.exp(arg):
                    xs[r], xs[r+1] = xs[r+1], xs[r]
                    gcs[r], gcs[r+1] = gcs[r+1], gcs[r]
                    cuts[r], cuts[r+1] = cuts[r+1], cuts[r]

        metrics.record_cut(best_cut)

    return x_best, metrics


# -------------------------------------------------------------
# Parallel Tempering (replica exchange) - classical gold standard
# -------------------------------------------------------------

def parallel_tempering(G, budget_seconds, best_known=None, seed=None,
                       n_replicas=8, beta_min=0.1, beta_max=3.0,
                       swap_interval=10):
    """
    Plain Parallel Tempering (replica exchange) for Max-Cut.
    n_replicas at a geometric beta ladder; each does gain-weighted
    Metropolis flips; adjacent replicas swap on the PT criterion.
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics(); metrics.start()
    nodes = list(G.nodes()); n = len(nodes)

    # geometric temperature ladder
    betas = np.geomspace(beta_min, beta_max, n_replicas)

    # one state per replica
    xs = [random_cut(G, rng) for _ in range(n_replicas)]
    gcs = [GainCache() for _ in range(n_replicas)]
    cuts = []
    for r in range(n_replicas):
        gcs[r].update(G, xs[r])
        cuts.append(compute_cut_value(G, xs[r]))

    x_best = dict(xs[0]); best_cut = max(cuts)
    bi = int(np.argmax(cuts)); x_best = dict(xs[bi])
    metrics.record_cut(best_cut)

    step = 0
    while time.time() - metrics.start_time < budget_seconds:
        step += 1
        # one Metropolis sweep-ish per replica (a batch of flips)
        for r in range(n_replicas):
            beta = betas[r]
            for _ in range(max(1, n // 10)):
                v = nodes[int(rng.integers(0, n))]
                delta = gcs[r].gain[v]   # gain = +cut change if flipped
                if delta >= 0 or rng.random() < np.exp(beta * delta):
                    xs[r][v] = 1 - xs[r][v]
                    gcs[r].incremental_update(G, xs[r], v)
                    cuts[r] += delta
                    if cuts[r] > best_cut:
                        best_cut = cuts[r]; x_best = dict(xs[r])
                        if best_known:
                            metrics.check_time_to_target(best_cut, best_known)

        # replica exchange on adjacent pairs
        if step % swap_interval == 0:
            for r in range(n_replicas - 1):
                d_beta = betas[r] - betas[r+1]
                d_cut = cuts[r] - cuts[r+1]
                # swap accept: exp((beta_r - beta_{r+1})(E_{r+1}-E_r));
                # for Max-Cut we maximise cut, so use d_beta * (cut_r - cut_{r+1}) sign
                arg = d_beta * (cuts[r+1] - cuts[r])
                if arg >= 0 or rng.random() < np.exp(arg):
                    xs[r], xs[r+1] = xs[r+1], xs[r]
                    gcs[r], gcs[r+1] = gcs[r+1], gcs[r]
                    cuts[r], cuts[r+1] = cuts[r+1], cuts[r]

        metrics.record_cut(best_cut)

    return x_best, metrics

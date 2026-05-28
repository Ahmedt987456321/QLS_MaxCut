"""
Adaptive QLS — research contribution.
Adaptive trigger (EMA-based), pluggable selector,
momentum-damped k update, pool persistence on restart.
"""

import time
import numpy as np
from src.gain_cache import GainCache
from src.local_search import one_flip_ls, compute_cut_value, random_cut
from src.qubo import build_local_qubo, merge_proposal
from src.metrics import Metrics
from src.gain_cache import GainCache



def adaptive_qls(G, budget_seconds, selector, backend,
                 k_min=5, k_max=30, n_reads=100,
                 best_known=None, seed=None,
                 acceptance='improvement',
                 T_initial=2.0, T_min=0.001, cooling=0.995):
    """
    Adaptive QLS — pluggable selector, EMA trigger, adaptive k.

    Parameters
    ----------
    G              : NetworkX graph
    budget_seconds : float — wall-clock time limit
    selector       : callable — selector(G, gc, k, pool, rng) -> S
    backend        : callable — backend(Q, S, n_reads, init) -> x_local
    k_min          : int — minimum neighbourhood size
    k_max          : int — maximum neighbourhood size
    n_reads        : int — solver shots per call
    best_known     : float or None
    seed           : int or None

    Returns
    -------
    x_best  : dict
    metrics : Metrics
    """
    rng = np.random.default_rng(seed)
    gc = GainCache()
    metrics = Metrics()
    metrics.start()

    nodes = list(G.nodes())

    # ── pool warm-start: 5 diversified cuts ──────────────────────
    pool = []
    for i in range(5):
        x_init = random_cut(G, rng)
        gc_temp = GainCache()
        x_opt, gc_temp, _, _ = one_flip_ls(G, x_init, gc_temp)
        pool.append(dict(x_opt))

    # start from best pool solution
    x = max(pool, key=lambda p: compute_cut_value(G, p))
    x = dict(x)
    x_best = dict(x)
    best_cut = compute_cut_value(G, x)

    # ── adaptive state ────────────────────────────────────────────
    k = k_min
    T = T_initial 
    ema_esc = 0.5          # EMA escape probability — item 12
    alpha = 0.3            # EMA smoothing factor
    k_window = []          # gain window for k update — item 13
    k_cooldown = 0         # prevent k oscillation
    k_trend_count = 0      # consecutive consistent trends needed

    while time.time() - metrics.start_time < budget_seconds:

        # ── Phase 1: classical descent ────────────────────────────
        x, gc, at_opt, n_flips = one_flip_ls(G, x, gc)
        current_cut = compute_cut_value(G, x)
        metrics.record_cut(current_cut)

        if current_cut > best_cut:
            best_cut = current_cut
            x_best = dict(x)
            if best_known:
                metrics.check_time_to_target(best_cut, best_known)

        if time.time() - metrics.start_time >= budget_seconds:
            break

        # ── Phase 2: EMA-based adaptive trigger ───────────────────
        # only call solver at local optimum OR when escape rate low
        if not at_opt and ema_esc >= 0.2:
            continue

        # ── Phase 3: plateau detection metric ─────────────────────
        eps = 1e-6
        frac_zero = sum(
            1 for v in nodes if abs(gc.gain[v]) < eps
        ) / len(nodes)

        # ── Phase 4: pluggable selector ───────────────────────────
        k_actual = min(k, len(nodes))
        try:
            S = selector(G, gc, k_actual, pool=pool, rng=rng)
        except Exception:
            # fallback to random if selector fails
            S = list(rng.choice(nodes, size=k_actual, replace=False))

    

        # ── Phase 5: local solve ──────────────────────────────────
        Q = build_local_qubo(G, x, S)
        if not Q or all(abs(v) < 1e-10 for v in Q.values()):
            continue
        x_local = backend(Q, S, n_reads=n_reads)
        x_prop = merge_proposal(x, x_local, S)

        # ── Phase 5b: look-ahead — run descent before accepting ───
        if acceptance == 'lookahead':
            gc_temp = GainCache()
            x_prop, gc_temp, _, _ = one_flip_ls(G, x_prop, gc_temp)

        delta = compute_cut_value(G, x_prop) - current_cut

        sel_name = getattr(selector, '__name__', 'unknown')
        metrics.record_qls_call(
            delta=delta,
            k=k_actual,
            paradigm='adaptive_qls',
            plateau_frac=frac_zero,
            selector=sel_name
        )

           

        
        # ── Phase 6: acceptance + pool update ─────────────────────
        accepted = False

        if delta > 0:
            # always accept improvements
            accepted = True
        elif acceptance == 'sa_cooling':
            # SA cooling schedule — accept worsening with temperature prob
            if T > T_min:
                prob = float(np.exp(delta / T))
                if rng.random() < prob:
                    accepted = True
        elif acceptance == 'fixed_temp':
            # Liu & Goan style — fixed temperature Metropolis
            gamma = 1.0 / T_initial
            prob = float(np.exp(-gamma * abs(delta)))
            if rng.random() < prob:
                accepted = True
        # else: improvement_only — accepted stays False

        if accepted:
            x = x_prop
            gc.invalidate()
            pool.append(dict(x))
            if len(pool) > 20:
                pool.pop(0)
        else:
            # pool persistence on restart — keep top-5
            if ema_esc < 0.05:
                top5 = sorted(pool,
                    key=lambda p: compute_cut_value(G, p),
                    reverse=True)[:5]
                pool = list(top5)
                x = random_cut(G, rng)
                gc.invalidate()

        # ── Phase 6b: cool the temperature ────────────────────────
        if acceptance == 'sa_cooling':
            T = max(T_min, T * cooling)

        # ── Phase 7: EMA update ───────────────────────────────────
        ema_esc = alpha * float(delta > 0) + (1 - alpha) * ema_esc

        # ── Phase 8: momentum-damped k update ─────────────────────
        k_window.append(delta)
        k_cooldown -= 1

        if len(k_window) >= 10 and k_cooldown <= 0:
            # split window into 3 sub-windows of 3
            w = k_window[-9:]
            t0 = np.mean(w[0:3])
            t1 = np.mean(w[3:6])
            t2 = np.mean(w[6:9])

            if t0 < t1 < t2:
                # consistent improvement trend — shrink k
                k = max(k_min, k - 2)
                k_cooldown = 5
            elif t0 > t1 > t2:
                # consistent worsening trend — grow k
                k = min(k_max, k + 2)
                k_cooldown = 5

            k_window = []  # reset window after decision

    return x_best, metrics
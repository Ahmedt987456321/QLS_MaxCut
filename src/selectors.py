"""
Neighbourhood selectors for Adaptive QLS.
All selectors share the same interface:
    selector(G, gc, k, pool=None) -> S (list of vertices)
"""

import numpy as np
from src.gain_cache import GainCache


def _plateau_aware_rank(G, gc, mode):
    """
    Rank vertices by gain with plateau detection.
    If >80% of vertices have |gain| near zero (plateau),
    use degree as a tiebreaker.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache (must be valid)
    mode : 'frustrated' (ascending |gain|) or 'impact' (descending |gain|)

    Returns
    -------
    ranked : list of vertices in order
    """
    gc.assert_valid()
    nodes = list(G.nodes())
    eps = 1e-6

    frac_zero = sum(1 for v in nodes if abs(gc.gain[v]) < eps) / len(nodes)

    if frac_zero > 0.8:
        # plateau detected — use degree as tiebreaker
        if mode == 'frustrated':
            ranked = sorted(nodes,
                key=lambda v: (abs(gc.gain[v]), -G.degree(v)))
        else:
            ranked = sorted(nodes,
                key=lambda v: (-abs(gc.gain[v]), -G.degree(v)))
    else:
        if mode == 'frustrated':
            ranked = sorted(nodes, key=lambda v: abs(gc.gain[v]))
        else:
            ranked = sorted(nodes, key=lambda v: -abs(gc.gain[v]))

    return ranked


def select_random(G, gc, k, pool=None, rng=None):
    """
    Random selector — uniform sample of k vertices.
    Baseline: Tomesh et al. 2022 pattern.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : ignored
    rng  : numpy.random.Generator or None

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())
    k_actual = min(k, len(nodes))
    return list(rng.choice(nodes, size=k_actual, replace=False))


def select_frustrated(G, gc, k, pool=None, rng=None):
    """
    Frustrated selector — vertices with |gain| closest to zero.
    Novel proposal: locally indifferent vertices may unlock
    multi-flip improvements that one-flip search cannot find.
    Motivated by BLS directed perturbation evidence.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : ignored
    rng  : ignored (deterministic)

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    ranked = _plateau_aware_rank(G, gc, mode='frustrated')
    return ranked[:min(k, len(ranked))]

def select_frustrated_connected(G, gc, k, pool=None, rng=None):
    """
    Connected frustrated selector — novel contribution.
    Seeds from the most frustrated vertex then grows a connected
    subgraph by adding neighbours with lowest |gain|.

    This ensures the QUBO subproblem has meaningful internal edges,
    fixing the disconnected-subgraph problem of plain frustrated selection.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : ignored
    rng  : numpy.random.Generator or None

    Returns
    -------
    S : list of k vertices forming a connected subgraph
    """
    gc.assert_valid()
    nodes = list(G.nodes())
    eps = 1e-6

    if len(nodes) <= k:
        return nodes

    # find seed — vertex with |gain| closest to zero
    # with degree tiebreaker for plateau detection
    frac_zero = sum(1 for v in nodes if abs(gc.gain[v]) < eps) / len(nodes)

    if frac_zero > 0.8:
        # plateau — seed from highest degree frustrated vertex
        seed = min(nodes, key=lambda v: (abs(gc.gain[v]), -G.degree(v)))
    else:
        seed = min(nodes, key=lambda v: abs(gc.gain[v]))

    # grow connected subgraph greedily
    S = [seed]
    frontier = set(G.neighbors(seed))  # vertices adjacent to S

    while len(S) < k and frontier:
        # pick frontier vertex with lowest |gain|
        best = min(frontier, key=lambda v: abs(gc.gain[v]))
        S.append(best)

        # expand frontier with new neighbours not already in S
        for u in G.neighbors(best):
            if u not in S:
                frontier.add(u)
        frontier.discard(best)

    # if graph is disconnected and frontier exhausted before k
    # fill remaining slots with global frustrated vertices
    if len(S) < k:
        remaining = [v for v in nodes if v not in set(S)]
        remaining_sorted = sorted(remaining,
                                  key=lambda v: abs(gc.gain[v]))
        S.extend(remaining_sorted[:k - len(S)])

    return S[:k]
def select_impact(G, gc, k, pool=None, rng=None):
    """
    Impact selector — vertices with largest |gain|.
    Literature comparator: qbsolv / Atobe 2022 style.
    Selects vertices with the highest single-flip impact.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : ignored
    rng  : ignored (deterministic)

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    ranked = _plateau_aware_rank(G, gc, mode='impact')
    return ranked[:min(k, len(ranked))]


def select_clustering(G, gc, k, pool=None, rng=None):
    """
    Clustering selector — correlation-based subgraph selection.
    Based on Zhao & Tang 2025.
    Falls back to select_frustrated if pool is too small.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : list of dicts — recent incumbent solutions
    rng  : numpy.random.Generator or None

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    min_pool = 5

    # pool size guard — fallback if not enough solutions
    if pool is None or len(pool) < min_pool:
        return select_frustrated(G, gc, k, pool=pool, rng=rng)

    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    # build correlation matrix from solution pool
    # C[i,j] = fraction of pool solutions where x[i] == x[j]
    pool_array = np.array(
        [[p[v] for v in nodes] for p in pool],
        dtype=float
    )

    # use cached clusters if available
    if (hasattr(select_clustering, '_cache') and
            select_clustering._cache.get('pool_size') == len(pool)):
        clusters = select_clustering._cache['clusters']
    else:
        try:
            from sklearn.cluster import SpectralClustering
            n_clusters = max(2, n // k)
            n_clusters = min(n_clusters, len(pool))

            # correlation as similarity
            C = np.corrcoef(pool_array.T)
            C = (C + 1) / 2  # normalise to [0, 1]

            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=3
            )
            labels = sc.fit_predict(C)

            clusters = {}
            for i, v in enumerate(nodes):
                c = labels[i]
                if c not in clusters:
                    clusters[c] = []
                clusters[c].append(v)

            # cache result
            select_clustering._cache = {
                'pool_size': len(pool),
                'clusters': clusters
            }

        except Exception:
            # clustering failed — fallback to frustrated
            return select_frustrated(G, gc, k, pool=pool, rng=rng)

    # pick cluster with highest total |gain|
    best_cluster_id = max(
        clusters.keys(),
        key=lambda c: sum(abs(gc.gain[v]) for v in clusters[c])
    )
    best_cluster = clusters[best_cluster_id]

    if rng is None:
        rng = np.random.default_rng()

    k_actual = min(k, len(best_cluster))
    return list(rng.choice(best_cluster, size=k_actual, replace=False))


def select_meta_rule(G, gc, k, pool=None, rng=None):
    """
    Rule-based meta-selector.
    Switches between selectors based on current search state.

    Rules:
    - plateau (>80% zero gain)  -> impact + degree tiebreaker
    - low pool diversity        -> frustrated
    - pool large enough         -> clustering
    - default                   -> frustrated
    """
    gc.assert_valid()
    nodes = list(G.nodes())
    eps = 1e-6

    frac_zero = sum(1 for v in nodes if abs(gc.gain[v]) < eps) / len(nodes)

    # rule 1: plateau detected
    if frac_zero > 0.8:
        return select_impact(G, gc, k, pool=pool, rng=rng)

    # rule 2: pool diversity check
    if pool is not None and len(pool) >= 2:
        pool_array = np.array(
            [[p[v] for v in nodes] for p in pool], dtype=float
        )
        # mean pairwise hamming distance
        n_pool = len(pool_array)
        diffs = 0
        count = 0
        for i in range(n_pool):
            for j in range(i + 1, n_pool):
                diffs += np.sum(pool_array[i] != pool_array[j])
                count += len(nodes)
        diversity = diffs / count if count > 0 else 1.0

        if diversity < 0.1:
            return select_frustrated(G, gc, k, pool=pool, rng=rng)

    # rule 3: clustering if pool large enough
    if pool is not None and len(pool) >= 5:
        return select_clustering(G, gc, k, pool=pool, rng=rng)

    # default
    return select_frustrated(G, gc, k, pool=pool, rng=rng)


def get_selector(name):
    """
    Factory function — returns selector by name.

    Parameters
    ----------
    name : str — 'random', 'frustrated', 'impact',
                 'clustering', 'meta_rule'

    Returns
    -------
    selector function
    """
    selectors = {
        'random':               select_random,
        'frustrated':           select_frustrated,
        'frustrated_connected': select_frustrated_connected,
        'impact':               select_impact,
        'clustering':           select_clustering,
        'meta_rule':            select_meta_rule,
    }
    
    if name not in selectors:
        raise ValueError(
            f"Unknown selector '{name}'. "
            f"Choose from: {list(selectors.keys())}"
        )
    return selectors[name]
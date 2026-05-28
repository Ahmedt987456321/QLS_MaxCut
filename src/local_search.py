"""
OneFlipLS — best-improvement one-flip local search for Max-Cut.
Returns the solution, updated GainCache, and a flag indicating
whether a one-flip local optimum was reached.
"""

from src.gain_cache import GainCache


def compute_cut_value(G, x):
    """Total weight of edges crossing the cut."""
    cut = 0.0
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        if x[u] != x[v]:
            cut += w
    return cut


def one_flip_ls(G, x, gc, max_iter=None):
    """
    Best-improvement one-flip local search.

    Parameters
    ----------
    G        : NetworkX graph
    x        : dict {vertex: 0 or 1} — current assignment
    gc       : GainCache instance
    max_iter : int or None — iteration budget (None = run to local optimum)

    Returns
    -------
    x        : improved assignment
    gc       : updated GainCache
    at_opt   : True if stopped at one-flip local optimum
    n_flips  : number of flips performed
    """
    x = dict(x)          # work on a copy
    gc.update(G, x)      # always recompute on entry

    n_flips = 0
    iter_count = 0

    while True:
        if max_iter is not None and iter_count >= max_iter:
            return x, gc, False, n_flips

        best_v, best_gain = gc.best_flip()

        if best_gain <= 0:
            # no improving flip exists — at local optimum
            return x, gc, True, n_flips

        # apply the best flip
        x[best_v] = 1 - x[best_v]
        gc.incremental_update(G, x, best_v)

        n_flips += 1
        iter_count += 1


def random_cut(G, rng=None):
    """
    Generate a random binary assignment for all vertices.

    Parameters
    ----------
    G   : NetworkX graph
    rng : numpy.random.Generator or None

    Returns
    -------
    x : dict {vertex: 0 or 1}
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    return {v: int(rng.integers(0, 2)) for v in G.nodes()}
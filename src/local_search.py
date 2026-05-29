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


def one_flip_ls_with_plateau(G, x, gc, max_plateau=None):
    """
    Best-improvement local search with plateau moves for +-1 graphs.

    If max_plateau=0, behaves exactly like one_flip_ls (no plateau moves).
    This allows BLS to use plain descent on +1 weight graphs (G14, G22)
    and plateau search on +-1 weight graphs (G11).

    Parameters
    ----------
    G           : NetworkX graph
    x           : dict {vertex: 0 or 1}
    gc          : GainCache instance
    max_plateau : int or None
                  None  → default n*5 plateau budget (for ±1 graphs)
                  0     → no plateau moves (plain descent, for +1 graphs)
                  k>0   → exactly k plateau moves allowed

    Returns
    -------
    x       : improved assignment
    gc      : updated GainCache
    at_opt  : True if at local optimum
    n_flips : number of flips performed
    """
    x = dict(x)
    gc.update(G, x)

    nodes = list(G.nodes())
    n = len(nodes)

    # max_plateau=0 means plain descent — no plateau moves at all
    if max_plateau == 0:
        n_flips = 0
        while True:
            best_v, best_gain = gc.best_flip()
            if best_gain <= 0:
                return x, gc, True, n_flips
            x[best_v] = 1 - x[best_v]
            gc.incremental_update(G, x, best_v)
            n_flips += 1

    # default plateau budget
    if max_plateau is None:
        max_plateau = n * 5

    n_flips = 0
    plateau_moves = 0
    plateau_tabu = {}
    plateau_iter = 0

    while True:
        best_v, best_gain = gc.best_flip()

        if best_gain > 0:
            # improving move — always take it
            x[best_v] = 1 - x[best_v]
            gc.incremental_update(G, x, best_v)
            n_flips += 1
            plateau_moves = 0
            plateau_tabu = {}
            plateau_iter = 0

        elif best_gain == 0 and plateau_moves < max_plateau:
            # plateau move — take first zero-gain non-tabu vertex
            best_plateau_v = None
            for v in nodes:
                if gc.gain[v] == 0 and plateau_tabu.get(v, 0) <= plateau_iter:
                    best_plateau_v = v
                    break

            if best_plateau_v is None:
                return x, gc, True, n_flips

            plateau_iter += 1
            x[best_plateau_v] = 1 - x[best_plateau_v]
            gc.incremental_update(G, x, best_plateau_v)
            plateau_tabu[best_plateau_v] = plateau_iter + 5
            n_flips += 1
            plateau_moves += 1

        else:
            return x, gc, True, n_flips
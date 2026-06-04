"""
Neighbourhood selectors for Adaptive QLS.
All selectors share the same interface:
    selector(G, gc, k, pool=None) -> S (list of vertices)
"""

import numpy as np
import networkx as nx
from src.gain_cache import GainCache

def is_signed_graph_balanced(G, x=None, tol=1e-4):
    """
    Test signed-graph balance using the ACTUAL edge-weight signs.

    For ±1-weighted instances (G11/G12/G13), the sign pattern IS the
    frustration. Balance (frustration index zero) holds iff the signed
    Laplacian L = D - A_signed has smallest eigenvalue ~0 (Zaslavsky).

    Uses the real edge signs from the graph weights, NOT a uniform
    all-negative signature — that is what distinguishes balanced G11
    from frustrated G13.

    Returns (is_balanced, smallest_signed_laplacian_eigenvalue).
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        sigma = np.sign(w) if w != 0 else 1.0   # ACTUAL edge sign
        wabs = abs(w)
        i, j = idx[u], idx[v]
        # signed Laplacian L = D - A_signed: off-diagonal = -sigma*|w|
        rows += [i, j]; cols += [j, i]; vals += [-sigma * wabs, -sigma * wabs]
        deg[i] += wabs; deg[j] += wabs

    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    try:
        eigvals = spla.eigsh(L, k=2, which='SM', return_eigenvectors=False,
                             tol=1e-7, maxiter=8000)
        smallest = float(min(eigvals))
    except Exception:
        smallest = float('nan')

    return (abs(smallest) < tol, smallest)

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



def select_random(G, gc, k, pool=None, rng=None, x=None):
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


def select_frustrated(G, gc, k, pool=None, rng=None, x=None):    
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


def select_frustrated_connected(G, gc, k, pool=None, rng=None, x=None):
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

def select_fiedler(G, gc, k, pool=None, rng=None, x=None):
    """
    Fiedler-augmented selector — novel PhD contribution.
    Combines Fiedler vector with gain-based selection.

    Mathematical basis: Trevisan 2009 (Max Cut and Smallest Eigenvalue)
    proves vertices near the zero-crossing of the Fiedler vector are
    the most uncertain for Max-Cut — exactly the same intuition as the
    frustrated selector but grounded in spectral graph theory.

    Combined score: alpha * entropy(gain) + (1-alpha) * fiedler_proximity
    This selects vertices that are BOTH informationally uncertain (low
    |gain|) AND spectrally uncertain (near Fiedler zero-crossing).

    Uses frustration-weighted Laplacian: cut edges get weight +1,
    non-cut edges get weight -1. The Fiedler vector of this modified
    graph directly points at the most violated bipartite community.

    Parameters
    ----------
    G     : NetworkX graph
    gc    : GainCache
    k     : int — neighbourhood size
    pool  : ignored
    rng   : numpy.random.Generator or None
    x     : dict — current vertex assignments (used for weighted Laplacian)

    Returns
    -------
    S : list of k vertices — spectrally and informationally uncertain
    """
    gc.assert_valid()
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nodes = list(G.nodes())
    n = len(nodes)

    if n <= k:
        return nodes

    if rng is None:
        rng = np.random.default_rng()

    node_idx = {v: i for i, v in enumerate(nodes)}

    # ── build frustration-weighted Laplacian ──────────────────────
    # cut edges (x[u] != x[v]) get weight +1
    # non-cut edges (x[u] == x[v]) get weight -1
    # If x is not available, fall back to plain Laplacian
    rows, cols, vals = [], [], []
    deg = np.zeros(n)

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = node_idx[u], node_idx[v]

        if x is not None:
            # frustration-weighted: cut=+1, non-cut=-1
            edge_sign = 1.0 if x[u] != x[v] else -1.0
            ew = abs(w) * edge_sign
        else:
            ew = abs(w)

        # Build adjacency part: A[i,j] = -ew, A[j,i] = -ew
        rows += [i, j]
        cols += [j, i]
        vals += [-ew, -ew]
        deg[i] += abs(ew)
        deg[j] += abs(ew)

    # Laplacian = D - A where D is diagonal degree matrix
    L = sp.diags(deg) + sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # ── compute Fiedler vector ────────────────────────────────────
    try:
        # get second smallest eigenvalue and vector
        eigenvalues, eigenvectors = spla.eigsh(
            L, k=2, which='SM', tol=1e-3, maxiter=1000)
        # second eigenvector is Fiedler vector
        fiedler = eigenvectors[:, 1]
    
    except Exception as e:
        # F25: make the fallback VISIBLE, not silent. A silent fallback
        # would make a "Fiedler" run secretly part-FConn.
        select_fiedler._fallback_count = getattr(
            select_fiedler, '_fallback_count', 0) + 1
        import warnings
        warnings.warn(
            f"select_fiedler eigensolver failed ({e}); fell back to FConn "
            f"(fallback #{select_fiedler._fallback_count})")
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)
   
   
    # ── compute combined score ────────────────────────────────────
    # alpha=0.6: 60% gain entropy, 40% Fiedler proximity to zero
    alpha = 0.6
    eps = 0.005

    scores = {}
    for v in nodes:
        i = node_idx[v]
        # gain entropy: high score when |gain| is low (uncertain)
        gain_score = 1.0 / (1.0 + abs(gc.gain[v]))
        # fiedler score: high score when near zero-crossing (boundary)
        fiedler_score = 1.0 / (eps + abs(fiedler[i]))
        scores[v] = alpha * gain_score + (1 - alpha) * fiedler_score

    # ── grow connected subgraph from highest-score seed ───────────
    seed = max(nodes, key=lambda v: scores[v])

    S = [seed]
    frontier = set(G.neighbors(seed))

    while len(S) < k and frontier:
        best = max(frontier, key=lambda v: scores[v])
        S.append(best)
        for u in G.neighbors(best):
            if u not in S:
                frontier.add(u)
        frontier.discard(best)

    # fill if disconnected
    if len(S) < k:
        remaining = sorted(
            [v for v in nodes if v not in set(S)],
            key=lambda v: -scores[v]
        )
        S.extend(remaining[:k - len(S)])

    return S[:k]



def select_adaptive_spectral(G, gc, k, pool=None, rng=None, x=None):
    """
    Adaptive spectral selector — novel PhD contribution.
    Routes between Fiedler and connected frustrated based on
    graph's algebraic connectivity (lambda_2).

    Theory: Fiedler vector is most informative when lambda_2 is small
    (sparse graphs with clear community structure like G11).
    Connected frustrated is better on dense graphs (large lambda_2)
    where spectral signal is diffuse.

    Threshold: lambda_2 < 1.0 -> use Fiedler
               lambda_2 >= 1.0 -> use connected frustrated

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int
    pool : ignored
    rng  : numpy.random.Generator or None
    x    : dict — current vertex assignments

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import numpy as np

    # cache lambda_2 per graph to avoid recomputing every call
    graph_id = id(G)
    if not hasattr(select_adaptive_spectral, '_lambda2_cache'):
        select_adaptive_spectral._lambda2_cache = {}

    if graph_id not in select_adaptive_spectral._lambda2_cache:
        # compute algebraic connectivity
        nodes = list(G.nodes())
        n = len(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}

        rows, cols, vals = [], [], []
        for u, v, data in G.edges(data=True):
            w = abs(data.get('weight', 1.0))
            i, j = node_idx[u], node_idx[v]
            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            vals.extend([w, w, -w, -w])

        L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        try:
            eigenvalues, _ = spla.eigsh(L, k=2, which='SM',
                                        tol=1e-3, maxiter=1000)
            lambda2 = sorted(eigenvalues)[1]
        except Exception:
            lambda2 = 1.0  # fallback

        select_adaptive_spectral._lambda2_cache[graph_id] = lambda2

    lambda2 = select_adaptive_spectral._lambda2_cache[graph_id]

    # route based on algebraic connectivity
    if lambda2 < 1.0:
        # sparse graph — Fiedler is informative
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=x)
    else:
        # dense graph — connected frustrated is better
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)

def select_smart_adaptive(G, gc, k, pool=None, rng=None, x=None):
    """
    Smart adaptive selector — novel PhD contribution.
    Three-tier routing combining static graph structure analysis
    with dynamic runtime escape rate monitoring.

    Tier 1 (static): bipartiteness check
        bipartite → Fiedler (provably exact, Desai-Rao theorem)
    Tier 2 (static): algebraic connectivity
        non-bipartite, lambda2 < 2.0 → Fiedler with plain Laplacian
    Tier 3 (static): dense graphs
        lambda2 >= 2.0 → connected frustrated

    Dynamic override: monitors escape rate EMA over last 10 calls.
        If escape rate drops below 0.1 → switch to other selector.
        If escape rate recovers → switch back.

    This makes the selector genuinely adaptive to the search state
    not just the graph structure.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int
    pool : ignored
    rng  : numpy.random.Generator or None
    x    : dict — current vertex assignments

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    import networkx as nx
    import numpy as np

    # ── initialise state caches ───────────────────────────────────
    graph_id = id(G)

    if not hasattr(select_smart_adaptive, '_state'):
        select_smart_adaptive._state = {}

    if graph_id not in select_smart_adaptive._state:
        # compute static graph features once
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        nodes = list(G.nodes())
        n = len(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}

      
        # Bipartiteness of the underlying graph predicts Fiedler exact
        # recovery. Verified empirically (check_balance.py, 2026-06-01):
        # nx.is_bipartite gives the correct split G11/G12=True, G13=False,
        # G14/G22/G1=False — whereas is_signed_graph_balanced was
        # anti-correlated (mislabelled +1 graphs as balanced). See audit F9.
        is_bipartite = nx.is_bipartite(G)

        # algebraic connectivity
        rows, cols, vals = [], [], []
        for u, v, data in G.edges(data=True):
            w = abs(data.get('weight', 1.0))
            i, j = node_idx[u], node_idx[v]
            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            vals.extend([w, w, -w, -w])
        L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        try:
            eigenvalues, _ = spla.eigsh(L, k=2, which='SM',
                                        tol=1e-3, maxiter=1000)
            lambda2 = sorted(eigenvalues)[1]
        except Exception:
            lambda2 = 1.0


        # determine static selector
        # bipartite ±1 graphs (G11/G12) → Fiedler (exact recovery)
        # everything else → FConn (the safe generalist; Fiedler is
        # harmful on non-bipartite graphs, confirmed on G13/G14/G22/G1)
        if is_bipartite:
            static_selector = 'fiedler'
        else:
            static_selector = 'fconn'

        
        select_smart_adaptive._state[graph_id] = {
            'is_bipartite': is_bipartite,
            'lambda2': lambda2,
            'static_selector': static_selector,
            'current_selector': static_selector,
            'escape_ema': 0.5,      # EMA of escape success
            'call_count': 0,
            'last_cut': None,
        }

    state = select_smart_adaptive._state[graph_id]
    state['call_count'] += 1

    # ── dynamic escape rate monitoring ────────────────────────────
    # update escape EMA based on whether cut improved since last call
    if x is not None and state['last_cut'] is not None:
        from src.local_search import compute_cut_value
        current_cut = compute_cut_value(G, x)
        improved = 1.0 if current_cut > state['last_cut'] else 0.0
        # EMA with alpha=0.3
        state['escape_ema'] = 0.3 * improved + 0.7 * state['escape_ema']

        # dynamic switching: if escape rate low, try other selector
        if state['call_count'] > 5:  # warmup period
            if state['escape_ema'] < 0.1:
                # current selector not working — switch
                if state['current_selector'] in ('fiedler', 'fiedler_plain'):
                    state['current_selector'] = 'fconn'
                else:
                    # switch back to static default
                    state['current_selector'] = state['static_selector']
                state['escape_ema'] = 0.5  # reset after switch
            elif state['escape_ema'] > 0.4:
                # current selector working well — allow return to static
                state['current_selector'] = state['static_selector']

    # update last cut
    if x is not None:
        from src.local_search import compute_cut_value
        state['last_cut'] = compute_cut_value(G, x)

    # ── call appropriate selector ─────────────────────────────────
    sel = state['current_selector']

    if sel == 'fiedler':
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=x)
    elif sel == 'fiedler_plain':
        # Fiedler with plain unsigned Laplacian for near-bipartite graphs
        # temporarily set x=None to force plain Laplacian in select_fiedler
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=None)
    else:
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)

def select_cut_polytope(G, gc, k, pool=None, rng=None, x=None):
    """
    Cut polytope selector — novel PhD contribution.
    Finds vertices involved in the most violated odd-cycle
    inequalities of the cut polytope at the current solution x.

    For each triangle (u,v,w) in G, checks whether the number
    of cut edges is even — which violates the odd-cycle inequality.
    Vertices in the most violated triangles are selected as S.

    Mathematical basis: Barahona & Mahjoub 1986 — odd-cycle
    inequalities define facets of CUT(G). A solution violating
    these inequalities is not a vertex of the cut polytope and
    must change to reach a better cut.

    No published Max-Cut heuristic uses facet violations
    as a destroy rule — this is a novel PhD contribution.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : list of solution dicts (fallback if x is None)
    rng  : numpy.random.Generator or None
    x    : dict — current vertex assignments {v: 0 or 1}

    Returns
    -------
    S : list of k vertices from most violated odd cycles
    """
    gc.assert_valid()
    nodes = list(G.nodes())

    if len(nodes) <= k:
        return nodes

    if rng is None:
        rng = np.random.default_rng()

    # ── get current assignment ────────────────────────────────────
    # prefer x passed directly, fall back to best pool solution
    if x is None:
        if pool and len(pool) > 0:
            from src.local_search import compute_cut_value
            x = max(pool, key=lambda p: compute_cut_value(G, p))
        else:
            # no assignment available — fall back to frustrated_connected
            return select_frustrated_connected(
                G, gc, k, pool=pool, rng=rng, x=None)

    # ── compute violation score per vertex ────────────────────────
    violation_score = {v: 0.0 for v in nodes}

    # check every triangle for odd-cycle violation
    # triangle (u,v,w) is violated if number of cut edges is even
    # cut edge = edge where x[u] != x[v]
    triangles_checked = 0
    max_triangles = 1000

    for u in nodes:
        if triangles_checked >= max_triangles:
            break
        for v in G.neighbors(u):
            if v <= u:
                continue
            for w in G.neighbors(v):
                if w <= v:
                    continue
                if not G.has_edge(u, w):
                    continue
                if triangles_checked >= max_triangles:
                    break

                # count cut edges in triangle
                cut_uv = int(x[u] != x[v])
                cut_vw = int(x[v] != x[w])
                cut_uw = int(x[u] != x[w])
                n_cut = cut_uv + cut_vw + cut_uw

                # odd-cycle violated if n_cut is even (0 or 2)
                if n_cut % 2 == 0:
                    # get edge weights
                    w_uv = G[u][v].get('weight', 1.0)
                    w_vw = G[v][w].get('weight', 1.0)
                    w_uw = G[u][w].get('weight', 1.0)
                    violation = (w_uv + w_vw + w_uw) / 3.0

                    violation_score[u] += violation
                    violation_score[v] += violation
                    violation_score[w] += violation

                triangles_checked += 1

    # ── if no triangles found, add edge-based violation score ─────
    # for sparse graphs with few triangles (like G11)
    if triangles_checked == 0 or max(violation_score.values()) == 0:
        # fallback — score edges where both endpoints same side
        # and edge weight is high (non-cut edge with high weight)
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1.0)
            if x[u] == x[v]:
                # same side — this edge not cut — potential violation
                edge_score = abs(w)
                violation_score[u] += edge_score
                violation_score[v] += edge_score

    # ── grow connected subgraph from highest violation vertex ─────
    seed = max(nodes, key=lambda v: violation_score[v])

    S = [seed]
    frontier = set(G.neighbors(seed))

    while len(S) < k and frontier:
        best = max(frontier, key=lambda v: violation_score[v])
        S.append(best)
        for u in G.neighbors(best):
            if u not in S:
                frontier.add(u)
        frontier.discard(best)

    # fill if disconnected graph exhausted frontier
    if len(S) < k:
        remaining = sorted(
            [v for v in nodes if v not in set(S)],
            key=lambda v: -violation_score[v]
        )
        S.extend(remaining[:k - len(S)])

    return S[:k]

def select_topological(G, gc, k, pool=None, rng=None, x=None):
    """
    Topological selector — novel PhD contribution for toroidal graphs.
    Every log(n) iterations seeds the subproblem from a non-contractible
    loop (meridian or longitude cycle) rather than from frustrated faces.

    Motivated by Galluccio-Loebl-Vondrák theorem: G11 is a toroidal
    graph where non-contractible cycles generate facets of the cut
    polytope that face-cycle selection cannot reach. When AQLS gets
    stuck it may be due to a topological obstruction — a global loop
    frustration that no local face-cycle move can resolve.

    For non-toroidal graphs falls back to select_frustrated_connected.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int — neighbourhood size
    pool : list of solution dicts
    rng  : numpy.random.Generator or None
    x    : dict — current vertex assignments

    Returns
    -------
    S : list of k vertices seeded from non-contractible loop
    """
    gc.assert_valid()
    nodes = list(G.nodes())
    n = len(nodes)

    if rng is None:
        rng = np.random.default_rng()

    # ── detect toroidal grid structure ────────────────────────────
    # G11: 8 cols x 100 rows, nodes 1-800
    # detect by checking if n is divisible by 8 and step=8 exists
    cols, rows = _detect_toroidal_grid(G, nodes)

    if cols is None:
        # not a toroidal grid — fall back to connected frustrated
        return select_frustrated_connected(
            G, gc, k, pool=pool, rng=rng, x=x)

    # ── build non-contractible cycles ─────────────────────────────
    def node_id(r, c):
        return (r % rows) * cols + (c % cols) + min(nodes)

    # meridian cycles — length = rows, one per column
    meridians = [
        [node_id(r, c) for r in range(rows)]
        for c in range(cols)
    ]

    # longitude cycles — length = cols, one per row
    longitudes = [
        [node_id(r, c) for c in range(cols)]
        for r in range(rows)
    ]

    all_loops = meridians + longitudes

    # ── score each loop by total frustration ─────────────────────
    # most frustrated loop = most likely to contain topological obstruction
    def loop_frustration(loop):
        return sum(1.0 / (1.0 + abs(gc.gain[v]))
                   for v in loop if v in gc.gain)

    best_loop = max(all_loops, key=loop_frustration)

    # ── seed from most frustrated vertex in the best loop ─────────
    seed = min(best_loop, key=lambda v: abs(gc.gain.get(v, 1.0)))

    # ── grow connected subgraph from seed ─────────────────────────
    # same strategy as frustrated_connected but seeded topologically
    S = [seed]
    frontier = set(G.neighbors(seed))

    while len(S) < k and frontier:
        best = min(frontier, key=lambda v: abs(gc.gain.get(v, 1.0)))
        S.append(best)
        for u in G.neighbors(best):
            if u not in S:
                frontier.add(u)
        frontier.discard(best)

    # fill if needed
    if len(S) < k:
        remaining = sorted(
            [v for v in nodes if v not in set(S)],
            key=lambda v: abs(gc.gain.get(v, 1.0))
        )
        S.extend(remaining[:k - len(S)])

    return S[:k]


def _detect_toroidal_grid(G, nodes):
    """
    Detect if G is a toroidal grid and return (cols, rows).
    Returns (None, None) if not detected.
    """
    n = len(nodes)
    min_node = min(nodes)

    # check if all nodes have degree 4
    if any(G.degree(v) != 4 for v in nodes):
        return None, None

    # find the column step by looking at neighbour differences
    # on a cols x rows toroidal grid, each node connects to
    # +1, -1 (horizontal) and +cols, -cols (vertical)
    sample_node = min_node
    neighbour_diffs = sorted([
        abs(v - sample_node) for v in G.neighbors(sample_node)
    ])

    # the two unique positive diffs are 1 (horizontal) and cols (vertical)
    # wraparound gives n-1 and n-cols
    small_diffs = [d for d in neighbour_diffs if d <= n // 2]

    if len(small_diffs) < 2:
        return None, None

    small_diffs = sorted(set(small_diffs))

    if len(small_diffs) >= 2:
        col_step = small_diffs[1]  # larger of the two small diffs
        if n % col_step == 0:
            cols = col_step
            rows = n // cols
            return cols, rows

    return None, None

def select_hybrid_topo(G, gc, k, pool=None, rng=None, x=None):
    """
    Hybrid topological selector — combines connected frustrated
    with periodic topological seeding.

    Every log(n) iterations seeds from a non-contractible loop
    instead of from frustrated faces. This gives access to both
    face-cycle facets (via frustrated selection) and topological
    facets (via non-contractible loops) of the cut polytope.

    Parameters
    ----------
    G    : NetworkX graph
    gc   : GainCache
    k    : int
    pool : list of solution dicts
    rng  : numpy.random.Generator or None
    x    : dict — current vertex assignments

    Returns
    -------
    S : list of k vertices
    """
    gc.assert_valid()
    import math

    nodes = list(G.nodes())
    n = len(nodes)

    if rng is None:
        rng = np.random.default_rng()

    # every log(n) calls use topological seeding
    log_n = max(2, int(math.log(n)))

    # use call counter stored as function attribute
    if not hasattr(select_hybrid_topo, '_call_count'):
        select_hybrid_topo._call_count = 0
    select_hybrid_topo._call_count += 1

    if select_hybrid_topo._call_count % log_n == 0:
        # topological iteration
        return select_topological(G, gc, k, pool=pool, rng=rng, x=x)
    else:
        # standard connected frustrated iteration
        return select_frustrated_connected(
            G, gc, k, pool=pool, rng=rng, x=x)

def select_impact(G, gc, k, pool=None, rng=None, x=None):
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


def select_clustering(G, gc, k, pool=None, rng=None, x=None):
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



def select_meta_rule(G, gc, k, pool=None, rng=None, x=None):
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


def select_lambda2_routed(G, gc, k, pool=None, rng=None, x=None,
                          lambda2_threshold=0.2):
    """Lambda2-routed meta-selector (confound-controlled finding).

    Measures algebraic connectivity (lambda2) ONCE and routes:
      lambda2 < threshold -> Fiedler   (low-connectivity / structured)
      lambda2 >= threshold -> FConn    (high-connectivity / expander-like)

    Threshold default 0.2 sits in the empirical gap between Fiedler-wins
    (G11/G13: lambda2 0.004-0.038) and FConn-wins (reg4: ~0.53; dense: 6-25).
    lambda2 cached on G to avoid recomputation per call.
    """
    import networkx as nx
    lam2 = G.graph.get("_lambda2_cache")
    if lam2 is None:
        try:
            lam2 = nx.algebraic_connectivity(G, method="lanczos")
        except Exception:
            lam2 = float("inf")   # disconnected/failed -> treat as high
        G.graph["_lambda2_cache"] = lam2
    if lam2 < lambda2_threshold:
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=x)
    else:
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)


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
        'cut_polytope':         select_cut_polytope,
        'impact':               select_impact,
        'clustering':           select_clustering,
        'meta_rule':            select_meta_rule,
        'topological':          select_topological,
        'hybrid_topo':          select_hybrid_topo,
        'fiedler':              select_fiedler,
        'adaptive_spectral':    select_adaptive_spectral,
        'smart_adaptive':       select_smart_adaptive,
    }
  
    
    if name not in selectors:
        raise ValueError(
            f"Unknown selector '{name}'. "
            f"Choose from: {list(selectors.keys())}"
        )
    return selectors[name]


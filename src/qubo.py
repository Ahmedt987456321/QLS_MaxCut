"""
QUBO subproblem construction for Max-Cut local search.
BuildLocalQUBO extracts a local QUBO model over a selected
vertex subset S, with boundary conditions fixed by the incumbent.
"""


def build_local_qubo(G, x, S):
    """
    Build a QUBO dict for the subgraph induced by vertex subset S.
    Vertices outside S are fixed to their current assignment in x.

    Minimising the returned Q is equivalent to maximising the
    cut contribution of S given the fixed boundary.

    Parameters
    ----------
    G : NetworkX graph (weighted or unweighted)
    x : dict {vertex: 0 or 1} — current full assignment
    S : list or set of vertices — the selected neighbourhood

    Returns
    -------
    Q : dict {(i, j): value} — upper-triangular QUBO matrix
        diagonal entries use (i, i) keys
    """
    S = set(S)
    Q = {}

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)

        u_in = u in S
        v_in = v in S

        if u_in and v_in:
            # internal edge — both endpoints free to optimise
            # maximising cut: penalise same-side assignment
            Q[(u, u)] = Q.get((u, u), 0.0) - w
            Q[(v, v)] = Q.get((v, v), 0.0) - w
            Q[(u, v)] = Q.get((u, v), 0.0) + 2.0 * w

        elif u_in and not v_in:
            # boundary edge — v is fixed at x[v]
            # reward u being on the opposite side from v
            if x[v] == 0:
                Q[(u, u)] = Q.get((u, u), 0.0) - w
            else:
                Q[(u, u)] = Q.get((u, u), 0.0) + w

        elif v_in and not u_in:
            # boundary edge — u is fixed at x[u]
            if x[u] == 0:
                Q[(v, v)] = Q.get((v, v), 0.0) - w
            else:
                Q[(v, v)] = Q.get((v, v), 0.0) + w

    return Q


def qubo_energy(Q, x_local, S):
    """
    Evaluate QUBO energy for a given local assignment.
    Lower energy = better cut within S.

    Parameters
    ----------
    Q       : dict — QUBO matrix from build_local_qubo
    x_local : dict {vertex: 0 or 1} — assignment for vertices in S
    S       : list or set of vertices in the subproblem

    Returns
    -------
    energy : float
    """
    S = list(S)
    energy = 0.0
    for i in range(len(S)):
        vi = S[i]
        xi = x_local.get(vi, 0)
        energy += Q.get((vi, vi), 0.0) * xi
        for j in range(i + 1, len(S)):
            vj = S[j]
            xj = x_local.get(vj, 0)
            energy += (Q.get((vi, vj), 0.0) + Q.get((vj, vi), 0.0)) * xi * xj
    return energy


def merge_proposal(x, x_local, S):
    """
    Merge a local assignment back into the full solution.

    Parameters
    ----------
    x       : dict — current full assignment
    x_local : dict — local assignment for vertices in S
    S       : list or set of vertices

    Returns
    -------
    x_prop : dict — new full assignment with S updated
    """
    x_prop = dict(x)
    for v in S:
        if v in x_local:
            x_prop[v] = x_local[v]
    return x_prop
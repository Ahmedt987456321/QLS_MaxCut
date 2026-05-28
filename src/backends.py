"""
Backend solvers for local QUBO subproblems.
All backends share the same interface:
    backend(Q, S, n_reads, init=None) -> x_local (dict {vertex: 0 or 1})
"""

import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler as _NealSampler
from simulated_bifurcation import maximize
from simulated_bifurcation.optimizer.simulated_bifurcation_engine import SimulatedBifurcationEngine


def backend_exact(Q, S, n_reads=None, init=None):
    """
    Exact brute-force solver. Only use for |S| <= 20.

    Parameters
    ----------
    Q      : dict — QUBO from build_local_qubo
    S      : list of vertices in subproblem
    n_reads: ignored for exact solver
    init   : ignored for exact solver

    Returns
    -------
    x_local : dict {vertex: 0 or 1} — optimal assignment
    """
    S = list(S)
    n = len(S)
    assert n <= 20, f"Exact solver only for |S| <= 20, got |S|={n}"

    best_energy = float('inf')
    best_x = None

    for mask in range(2 ** n):
        x_local = {}
        for i, v in enumerate(S):
            x_local[v] = (mask >> i) & 1

        energy = 0.0
        for i in range(n):
            vi = S[i]
            xi = x_local[vi]
            energy += Q.get((vi, vi), 0.0) * xi
            for j in range(i + 1, n):
                vj = S[j]
                xj = x_local[vj]
                energy += Q.get((vi, vj), 0.0) * xi * xj

        if energy < best_energy:
            best_energy = energy
            best_x = dict(x_local)

    return best_x


def backend_neal(Q, S, n_reads=100, init=None):
    """
    Simulated annealing via D-Wave neal.
    Quantum-inspired classical baseline.

    Parameters
    ----------
    Q      : dict — QUBO from build_local_qubo
    S      : list of vertices in subproblem
    n_reads: int — number of SA runs
    init   : dict {vertex: 0 or 1} or None — warm start

    Returns
    -------
    x_local : dict {vertex: 0 or 1} — best assignment found
    """
    S = list(S)

    # build dimod BQM from our QUBO dict
    bqm = dimod.BinaryQuadraticModel('BINARY')
    for (i, j), val in Q.items():
        if i == j:
            bqm.add_variable(i, val)
        else:
            bqm.add_interaction(i, j, val)

    sampler = _NealSampler()

    # prepare warm start if provided
    initial_states = None
    if init is not None:
        initial_states = [{v: init[v] for v in S if v in init}]

    response = sampler.sample(
        bqm,
        num_reads=n_reads,
        initial_states=initial_states
    )

    best_sample = response.first.sample
    return {v: best_sample[v] for v in S if v in best_sample}


def backend_sbm(Q, S, n_reads=200, init=None):
    """
    Simulated Bifurcation Machine backend.
    Uses PyTorch-based SBM as the local QUBO solver.

    Automatically routes to neal for sparse subproblems
    (avg degree < 5) where SBM has convergence issues.
    Uses SBM only for dense subproblems where it excels.

    Uses discrete mode for k>=300 (better quality).
    Uses ballistic mode for k<300 (faster).

    Parameters
    ----------
    Q       : dict — QUBO coefficients {(i,j): value}
    S       : list — vertices in subproblem
    n_reads : int — number of parallel agents
    init    : ignored (SBM initialises internally)

    Returns
    -------
    x_local : dict — {vertex: 0 or 1} best solution found
    """
    try:
        import torch
        from simulated_bifurcation import maximize

        n = len(S)
        if n == 0:
            return {}

        # map vertices to indices 0..n-1
        idx = {v: i for i, v in enumerate(S)}

        # build QUBO matrix as torch tensor
        J = torch.zeros(n, n, dtype=torch.float32)
        h = torch.zeros(n, dtype=torch.float32)

        for (i, j), val in Q.items():
            if i not in idx or j not in idx:
                continue
            ii, jj = idx[i], idx[j]
            if ii == jj:
                h[ii] -= float(val)
            else:
                J[ii, jj] -= float(val) / 2.0
                J[jj, ii] -= float(val) / 2.0

        # ── density check — SBM needs dense subproblems ──────────
        edge_count = (J != 0).sum().item() // 2
        avg_degree = (2 * edge_count) / max(n, 1)
        if avg_degree < 5:
            # sparse subproblem — fall back to neal
            return backend_neal(Q, S, n_reads=n_reads, init=init)

        # choose mode based on problem size
        mode = "discrete" if n >= 300 else "ballistic"

        # run SBM
        spins, energy = maximize(
            J, h,
            agents=n_reads,
            mode=mode,
            max_steps=500,
            verbose=False,
            domain='spin'
        )

        # convert spins {-1, +1} to binary {0, 1}
        best_spins = spins.numpy() if hasattr(spins, 'numpy') else spins

        x_local = {}
        for v in S:
            i = idx[v]
            spin = int(best_spins[i]) if i < len(best_spins) else 0
            x_local[v] = 1 if spin == 1 else 0

        return x_local

    except Exception as e:
        # fallback to neal if SBM fails for any reason
        return backend_neal(Q, S, n_reads=n_reads, init=init)

def backend_sqa(Q, S, n_reads=100, init=None):
    """
    Simulated Quantum Annealing via OpenJij.
    Quantum-inspired with transverse field simulation.

    Parameters
    ----------
    Q      : dict — QUBO from build_local_qubo
    S      : list of vertices in subproblem
    n_reads: int — number of SQA runs
    init   : dict {vertex: 0 or 1} or None — warm start

    Returns
    -------
    x_local : dict {vertex: 0 or 1} — best assignment found
    """
    try:
        import openjij as oj
    except ImportError:
        raise ImportError("OpenJij not installed. Run: pip install openjij")

    S = list(S)

    # convert to openjij format
    Q_oj = {}
    for (i, j), val in Q.items():
        Q_oj[(i, j)] = val

    sampler = oj.SQASampler()

    response = sampler.sample_qubo(
        Q_oj,
        num_reads=n_reads
    )

    best_sample = response.first.sample
    return {v: best_sample.get(v, 0) for v in S}


def get_backend(name):
    """
    Factory function — returns backend by name.

    Parameters
    ----------
    name : str — 'exact', 'neal', 'sqa'

    Returns
    -------
    backend function
    """
    backends = {
        'exact': backend_exact,
        'neal':  backend_neal,
        'sqa':   backend_sqa,
    }
    if name not in backends:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {list(backends.keys())}"
        )
    return backends[name]
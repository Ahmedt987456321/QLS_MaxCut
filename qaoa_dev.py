"""QAOA Phase-5 backend (qiskit 2.x + Aer) - dev/validation, small k only."""
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2
from scipy.optimize import minimize

def _qubo_to_ising(Q, S):
    S = list(S); idx = {v: i for i, v in enumerate(S)}; n = len(S)
    h = np.zeros(n); Jd = {}
    for (a, b), w in Q.items():
        if a == b:
            h[idx[a]] += w / 2.0
        else:
            i, j = idx[a], idx[b]; key = (min(i,j), max(i,j))
            Jd[key] = Jd.get(key, 0.0) + w / 4.0
            h[idx[a]] += w/4.0; h[idx[b]] += w/4.0
    return Jd, h, S, n

def _ising_hamiltonian(Jd, h, n):
    terms = []
    for (i, j), J in Jd.items():
        z = ["I"]*n; z[i]="Z"; z[j]="Z"
        terms.append(("".join(reversed(z)), J))
    for i in range(n):
        if abs(h[i]) > 1e-12:
            z = ["I"]*n; z[i]="Z"
            terms.append(("".join(reversed(z)), h[i]))
    return SparsePauliOp.from_list(terms)

def backend_qaoa(Q, S, n_reads=None, init=None, reps=4, maxiter=300):
    """QAOA via qiskit Aer. Small k only (<=16). Minimizes QUBO energy."""
    Jd, h, S, n = _qubo_to_ising(Q, S)
    if n == 0 or not Jd:
        return {v: 0 for v in S}
    H = _ising_hamiltonian(Jd, h, n)
    ansatz = qaoa_ansatz(H, reps=reps)
    sim = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
    ansatz_t = pm.run(ansatz)
    sampler = SamplerV2()

    def expval(params):
        qc = ansatz.assign_parameters(params)
        from qiskit.primitives import StatevectorEstimator
        est = StatevectorEstimator()
        return float(est.run([(qc, H)]).result()[0].data.evs)

    x0 = np.random.default_rng(0).uniform(0, np.pi, ansatz.num_parameters)
    res = minimize(expval, x0, method="COBYLA", options={"maxiter": maxiter})

    # sample best bitstring from optimized circuit
    qc = ansatz.assign_parameters(res.x)
    qc.measure_all()
    qc_t = pm.run(qc)
    job = sampler.run([qc_t], shots=512)
    counts = job.result()[0].data.meas.get_counts()
    from src.qubo import qubo_energy as _qe
    best_sol, best_e = None, float("inf")
    for bitstr in counts:
        bits = [int(b) for b in reversed(bitstr)]
        sol = {S[i]: bits[i] for i in range(n)}
        e = _qe(Q, sol, S)
        if e < best_e:
            best_e, best_sol = e, sol
    return best_sol


if __name__ == "__main__":
    from src.graph import load_gset
    from src.local_search import random_cut, one_flip_ls
    from src.gain_cache import GainCache
    from src.selectors import select_frustrated_connected
    from src.qubo import build_local_qubo, qubo_energy
    from src.backends import backend_exact

    G = load_gset("data/gset/G11.txt")
    print("QAOA vs exact on G11 (k=8):")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = random_cut(G, rng); gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc); gc.update(G, x)
        S = select_frustrated_connected(G, gc, 8, rng=rng, x=x)
        Q = build_local_qubo(G, x, S)
        e_q = qubo_energy(Q, backend_qaoa(Q, S), S)
        e_e = qubo_energy(Q, backend_exact(Q, S), S)
        gap = e_q - e_e
        print(f"  seed {seed}: QAOA={e_q:.1f} exact={e_e:.1f} gap={gap:.1f}")

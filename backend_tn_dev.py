"""backend_tn: treewidth-gated TN backend via persistent Julia GTN worker."""
import json, subprocess, atexit
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in
from src.backends import backend_neal

_PROC = None
TW_GATE = 28

def _worker():
    global _PROC
    if _PROC is None:
        _PROC = subprocess.Popen(
            ["julia", "tn_server.jl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        atexit.register(lambda: _PROC.terminate())
    return _PROC

def _qubo_to_ising(Q, S):
    S = list(S); idx = {v: i for i, v in enumerate(S)}; n = len(S)
    h = np.zeros(n); Jd = {}
    for (a, b), w in Q.items():
        if a == b:
            h[idx[a]] += w / 2.0
        else:
            i, j = idx[a], idx[b]; key = (min(i, j), max(i, j))
            Jd[key] = Jd.get(key, 0.0) + w / 4.0
            h[idx[a]] += w / 4.0; h[idx[b]] += w / 4.0
    edges = [[i + 1, j + 1] for (i, j) in Jd]
    J = [Jd[(i, j)] for (i, j) in Jd]
    return edges, J, h.tolist(), idx, S, n

def _treewidth(Q, S):
    S = set(S)
    H = nx.Graph()
    H.add_nodes_from(S)
    for (a, b) in Q:
        if a != b and a in S and b in S:
            H.add_edge(a, b)
    if H.number_of_nodes() == 0:
        return 0
    tw, _ = treewidth_min_fill_in(H)
    return tw

def backend_tn(Q, S, n_reads=None, init=None):
    """Exact TN via GTN if treewidth <= gate, else neal fallback."""
    S = list(S)
    tw = _treewidth(Q, S)
    if tw > TW_GATE:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)

    edges, J, h, idx, Slist, n = _qubo_to_ising(Q, S)
    if not edges:  # no internal couplings -> trivial
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)

    p = _worker()
    p.stdin.write(json.dumps({"n": n, "edges": edges, "J": J, "h": h}) + "\n")
    p.stdin.flush()
    out = json.loads(p.stdout.readline())
    spins = out["config"]
    return {Slist[i]: int(spins[i]) for i in range(n)}  # verified: bits = x


if __name__ == "__main__":
    # validate against backend_exact on several G11 patches
    from src.graph import load_gset
    from src.local_search import random_cut, one_flip_ls
    from src.gain_cache import GainCache
    from src.selectors import select_frustrated_connected
    from src.qubo import build_local_qubo, qubo_energy
    from src.backends import backend_exact

    G = load_gset("data/gset/G11.txt")
    print("Validating backend_tn vs backend_exact on G11 (k=14):")
    ok = True
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = random_cut(G, rng)
        gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc); gc.update(G, x)
        S = select_frustrated_connected(G, gc, 14, rng=rng, x=x)
        Q = build_local_qubo(G, x, S)
        e_tn = qubo_energy(Q, backend_tn(Q, S), S)
        e_ex = qubo_energy(Q, backend_exact(Q, S), S)
        match = abs(e_tn - e_ex) < 1e-6
        ok = ok and match
        print(f"  seed {seed}: TN={e_tn:.1f} exact={e_ex:.1f} "
              f"tw={_treewidth(Q,S)} {'OK' if match else 'MISMATCH'}")
    print("ALL MATCH" if ok else "FAILURES - do not use")

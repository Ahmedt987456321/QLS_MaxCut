with open("src/backends.py", encoding="utf-8") as f:
    src = f.read()
idx = src.index("def get_backend(name):")
head = src[:idx]
new_block = r'''import os as _os
import json as _json
import subprocess as _subprocess
import atexit as _atexit

_TN_PROC = None
_TN_GATE = 28
_TN_SERVER = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "tn_server.jl")


def _tn_worker():
    global _TN_PROC
    if _TN_PROC is not None:
        return _TN_PROC
    if not _os.path.exists(_TN_SERVER):
        return None
    try:
        _TN_PROC = _subprocess.Popen(["julia", _TN_SERVER],
            stdin=_subprocess.PIPE, stdout=_subprocess.PIPE,
            text=True, bufsize=1)
        _atexit.register(lambda: _TN_PROC and _TN_PROC.terminate())
        return _TN_PROC
    except Exception:
        return None


def _tn_treewidth(Q, S):
    import networkx as nx
    from networkx.algorithms.approximation import treewidth_min_fill_in
    S = set(S)
    H = nx.Graph()
    H.add_nodes_from(S)
    for (a, b) in Q:
        if a != b and a in S and b in S:
            H.add_edge(a, b)
    if H.number_of_nodes() == 0:
        return 0
    return treewidth_min_fill_in(H)[0]


def backend_tn(Q, S, n_reads=None, init=None):
    """Exact TN backend via GenericTensorNetworks.jl, gated on treewidth.
    Falls back to neal when treewidth>gate, no internal couplings, or the
    Julia/GTN worker is unavailable. Verified: Q_ii->h+=Q_ii/2;
    Q_ij->J+=Q_ij/4,h+=Q_ij/4; config bits read directly as x."""
    import numpy as np
    S = list(S)
    try:
        tw = _tn_treewidth(Q, S)
    except Exception:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)
    if tw > _TN_GATE:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)
    idx = {v: i for i, v in enumerate(S)}
    n = len(S)
    h = np.zeros(n)
    Jd = {}
    for (a, b), w in Q.items():
        if a == b:
            h[idx[a]] += w / 2.0
        else:
            i, j = idx[a], idx[b]
            key = (min(i, j), max(i, j))
            Jd[key] = Jd.get(key, 0.0) + w / 4.0
            h[idx[a]] += w / 4.0
            h[idx[b]] += w / 4.0
    edges = [[i + 1, j + 1] for (i, j) in Jd]
    if not edges:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)
    J = [Jd[(i, j)] for (i, j) in Jd]
    proc = _tn_worker()
    if proc is None:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)
    try:
        proc.stdin.write(_json.dumps(
            {"n": n, "edges": edges, "J": J, "h": h.tolist()}) + "\n")
        proc.stdin.flush()
        out = _json.loads(proc.stdout.readline())
        spins = out["config"]
        return {S[i]: int(spins[i]) for i in range(n)}
    except Exception:
        return backend_neal(Q, S, n_reads=n_reads or 100, init=init)


def get_backend(name):
    """Factory -- returns backend by name ('exact','neal','sqa','tn')."""
    backends = {"exact": backend_exact, "neal": backend_neal,
                "sqa": backend_sqa, "tn": backend_tn}
    if name not in backends:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {list(backends.keys())}")
    return backends[name]
'''
with open("src/backends.py", "w", encoding="utf-8") as f:
    f.write(head + new_block)
print("backend_tn added")

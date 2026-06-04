# PhD probe v2: USE_NEAL is now argv[3] (0=steepest descent, 1=neal). Default 0.
import sys, csv, os
import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else r"data\gset\G11.txt"
K_SUPPORT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
USE_NEAL = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
N_SUPPORTS = 40
N_RANDOM   = 12
NEAL_SWEEPS = 80
RNG = np.random.default_rng(20260604)

def load_gset(path):
    with open(path) as f:
        first = f.readline().split()
        n, m = int(first[0]), int(first[1])
        A = np.zeros((n, n))
        for line in f:
            p = line.split()
            if len(p) < 2: continue
            u, v = int(p[0]) - 1, int(p[1]) - 1
            w = float(p[2]) if len(p) > 2 else 1.0
            A[u, v] = w; A[v, u] = w
    return A, n, m

def cut_value(A, s):
    return 0.25 * (A.sum() - s @ (A @ s))

def steepest_descent(A, s0, max_iter=20000):
    s = s0.copy().astype(float)
    for _ in range(max_iter):
        g = (A @ s) * s
        i = int(np.argmax(g))
        if g[i] <= 1e-12: break
        s[i] = -s[i]
    return s

def solve(A, seed_spin):
    n = len(seed_spin)
    if USE_NEAL:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
            J = {}
            iu = np.triu_indices(n, 1)
            for u, v in zip(*iu):
                if A[u, v] != 0.0:
                    J[(int(u), int(v))] = float(A[u, v])
            init = {i: int(seed_spin[i]) for i in range(n)}
            res = SimulatedAnnealingSampler().sample_ising(
                {}, J, num_reads=1, num_sweeps=NEAL_SWEEPS,
                initial_states=init, initial_states_generator="none", seed=0)
            smp = res.first.sample
            x = np.array([smp[i] for i in range(n)], dtype=float)
            return cut_value(A, x)
        except Exception:
            pass
    return cut_value(A, steepest_descent(A, seed_spin))

def induced(A, S):
    return A[np.ix_(S, S)]

def deloc_and_seed(As):
    d = np.abs(As).sum(axis=1)
    L = np.diag(d) - As
    w, V = eigh(L)
    v = V[:, -1]; v = v / np.linalg.norm(v)
    pr = 1.0 / np.sum(v**4)
    deloc = pr / len(v)
    seed = np.sign(v); seed[seed == 0] = 1.0
    return deloc, seed

def adjacency_lists(A):
    return [np.nonzero(A[i])[0] for i in range(A.shape[0])]

def sample_support(adj, n, k):
    start = int(RNG.integers(n))
    S = [start]; seen = {start}; frontier = list(adj[start])
    RNG.shuffle(frontier)
    while len(S) < k and frontier:
        v = frontier.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v))
        nb = list(adj[v]); RNG.shuffle(nb)
        frontier.extend(nb)
    return S

A, n, m = load_gset(INSTANCE)
adj = adjacency_lists(A)
print(f"instance={INSTANCE}  n={n} m={m}  k={K_SUPPORT}  "
      f"supports={N_SUPPORTS}  randoms={N_RANDOM}  neal={USE_NEAL}")

rows = []
for t in range(N_SUPPORTS):
    S = sample_support(adj, n, K_SUPPORT)
    if len(S) < max(8, K_SUPPORT // 4):
        continue
    As = induced(A, S)
    if np.count_nonzero(As) == 0:
        continue
    deloc, seed = deloc_and_seed(As)
    cut_spec = solve(As, seed)
    rc = np.array([solve(As, RNG.choice([-1.0, 1.0], len(S))) for _ in range(N_RANDOM)])
    mu, sd = rc.mean(), rc.std()
    z = (cut_spec - mu) / sd if sd > 1e-9 else 0.0
    rel = (cut_spec - mu) / mu * 100 if mu != 0 else 0.0
    rows.append(dict(t=t, k=len(S), deloc=deloc, cut_spec=cut_spec,
                     rand_mu=mu, z=z, rel=rel))
    print(f"  support {t:2d}  k={len(S):4d}  deloc={deloc:.3f}  "
          f"cut_spec={cut_spec:9.1f}  rand_mu={mu:9.1f}  z={z:+5.2f}  rel%={rel:+6.2f}")

dl = np.array([r["deloc"] for r in rows])
rl = np.array([r["rel"] for r in rows])
zz = np.array([r["z"] for r in rows])
rho_r, p_r = spearmanr(dl, rl)
rho_z, p_z = spearmanr(dl, zz)
print(f"\nSpearman(deloc, rel%) rho={rho_r:+.3f} p={p_r:.3g}")
print(f"Spearman(deloc, z)    rho={rho_z:+.3f} p={p_z:.3g}")

os.makedirs("results", exist_ok=True)
base = os.path.splitext(os.path.basename(INSTANCE))[0]
mode = "neal" if USE_NEAL else "sd"
out = rf"results\deloc_probe_{base}_{mode}.csv"
with open(out, "w", newline="") as f:
    wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wri.writeheader(); wri.writerows(rows)
print(f"wrote {out}")

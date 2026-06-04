# Does a SPECTRAL warm-start save neal sweep budget on sub-QUBOs?
# Curve: mean cut vs num_sweeps, cold neal vs spectral-warm neal.
# Self-check: confirms the warm-start is actually being used.
import sys, csv, os
import numpy as np
from scipy.linalg import eigh

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else r"data\gset\G11.txt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 400
N_SUP   = 6
READS   = 4
SWEEPS  = [1, 5, 10, 20, 40, 80, 160]
RNG = np.random.default_rng(20260604)

from dwave.samplers import SimulatedAnnealingSampler
SAMP = SimulatedAnnealingSampler()

def load_gset(path):
    with open(path) as f:
        n, m = map(int, f.readline().split()[:2])
        A = np.zeros((n, n))
        for line in f:
            p = line.split()
            if len(p) < 2: continue
            u, v = int(p[0])-1, int(p[1])-1
            w = float(p[2]) if len(p) > 2 else 1.0
            A[u, v] = w; A[v, u] = w
    return A, n, m

def cut_value(A, s): return 0.25 * (A.sum() - s @ (A @ s))

def induced(A, S): return A[np.ix_(S, S)]

def deloc_and_seed(As):
    d = np.abs(As).sum(1); L = np.diag(d) - As
    w, V = eigh(L); v = V[:, -1]; v = v/np.linalg.norm(v)
    seed = np.sign(v); seed[seed == 0] = 1.0
    return seed

def to_J(As):
    n = As.shape[0]; J = {}
    iu = np.triu_indices(n, 1)
    for u, v in zip(*iu):
        if As[u, v] != 0.0:
            J[(int(u), int(v))] = float(As[u, v])
    return J

def neal_cut(As, J, sweeps, init=None, rng_seed=0):
    n = As.shape[0]
    kw = dict(num_reads=1, num_sweeps=int(max(sweeps, 1)), seed=int(rng_seed))
    if init is not None:
        kw['initial_states'] = {i: int(init[i]) for i in range(n)}
        kw['initial_states_generator'] = 'none'
    smp = SAMP.sample_ising({}, J, **kw).first.sample
    x = np.array([smp[i] for i in range(n)], float)
    return cut_value(As, x)

def adjacency_lists(A): return [np.nonzero(A[i])[0] for i in range(A.shape[0])]

def sample_support(adj, n, k):
    start = int(RNG.integers(n)); S = [start]; seen = {start}
    frontier = list(adj[start]); RNG.shuffle(frontier)
    while len(S) < k and frontier:
        v = frontier.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v))
        nb = list(adj[v]); RNG.shuffle(nb); frontier.extend(nb)
    return S

A, n, m = load_gset(INSTANCE); adj = adjacency_lists(A)
print(f"instance={INSTANCE} n={n} m={m} k={K} supports={N_SUP} reads={READS}")

# ---- WARM-START SELF-CHECK on first valid support ----
S0 = sample_support(adj, n, K); As0 = induced(A, S0); J0 = to_J(As0)
seed0 = deloc_and_seed(As0)
seed_cut = cut_value(As0, seed0)
warm_lo = neal_cut(As0, J0, 1, init=seed0, rng_seed=0)
cold_lo = neal_cut(As0, J0, 1, init=None,  rng_seed=0)
print("\n--- warm-start verification (support 0, 1 sweep) ---")
print(f"  spectral seed's own cut : {seed_cut:.1f}")
print(f"  neal WARM @1 sweep      : {warm_lo:.1f}   (should be ~seed cut if warm-start taking)")
print(f"  neal COLD @1 sweep      : {cold_lo:.1f}   (should be much lower / random)")
ok = abs(warm_lo - seed_cut) < 0.25 * abs(seed_cut - cold_lo) + 1
print(f"  => warm-start {'IS' if ok else 'is NOT'} taking\n")

# ---- sweep curves averaged across supports ----
cold = {s: [] for s in SWEEPS}; spec = {s: [] for s in SWEEPS}
for t in range(N_SUP):
    S = sample_support(adj, n, K); As = induced(A, S); J = to_J(As)
    if len(J) == 0: continue
    seed = deloc_and_seed(As)
    for s in SWEEPS:
        c = np.mean([neal_cut(As, J, s, init=None, rng_seed=r)   for r in range(READS)])
        w = np.mean([neal_cut(As, J, s, init=seed, rng_seed=r)   for r in range(READS)])
        cold[s].append(c); spec[s].append(w)

print(f"{'sweeps':>7s}{'cold_mean':>11s}{'spec_mean':>11s}{'gap':>8s}{'gap%':>8s}")
rows = []
for s in SWEEPS:
    cm, sm = np.mean(cold[s]), np.mean(spec[s])
    gap = sm - cm; gpct = gap/cm*100 if cm else 0.0
    rows.append(dict(sweeps=s, cold=cm, spec=sm, gap=gap, gap_pct=gpct))
    print(f"{s:7d}{cm:11.1f}{sm:11.1f}{gap:+8.1f}{gpct:+8.2f}")

os.makedirs("results", exist_ok=True)
base = os.path.splitext(os.path.basename(INSTANCE))[0]
out = rf"results\warmstart_{base}.csv"
with open(out, "w", newline="") as f:
    wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wri.writeheader(); wri.writerows(rows)
print(f"wrote {out}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4.5))
    plt.plot(SWEEPS, [r['cold'] for r in rows], 'o-', label='cold neal')
    plt.plot(SWEEPS, [r['spec'] for r in rows], 's-', label='spectral warm-start')
    plt.xscale('log'); plt.xlabel('neal num_sweeps'); plt.ylabel('mean sub-QUBO cut')
    plt.title(f'Spectral warm-start vs cold neal ? {base} (k={K})')
    plt.legend(frameon=False); plt.grid(alpha=0.3); plt.tight_layout()
    p = rf"results\warmstart_{base}.png"; plt.savefig(p, dpi=150); print(f"wrote {p}")
except Exception as e:
    print("(plot skipped:", e, ")")

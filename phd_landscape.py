# Landscape characterization of Max-Cut sub-QUBOs: exact (small k) + sampled (real k),
# with a calibration check that the sampled estimators match exact ground truth.
import sys, csv, os
import numpy as np
from itertools import product

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else r"data\gset\G11.txt"
RNG = np.random.default_rng(20260604)

# ---------- load + induced sub-QUBO ----------
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

def adj_lists(A): return [np.nonzero(A[i])[0] for i in range(A.shape[0])]

def sample_support(adj, n, k):
    start = int(RNG.integers(n)); S=[start]; seen={start}
    fr=list(adj[start]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v)); nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S

def induced(A, S): return A[np.ix_(S, S)]
def cut(A, s): return 0.25*(A.sum() - s @ (A @ s))

def descend(A, s0):  # steepest-descent (ascent on cut) to a local optimum
    s = s0.copy().astype(float)
    while True:
        g = (A @ s)*s; i = int(np.argmax(g))
        if g[i] <= 1e-12: break
        s[i] = -s[i]
    return s

def hamming(a, b): return int(np.sum(a != b))

# ---------- STAGE A: exact landscape on small k ----------
def exact_landscape(A):
    k = A.shape[0]; N = 1 << k
    cuts = np.empty(N)
    configs = np.empty((N, k))
    for idx in range(N):
        s = np.array([1.0 if (idx>>b)&1 else -1.0 for b in range(k)])
        configs[idx] = s; cuts[idx] = cut(A, s)
    # local optima: no single flip improves cut
    is_lo = np.ones(N, bool)
    for idx in range(N):
        s = configs[idx]; g = (A @ s)*s
        if np.any(g > 1e-9): is_lo[idx] = False
    lo_idx = np.where(is_lo)[0]
    best = cuts.max(); gopt_deg = int(np.sum(cuts >= best - 1e-9))
    # basin: descend every config, record which LO it reaches (tie-break lowest index)
    def descend_idx(idx):
        s = configs[idx].copy()
        while True:
            g = (A @ s)*s; i = int(np.argmax(g))
            if g[i] <= 1e-12: break
            s[i] = -s[i]
        return int(sum((1<<b) for b in range(k) if s[b] > 0))
    basin = {}
    for idx in range(N):
        b = descend_idx(idx); basin[b] = basin.get(b, 0) + 1
    # FDC: correlation of cut with -distance to nearest global optimum
    gopt = [configs[i] for i in range(N) if cuts[i] >= best-1e-9]
    dist = np.array([min(hamming(configs[i], g) for g in gopt) for i in range(N)])
    fdc = np.corrcoef(cuts, -dist)[0,1]
    # exact barrier between two best distinct optima (min over paths of max energy drop)
    return dict(k=k, n_lo=len(lo_idx), gopt_deg=gopt_deg, best=best,
                fdc=fdc, n_basins=len(basin),
                mean_basin=N/len(basin), exact=True)

# ---------- STAGE B: sampled estimators (run on ANY k) ----------
def sampled_landscape(A, n_restart=400):
    k = A.shape[0]
    los = {}   # rounded cut -> count, plus store representative configs
    reps = []
    for _ in range(n_restart):
        s = descend(A, RNG.choice([-1.0,1.0], k))
        c = round(float(cut(A, s)), 6)
        los[c] = los.get(c, 0) + 1
        if len(reps) < 60: reps.append((c, s))
    best = max(los)
    # distinct local-optimum cut-VALUES found (proxy for #optima; undercounts degenerate)
    n_lo_values = len(los)
    # estimate global-optimum basin fraction = fraction of restarts hitting best
    gbasin_frac = los[best] / n_restart
    # FDC vs best-found config
    gconf = [s for c,s in reps if c >= best-1e-9]
    if gconf:
        cs = np.array([c for c,_ in reps]); ds = np.array([min(hamming(s,g) for g in gconf) for _,s in reps])
        fdc = np.corrcoef(cs, -ds)[0,1] if len(cs)>2 else float('nan')
    else:
        fdc = float('nan')
    return dict(k=k, n_lo_values=n_lo_values, gbasin_frac=gbasin_frac,
                best=best, fdc=fdc, exact=False)

A, n, m = load_gset(INSTANCE)
adj = adj_lists(A)
base = os.path.splitext(os.path.basename(INSTANCE))[0]
print(f"=== {base}  n={n} m={m} ===\n")

# ---- STAGE A + calibration: small-k exact vs sampled on SAME instances ----
print("STAGE A  exact landscape on small k, with sampled-estimator calibration")
print(f"{'k':>3s}{'n_lo':>6s}{'gopt_deg':>9s}{'fdc_ex':>8s}{'basins':>8s} | "
      f"{'lo_val_s':>9s}{'gbasin_s':>9s}{'fdc_s':>7s}")
calib = []
for k in [10, 12, 14, 16, 18]:
    S = sample_support(adj, n, k)
    if len(S) < k: continue
    As = induced(A, S)
    ex = exact_landscape(As)
    sm = sampled_landscape(As, n_restart=400)
    calib.append((ex, sm))
    print(f"{k:3d}{ex['n_lo']:6d}{ex['gopt_deg']:9d}{ex['fdc']:8.3f}{ex['n_basins']:8d} | "
          f"{sm['n_lo_values']:9d}{sm['gbasin_frac']:9.3f}{sm['fdc']:7.3f}")

# ---- STAGE B: sampled on real k ----
print("\nSTAGE B  sampled landscape on real k")
print(f"{'k':>5s}{'n_lo_values':>12s}{'gbasin_frac':>12s}{'fdc':>8s}{'best':>9s}")
rows = []
for k in [100, 200, 400, 600]:
    S = sample_support(adj, n, k)
    if len(S) < max(8, k//4): continue
    As = induced(A, S)
    sm = sampled_landscape(As, n_restart=400)
    rows.append(sm)
    print(f"{sm['k']:5d}{sm['n_lo_values']:12d}{sm['gbasin_frac']:12.3f}{sm['fdc']:8.3f}{sm['best']:9.1f}")

os.makedirs("results", exist_ok=True)
with open(rf"results\landscape_{base}.csv","w",newline="") as f:
    wri = csv.writer(f); wri.writerow(["stage","k","n_lo","gopt_deg","fdc","extra"])
    for ex,sm in calib:
        wri.writerow(["A_exact",ex["k"],ex["n_lo"],ex["gopt_deg"],round(ex["fdc"],4),ex["n_basins"]])
        wri.writerow(["A_sampled",sm["k"],sm["n_lo_values"],"",round(sm["fdc"],4) if sm["fdc"]==sm["fdc"] else "",round(sm["gbasin_frac"],4)])
    for sm in rows:
        wri.writerow(["B_sampled",sm["k"],sm["n_lo_values"],"",round(sm["fdc"],4) if sm["fdc"]==sm["fdc"] else "",round(sm["gbasin_frac"],4)])
print(f"\nwrote results\\landscape_{base}.csv")

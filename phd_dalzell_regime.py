# Characterize AQLS sub-QUBOs on the Dalzell case-(a)/case-(b) boundary:
# how dense are near-optimal solutions? Dense => classical wins (case b);
# rare/isolated => quantum speedup regime (case a).
import sys, csv, os
import numpy as np
from scipy.linalg import eigh

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else r"data\gset\G11.txt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N_SUP   = 8
N_SAMPLE = 4000     # random configs for density estimate
N_DESCENT = 200     # random starts for local-optimum spread
RNG = np.random.default_rng(20260604)

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

def steepest_descent(A, s0, mi=20000):
    s = s0.copy().astype(float)
    for _ in range(mi):
        g = (A @ s) * s; i = int(np.argmax(g))
        if g[i] <= 1e-12: break
        s[i] = -s[i]
    return s

def induced(A, S): return A[np.ix_(S, S)]
def adj_lists(A): return [np.nonzero(A[i])[0] for i in range(A.shape[0])]

def sample_support(adj, n, k):
    start = int(RNG.integers(n)); S=[start]; seen={start}
    fr=list(adj[start]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v))
        nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S

A, n, m = load_gset(INSTANCE); adj = adj_lists(A)
print(f"instance={INSTANCE} n={n} m={m} k={K} supports={N_SUP}\n")
print(f"{'sup':>3s}{'k':>5s}{'best':>8s}{'rand_ratio':>11s}{'LO_spread':>10s}{'near90%':>9s}{'near95%':>9s}{'verdict':>9s}")

rows=[]
for t in range(N_SUP):
    S = sample_support(adj, n, K); As = induced(A, S); k = len(S)
    if np.count_nonzero(As)==0: continue
    # many random configs -> cost distribution
    costs = np.array([cut_value(As, RNG.choice([-1.0,1.0],k)) for _ in range(N_SAMPLE)])
    # local optima from random starts
    los = np.array([cut_value(As, steepest_descent(As, RNG.choice([-1.0,1.0],k)))
                    for _ in range(N_DESCENT)])
    best = los.max()
    rand_ratio = costs.mean()/best                 # how good is a random config
    lo_spread = (best - los.mean())/best            # gap from mean LO to best (small=dense)
    # density of near-optimal LOCAL OPTIMA (this is the Dalzell-relevant decay)
    near90 = np.mean(los >= 0.90*best)              # fraction of LOs within 10% of best
    near95 = np.mean(los >= 0.95*best)              # within 5%
    # case-(b) if near-optima abundant AND easy to reach
    verdict = "case-b" if near95 > 0.25 else ("mixed" if near95 > 0.05 else "case-a")
    rows.append(dict(sup=t,k=k,best=best,rand_ratio=rand_ratio,lo_spread=lo_spread,
                     near90=near90,near95=near95,verdict=verdict))
    print(f"{t:3d}{k:5d}{best:8.0f}{rand_ratio:11.3f}{lo_spread:10.3f}"
          f"{near90:9.2f}{near95:9.2f}{verdict:>9s}")

print(f"\n--- {os.path.basename(INSTANCE)} summary ---")
nb = sum(1 for r in rows if r['verdict']=='case-b')
na = sum(1 for r in rows if r['verdict']=='case-a')
nm = sum(1 for r in rows if r['verdict']=='mixed')
print(f"case-b (classical wins, dense near-optima): {nb}/{len(rows)}")
print(f"mixed:                                       {nm}/{len(rows)}")
print(f"case-a (quantum-relevant, rare optima):      {na}/{len(rows)}")
print(f"mean near95%={np.mean([r['near95'] for r in rows]):.3f}  "
      f"mean rand_ratio={np.mean([r['rand_ratio'] for r in rows]):.3f}")

os.makedirs("results", exist_ok=True)
base = os.path.splitext(os.path.basename(INSTANCE))[0]
out = rf"results\dalzell_regime_{base}.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {out}")

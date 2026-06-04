# Landscape v2: canonical optima counting (fold +/-1 flip), FDC vs single best,
# global-optimum degeneracy as first-class metric, restart-budget saturation check.
import sys, csv, os
import numpy as np

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else r"data\gset\G11.txt"
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

def adj_lists(A): return [np.nonzero(A[i])[0] for i in range(A.shape[0])]
def sample_support(adj, n, k):
    start=int(RNG.integers(n)); S=[start]; seen={start}; fr=list(adj[start]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v)); nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S
def induced(A,S): return A[np.ix_(S,S)]
def cut(A,s): return 0.25*(A.sum() - s @ (A @ s))
def descend(A,s0):
    s=s0.copy().astype(float)
    while True:
        g=(A@s)*s; i=int(np.argmax(g))
        if g[i]<=1e-12: break
        s[i]=-s[i]
    return s
def canon(s):  # fold global +/-1 flip: force first spin +1
    s = s if s[0] > 0 else -s
    return (s > 0).tobytes()
def hamming(a,b): return int(np.sum(a!=b))

# ---------- exact (small k) ----------
def exact_landscape(A):
    k=A.shape[0]; N=1<<k
    configs=np.array([[1.0 if (idx>>b)&1 else -1.0 for b in range(k)] for idx in range(N)])
    cuts=np.array([cut(A,configs[i]) for i in range(N)])
    canon_lo=set()
    for i in range(N):
        s=configs[i]; g=(A@s)*s
        if not np.any(g>1e-9): canon_lo.add(canon(s))
    best=cuts.max()
    gopt=[configs[i] for i in range(N) if cuts[i]>=best-1e-9]
    gopt_canon=len({canon(g) for g in gopt})
    dist=np.array([min(hamming(configs[i],g) for g in gopt) for i in range(N)])
    fdc=np.corrcoef(cuts,-dist)[0,1]
    return dict(k=k,n_lo=len(canon_lo),gopt_deg=gopt_canon,fdc=fdc)

# ---------- sampled (any k), canonical + budget saturation ----------
def sampled_landscape(A,n_restart=600):
    k=A.shape[0]
    seen_lo=set(); best=-1e18; best_cfg=None; samples=[]
    sat=[]  # (restarts, distinct canonical optima so far)
    for r in range(1,n_restart+1):
        s=descend(A,RNG.choice([-1.0,1.0],k))
        c=float(cut(A,s)); seen_lo.add(canon(s)); samples.append((c,s))
        if c>best: best=c; best_cfg=s.copy()
        if r in (50,100,200,400,600): sat.append((r,len(seen_lo)))
    # canonical degeneracy of best
    gdeg=len({canon(s) for c,s in samples if c>=best-1e-9})
    # FDC vs single best config
    cs=np.array([c for c,_ in samples]); ds=np.array([hamming(s,best_cfg) for _,s in samples])
    fdc=np.corrcoef(cs,-ds)[0,1] if len(cs)>2 and ds.std()>0 else float('nan')
    gbasin=sum(1 for c,_ in samples if c>=best-1e-9)/n_restart
    return dict(k=k,n_lo_canon=len(seen_lo),gopt_deg=gdeg,gbasin=gbasin,fdc=fdc,best=best,sat=sat)

A,n,m=load_gset(INSTANCE); adj=adj_lists(A)
base=os.path.splitext(os.path.basename(INSTANCE))[0]
print(f"=== {base}  n={n} m={m} ===\n")
print("STAGE A  exact vs sampled-canonical calibration")
print(f"{'k':>3s}{'n_lo_ex':>8s}{'gdeg_ex':>8s}{'fdc_ex':>7s} | {'n_lo_s':>7s}{'gdeg_s':>7s}{'fdc_s':>7s}")
for k in [10,12,14,16,18]:
    S=sample_support(adj,n,k)
    if len(S)<k: continue
    As=induced(A,S); ex=exact_landscape(As); sm=sampled_landscape(As,600)
    print(f"{k:3d}{ex['n_lo']:8d}{ex['gopt_deg']:8d}{ex['fdc']:7.3f} | "
          f"{sm['n_lo_canon']:7d}{sm['gopt_deg']:7d}{sm['fdc']:7.3f}")

print("\nSTAGE B  sampled on real k (n_lo_canon is a FLOOR; check saturation)")
print(f"{'k':>5s}{'n_lo_canon':>11s}{'gopt_deg':>9s}{'gbasin':>8s}{'fdc':>7s}{'best':>9s}  saturation(restarts:optima)")
rows=[]
for k in [100,200,400,600]:
    S=sample_support(adj,n,k)
    if len(S)<max(8,k//4): continue
    As=induced(A,S); sm=sampled_landscape(As,600); rows.append(sm)
    sat=" ".join(f"{r}:{c}" for r,c in sm['sat'])
    print(f"{sm['k']:5d}{sm['n_lo_canon']:11d}{sm['gopt_deg']:9d}{sm['gbasin']:8.3f}"
          f"{sm['fdc']:7.3f}{sm['best']:9.1f}  {sat}")

os.makedirs("results",exist_ok=True)
with open(rf"results\landscape_v2_{base}.csv","w",newline="") as f:
    wri=csv.writer(f); wri.writerow(["k","n_lo_canon","gopt_deg","gbasin","fdc","best"])
    for sm in rows: wri.writerow([sm["k"],sm["n_lo_canon"],sm["gopt_deg"],round(sm["gbasin"],4),
                                  round(sm["fdc"],4) if sm["fdc"]==sm["fdc"] else "",sm["best"]])
print(f"\nwrote results\\landscape_v2_{base}.csv")

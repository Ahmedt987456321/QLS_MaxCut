# FDC fix: correlate cut vs distance-to-best over configs spanning the full
# quality range (random walks of increasing length from best-found), not just
# local optima. Calibrated against exact FDC on small k.
import sys, os
import numpy as np
RNG = np.random.default_rng(20260604)

def load_gset(path):
    with open(path) as f:
        n,m=map(int,f.readline().split()[:2]); A=np.zeros((n,n))
        for line in f:
            p=line.split()
            if len(p)<2: continue
            u,v=int(p[0])-1,int(p[1])-1; w=float(p[2]) if len(p)>2 else 1.0
            A[u,v]=w; A[v,u]=w
    return A,n,m
def adj_lists(A): return [np.nonzero(A[i])[0] for i in range(A.shape[0])]
def sample_support(adj,n,k):
    st=int(RNG.integers(n)); S=[st]; seen={st}; fr=list(adj[st]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v)); nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S
def induced(A,S): return A[np.ix_(S,S)]
def cut(A,s): return 0.25*(A.sum()-s@(A@s))
def descend(A,s0):
    s=s0.copy().astype(float)
    while True:
        g=(A@s)*s; i=int(np.argmax(g))
        if g[i]<=1e-12: break
        s[i]=-s[i]
    return s
def hamming(a,b): return int(np.sum(a!=b))

def exact_fdc(A):
    k=A.shape[0]; N=1<<k
    cfg=np.array([[1.0 if (idx>>b)&1 else -1.0 for b in range(k)] for idx in range(N)])
    cuts=np.array([cut(A,cfg[i]) for i in range(N)]); best=cuts.max()
    gopt=[cfg[i] for i in range(N) if cuts[i]>=best-1e-9]
    dist=np.array([min(hamming(cfg[i],g) for g in gopt) for i in range(N)])
    return np.corrcoef(cuts,-dist)[0,1]

def find_best(A,n_restart=200):
    best=-1e18; bc=None
    for _ in range(n_restart):
        s=descend(A,RNG.choice([-1.0,1.0],A.shape[0])); c=cut(A,s)
        if c>best: best=c; bc=s.copy()
    return bc,best

def fdc_fixed(A,bc,n_samples=800):
    # walk increasing #flips away from best to span quality range
    k=A.shape[0]; cuts=[]; dists=[]
    for _ in range(n_samples):
        nflip=int(RNG.integers(1,k+1))           # 1..k random flips
        idx=RNG.choice(k,size=nflip,replace=False)
        s=bc.copy(); s[idx]=-s[idx]
        cuts.append(cut(A,s)); dists.append(hamming(s,bc))
    cuts=np.array(cuts); dists=np.array(dists)
    return np.corrcoef(cuts,-dists)[0,1] if dists.std()>0 else float('nan')

A,n,m=load_gset(sys.argv[1] if len(sys.argv)>1 else r"data\gset\G11.txt")
adj=adj_lists(A); base=os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv)>1 else "G11"
print(f"=== {base}  n={n} m={m} ===\n")
print("CALIBRATION  exact FDC vs fixed-sampled FDC (small k)")
print(f"{'k':>3s}{'fdc_exact':>11s}{'fdc_fixed':>11s}")
for k in [10,12,14,16,18]:
    S=sample_support(adj,n,k)
    if len(S)<k: continue
    As=induced(A,S); ex=exact_fdc(As); bc,_=find_best(As,200); fx=fdc_fixed(As,bc,800)
    print(f"{k:3d}{ex:11.3f}{fx:11.3f}")

print("\nLARGE k  fixed FDC (does the funnel survive to real sizes?)")
print(f"{'k':>5s}{'fdc_fixed':>11s}{'best':>9s}")
for k in [100,200,400,600]:
    S=sample_support(adj,n,k)
    if len(S)<max(8,k//4): continue
    As=induced(A,S); bc,best=find_best(As,200); fx=fdc_fixed(As,bc,800)
    print(f"{k:5d}{fx:11.3f}{best:9.1f}")

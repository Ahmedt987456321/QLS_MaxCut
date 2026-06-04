# FDC v3: descent-trajectory sampling. Record (cut, distance-to-best) at every
# step along many steepest-descent paths -> real landscape points spanning the
# full quality range. Calibrated against exact FDC on small k.
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
def hamming(a,b): return int(np.sum(a!=b))

def descend_record(A,s0):
    # steepest descent; yield (cut, config) at each step including start and end
    s=s0.copy().astype(float); traj=[]
    while True:
        traj.append((cut(A,s), s.copy()))
        g=(A@s)*s; i=int(np.argmax(g))
        if g[i]<=1e-12: break
        s[i]=-s[i]
    return traj

def best_via_descents(A,n_restart):
    best=-1e18; bc=None; allpts=[]
    for _ in range(n_restart):
        traj=descend_record(A,RNG.choice([-1.0,1.0],A.shape[0]))
        allpts.extend(traj)
        if traj[-1][0]>best: best=traj[-1][0]; bc=traj[-1][1].copy()
    return bc,best,allpts

def exact_fdc(A):
    k=A.shape[0]; N=1<<k
    cfg=np.array([[1.0 if (idx>>b)&1 else -1.0 for b in range(k)] for idx in range(N)])
    cuts=np.array([cut(A,cfg[i]) for i in range(N)]); best=cuts.max()
    gopt=[cfg[i] for i in range(N) if cuts[i]>=best-1e-9]
    dist=np.array([min(hamming(cfg[i],g) for g in gopt) for i in range(N)])
    return np.corrcoef(cuts,-dist)[0,1]

def fdc_traj(bc,allpts,cap=4000):
    if len(allpts)>cap:
        idx=RNG.choice(len(allpts),cap,replace=False); pts=[allpts[i] for i in idx]
    else: pts=allpts
    cuts=np.array([c for c,_ in pts]); dist=np.array([hamming(s,bc) for _,s in pts])
    return np.corrcoef(cuts,-dist)[0,1] if dist.std()>0 and cuts.std()>0 else float('nan')

A,n,m=load_gset(sys.argv[1] if len(sys.argv)>1 else r"data\gset\G11.txt")
adj=adj_lists(A); base=os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv)>1 else "G11"
print(f"=== {base}  n={n} m={m} ===\n")
print("CALIBRATION  exact FDC vs trajectory-sampled FDC (small k)")
print(f"{'k':>3s}{'fdc_exact':>11s}{'fdc_traj':>10s}")
for k in [10,12,14,16,18]:
    S=sample_support(adj,n,k)
    if len(S)<k: continue
    As=induced(A,S); ex=exact_fdc(As)
    bc,_,pts=best_via_descents(As,200); fx=fdc_traj(bc,pts)
    print(f"{k:3d}{ex:11.3f}{fx:10.3f}")

print("\nLARGE k  trajectory FDC (does the funnel survive to real sizes?)")
print(f"{'k':>5s}{'fdc_traj':>10s}{'best':>9s}{'n_pts':>8s}")
for k in [100,200,400,600]:
    S=sample_support(adj,n,k)
    if len(S)<max(8,k//4): continue
    As=induced(A,S); bc,best,pts=best_via_descents(As,200); fx=fdc_traj(bc,pts)
    print(f"{k:5d}{fx:10.3f}{best:9.1f}{len(pts):8d}")

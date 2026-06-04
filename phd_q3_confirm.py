# Q3 confirmation: do toroidal (G11) and random (G22) sub-QUBOs differ in
# ANNEALING DYNAMICS? 30 reps, Mann-Whitney U test, raw AND % late-gain.
import sys, os
import numpy as np
from scipy.stats import mannwhitneyu
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
def cut(A,s): return 0.25*(A.sum()-s@(A@s))
def connected_support(adj,n,k):
    st=int(RNG.integers(n)); S=[st]; seen={st}; fr=list(adj[st]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v)); nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S
def induced(A,S): return A[np.ix_(S,S)]

def annealed_run(A, sweeps=200, Tend=0.05):
    k=A.shape[0]; s=RNG.choice([-1.0,1.0],k)
    T0=max(1.0,np.abs(A).sum()/k); Ts=T0*(Tend/T0)**(np.linspace(0,1,sweeps))
    e=[]; acc=[]
    for sw in range(sweeps):
        T=Ts[sw]; a=0
        for i in RNG.permutation(k):
            dcut=((A[i]@s)*s[i])
            if dcut>0 or RNG.random()<np.exp(dcut/max(T,1e-9)):
                s[i]=-s[i]; a+=1
        e.append(cut(A,s)); acc.append(a/k)
    e=np.array(e); best=e.max(); half=e[len(e)//2]
    return dict(best=best,
                late_gain_pct=(best-half)/best*100 if best!=0 else 0.0,
                late_gain_raw=best-half,
                acc_late=np.mean(acc[-len(e)//4:]),
                frac2plateau=np.argmax(e>=0.99*best)/len(e),
                downs=np.mean(np.diff(e)<0))

def collect(path,k,reps=30):
    A,n,m=load_gset(path); adj=adj_lists(A); runs=[]
    for _ in range(reps):
        S=connected_support(adj,n,k)
        if len(S)<max(8,k//4): continue
        runs.append(annealed_run(induced(A,S)))
    return runs

K=int(sys.argv[1]) if len(sys.argv)>1 else 400
g11=collect(r"data\gset\G11.txt",K,30)
g22=collect(r"data\gset\G22.txt",K,30)
print(f"=== Q3 confirmation  k={K}  reps={len(g11)}/{len(g22)} (G11/G22) ===\n")
metrics=["late_gain_pct","late_gain_raw","acc_late","frac2plateau","downs"]
print(f"{'metric':>14s}{'G11_mean':>10s}{'G22_mean':>10s}{'U_p':>10s}{'verdict':>10s}")
for mt in metrics:
    a=np.array([r[mt] for r in g11]); b=np.array([r[mt] for r in g22])
    try: _,p=mannwhitneyu(a,b,alternative='two-sided')
    except Exception: p=float('nan')
    verdict="DIFF" if p<0.05 else "ns"
    print(f"{mt:>14s}{a.mean():10.3f}{b.mean():10.3f}{p:10.4g}{verdict:>10s}")
print("\nDIFF = toroidal vs random differ significantly (p<0.05) in that dynamic.")
print("Watch late_gain_raw: if it stays DIFF, the toroidal effect is real, not a")
print("small-cut percentage artifact. If only late_gain_pct is DIFF, it's arithmetic.")

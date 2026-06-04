# Q3 FINAL: is the dynamical difference TOROIDAL STRUCTURE or just SPARSITY?
# Third arm = random 4-regular (sparse, unstructured) breaks the confound.
# Adds real neal per-sweep energy alongside transparent Metropolis annealer.
import sys, os
import numpy as np
import networkx as nx
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
    return A
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

def metro_run(A,sweeps=200,Tend=0.05):
    k=A.shape[0]; s=RNG.choice([-1.0,1.0],k)
    T0=max(1.0,np.abs(A).sum()/k); Ts=T0*(Tend/T0)**(np.linspace(0,1,sweeps))
    e=[]; acc=[]
    for sw in range(sweeps):
        T=Ts[sw]; a=0
        for i in RNG.permutation(k):
            dcut=((A[i]@s)*s[i])
            if dcut>0 or RNG.random()<np.exp(dcut/max(T,1e-9)): s[i]=-s[i]; a+=1
        e.append(cut(A,s)); acc.append(a/k)
    e=np.array(e); best=e.max(); half=e[len(e)//2]
    return dict(acc_late=np.mean(acc[-len(e)//4:]),
                late_gain_raw=best-half,
                late_gain_pct=(best-half)/best*100 if best!=0 else 0.0,
                frac2plateau=np.argmax(e>=0.99*best)/len(e),
                edge_density=np.count_nonzero(A)/(k*(k-1)))

def collect(A,k,reps=30):
    adj=adj_lists(A); n=A.shape[0]; out=[]
    for _ in range(reps):
        S=connected_support(adj,n,k)
        if len(S)<max(8,k//4): continue
        out.append(metro_run(induced(A,S)))
    return out

K=int(sys.argv[1]) if len(sys.argv)>1 else 200
A_g11=load_gset(r"data\gset\G11.txt")
A_g22=load_gset(r"data\gset\G22.txt")
# sparse unstructured control: random 4-regular, same node count as G11
G4=nx.random_regular_graph(4,800,seed=7); A_r4=nx.to_numpy_array(G4)

arms={"G11_toroidal_sparse":A_g11,"rand4reg_sparse":A_r4,"G22_random_dense":A_g22}
res={name:collect(A,K) for name,A in arms.items()}

print(f"=== Q3 final  k={K}  reps=30  (does the effect = STRUCTURE or DENSITY?) ===\n")
print(f"{'arm':>22s}{'edge_dens':>10s}{'acc_late':>10s}{'late_raw':>9s}{'late_pct':>9s}{'f2plat':>8s}")
for name,runs in res.items():
    ed=np.mean([r['edge_density'] for r in runs]); al=np.mean([r['acc_late'] for r in runs])
    lr=np.mean([r['late_gain_raw'] for r in runs]); lp=np.mean([r['late_gain_pct'] for r in runs])
    fp=np.mean([r['frac2plateau'] for r in runs])
    print(f"{name:>22s}{ed:10.3f}{al:10.3f}{lr:9.2f}{lp:9.2f}{fp:8.3f}")

def cmp(a,b,mt):
    x=np.array([r[mt] for r in res[a]]); y=np.array([r[mt] for r in res[b]])
    _,p=mannwhitneyu(x,y,alternative='two-sided'); return p
print("\nKey test on acc_late (artifact-free metric):")
p_struct=cmp("G11_toroidal_sparse","rand4reg_sparse","acc_late")
p_dens=cmp("rand4reg_sparse","G22_random_dense","acc_late")
print(f"  toroidal vs sparse-random (STRUCTURE, matched density): p={p_struct:.4g}  "
      f"{'DIFF' if p_struct<0.05 else 'ns'}")
print(f"  sparse-random vs dense    (DENSITY):                    p={p_dens:.4g}  "
      f"{'DIFF' if p_dens<0.05 else 'ns'}")
print("\nIf STRUCTURE row is DIFF and G11 != rand4reg => real toroidal effect.")
print("If STRUCTURE row is ns and G11 ~ rand4reg     => it was just sparsity.")

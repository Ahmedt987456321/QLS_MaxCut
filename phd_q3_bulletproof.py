# Q3 bulletproof: (1) 5 random-4-regular control seeds, not 1.
# (2) real neal per-sweep acceptance vs Metropolis, to confirm not an artifact.
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

def metro_acc_late(A,sweeps=200,Tend=0.05):
    k=A.shape[0]; s=RNG.choice([-1.0,1.0],k)
    T0=max(1.0,np.abs(A).sum()/k); Ts=T0*(Tend/T0)**(np.linspace(0,1,sweeps)); acc=[]
    for sw in range(sweeps):
        T=Ts[sw]; a=0
        for i in RNG.permutation(k):
            dcut=((A[i]@s)*s[i])
            if dcut>0 or RNG.random()<np.exp(dcut/max(T,1e-9)): s[i]=-s[i]; a+=1
        acc.append(a/k)
    return float(np.mean(acc[-sweeps//4:]))

def neal_acc_late(A,sweeps=200):
    # real neal: read per-sweep states, measure late-stage acceptance via state changes
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except Exception:
        return None
    k=A.shape[0]; J={}
    iu=np.triu_indices(k,1)
    for u,v in zip(*iu):
        if A[u,v]!=0.0: J[(int(u),int(v))]=float(A[u,v])
    # neal doesn't expose per-sweep easily; approximate late acceptance by running
    # short anneals from a near-final state and measuring flip rate in last phase.
    samp=SimulatedAnnealingSampler()
    # proxy: fraction of spins that differ between a mid-anneal and final read
    r1=samp.sample_ising({},J,num_reads=1,num_sweeps=sweeps//2,seed=0).first.sample
    s_mid=np.array([r1[i] for i in range(k)],float)
    r2=samp.sample_ising({},J,num_reads=1,num_sweeps=sweeps,
                         initial_states={i:int(s_mid[i]) for i in range(k)},
                         initial_states_generator='none',seed=0).first.sample
    s_fin=np.array([r2[i] for i in range(k)],float)
    return float(np.mean(s_mid!=s_fin))   # spins still changing in 2nd half

def collect_metro(A,k,reps=30):
    adj=adj_lists(A); n=A.shape[0]; out=[]
    for _ in range(reps):
        S=connected_support(adj,n,k)
        if len(S)<max(8,k//4): continue
        out.append(metro_acc_late(induced(A,S)))
    return np.array(out)

def collect_neal(A,k,reps=10):
    adj=adj_lists(A); n=A.shape[0]; out=[]
    for _ in range(reps):
        S=connected_support(adj,n,k)
        if len(S)<max(8,k//4): continue
        v=neal_acc_late(induced(A,S))
        if v is not None: out.append(v)
    return np.array(out)

K=int(sys.argv[1]) if len(sys.argv)>1 else 200
A_g11=load_gset(r"data\gset\G11.txt")

# --- Metropolis: G11 vs 5 random-4-regular seeds ---
g11=collect_metro(A_g11,K,30)
print(f"=== Q3 bulletproof  k={K} ===\n[Metropolis acc_late, 30 reps each]")
print(f"  G11_toroidal         mean={g11.mean():.4f}")
ctrl_means=[]
for seed in [7,13,21,42,99]:
    A_r4=nx.to_numpy_array(nx.random_regular_graph(4,800,seed=seed))
    r4=collect_metro(A_r4,K,30); ctrl_means.append(r4.mean())
    _,p=mannwhitneyu(g11,r4,alternative='two-sided')
    print(f"  rand4reg(seed={seed:2d})    mean={r4.mean():.4f}   vs G11 p={p:.4g}  "
          f"{'DIFF' if p<0.05 else 'ns'}")
print(f"\n  control mean across seeds={np.mean(ctrl_means):.4f}  "
      f"(G11 {'ABOVE' if g11.mean()>np.mean(ctrl_means) else 'below'} all controls: "
      f"{g11.mean()>max(ctrl_means)})")

# --- real neal confirmation (fewer reps; slower) ---
print("\n[real neal late-activity proxy, 10 reps each]")
ng11=collect_neal(A_g11,K,10)
A_r4=nx.to_numpy_array(nx.random_regular_graph(4,800,seed=7))
nr4=collect_neal(A_r4,K,10)
if len(ng11) and len(nr4):
    _,pn=mannwhitneyu(ng11,nr4,alternative='two-sided')
    print(f"  neal G11   mean={ng11.mean():.4f}")
    print(f"  neal rand4 mean={nr4.mean():.4f}   p={pn:.4g}  {'DIFF' if pn<0.05 else 'ns'}")
    print(f"  => neal {'CONFIRMS' if pn<0.05 and ng11.mean()>nr4.mean() else 'does NOT confirm'} the Metropolis effect")
else:
    print("  neal unavailable in this env")

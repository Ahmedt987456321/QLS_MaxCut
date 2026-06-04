# Observe the landscape THROUGH the solver: instrument annealing dynamics on
# real sub-QUBOs. Logs energy-vs-sweep, acceptance rate, basin changes, late-
# improvement. Compares AQLS-style (connected/frustrated) vs random supports,
# and G11 (toroidal) vs G22 (random). No enumeration, no sampling-convergence.
import sys, csv, os
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
def cut(A,s): return 0.25*(A.sum()-s@(A@s))

# ---- two support types ----
def connected_support(adj,n,k):   # AQLS-style: BFS ball (connected neighbourhood)
    st=int(RNG.integers(n)); S=[st]; seen={st}; fr=list(adj[st]); RNG.shuffle(fr)
    while len(S)<k and fr:
        v=fr.pop()
        if v in seen: continue
        seen.add(v); S.append(int(v)); nb=list(adj[v]); RNG.shuffle(nb); fr.extend(nb)
    return S
def random_support(adj,n,k):      # control: k random vertices (ignores structure)
    return list(RNG.choice(n,size=min(k,n),replace=False))
def induced(A,S): return A[np.ix_(S,S)]

# ---- transparent instrumented simulated annealing (Metropolis, geometric schedule) ----
def annealed_run(A, sweeps=200, T0=None, Tend=0.05):
    k=A.shape[0]; s=RNG.choice([-1.0,1.0],k)
    # set T0 from coupling scale if not given
    if T0 is None: T0=max(1.0, np.abs(A).sum()/k)
    Ts=T0*(Tend/T0)**(np.linspace(0,1,sweeps))
    energies=[]; acc_hist=[]; basin_changes=0
    last_lo=None
    def local_opt(state):  # descend to nearest LO to identify basin
        x=state.copy()
        while True:
            g=(A@x)*x; i=int(np.argmax(g))
            if g[i]<=1e-12: break
            x[i]=-x[i]
        return tuple((x>0).astype(int)) if x[0]>0 else tuple((-x>0).astype(int))
    for sw in range(sweeps):
        T=Ts[sw]; acc=0
        order=RNG.permutation(k)
        for i in order:
            # delta in CUT for flipping i (we maximize cut <=> minimize -cut)
            dcut = ((A[i]@s)*s[i])   # change in cut if we flip i
            # accept if improves cut, or with Metropolis prob on the worsening
            if dcut>0 or RNG.random()<np.exp(dcut/max(T,1e-9)):
                s[i]=-s[i]; acc+=1
        energies.append(cut(A,s)); acc_hist.append(acc/k)
        if sw in (int(sweeps*0.25),int(sweeps*0.5),int(sweeps*0.75),sweeps-1):
            lo=local_opt(s)
            if last_lo is not None and lo!=last_lo: basin_changes+=1
            last_lo=lo
    e=np.array(energies)
    final=e[-1]; best=e.max()
    # metrics
    halfway=e[len(e)//2]
    late_gain=(best-halfway)/best*100 if best!=0 else 0.0   # improvement in 2nd half
    # "jaggedness": how often energy goes DOWN sweep-to-sweep (barrier-hopping signature)
    downs=np.mean(np.diff(e)<0)
    # time to reach 99% of final-best
    thresh=0.99*best; reached=np.argmax(e>=thresh)
    frac_to_plateau=reached/len(e)
    return dict(final=final,best=best,late_gain=late_gain,downs=downs,
                frac_to_plateau=frac_to_plateau,acc_early=np.mean(acc_hist[:len(e)//4]),
                acc_late=np.mean(acc_hist[-len(e)//4:]),basin_changes=basin_changes)

def profile(A,adj,n,k,kind,reps=5):
    out=[]
    for _ in range(reps):
        S = connected_support(adj,n,k) if kind=="connected" else random_support(adj,n,k)
        if len(S)<max(8,k//4): continue
        out.append(annealed_run(induced(A,S),sweeps=200))
    keys=out[0].keys()
    return {kk:np.mean([o[kk] for o in out]) for kk in keys}

INSTANCE=sys.argv[1] if len(sys.argv)>1 else r"data\gset\G11.txt"
K=int(sys.argv[2]) if len(sys.argv)>2 else 400
A,n,m=load_gset(INSTANCE); adj=adj_lists(A)
base=os.path.splitext(os.path.basename(INSTANCE))[0]
print(f"=== {base}  n={n} m={m}  k={K} (annealing dynamics, 200 sweeps, 5 reps) ===\n")
print(f"{'support':>11s}{'frac2plateau':>13s}{'late_gain%':>11s}{'down_steps':>11s}"
      f"{'acc_early':>10s}{'acc_late':>9s}{'basin_chg':>10s}")
for kind in ["connected","random"]:
    p=profile(A,adj,n,K,kind)
    print(f"{kind:>11s}{p['frac_to_plateau']:13.3f}{p['late_gain']:11.2f}{p['downs']:11.3f}"
          f"{p['acc_early']:10.3f}{p['acc_late']:9.3f}{p['basin_changes']:10.1f}")
print("\nReading guide:")
print(" frac2plateau LOW + late_gain LOW  => easy funnel (neal arrives early, coasts)")
print(" late_gain HIGH + basin_chg HIGH   => rugged barrier-hopping (neal keeps escaping)")
print(" down_steps HIGH                   => non-monotone descent = crossing barriers")

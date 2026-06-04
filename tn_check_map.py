"""Check Ising(J,h) reproduces QUBO energy for ALL assignments on a tiny sub-QUBO."""
import numpy as np, itertools
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected
from src.qubo import build_local_qubo, qubo_energy

G=load_gset("data/gset/G11.txt"); rng=np.random.default_rng(0)
x=random_cut(G,rng); gc=GainCache(); x,gc,_,_=one_flip_ls(G,x,gc); gc.update(G,x)
S=select_frustrated_connected(G,gc,8,rng=rng,x=x)   # tiny: 8 vertices, 256 assignments
Q=build_local_qubo(G,x,S)
S=list(S); idx={v:i for i,v in enumerate(S)}; n=len(S)

# build J,h with the standard map (no sign flip yet)
h=np.zeros(n); Jd={}
for (a,b),w in Q.items():
    if a==b: h[idx[a]]+=w/2.0
    else:
        i,j=idx[a],idx[b]; key=(min(i,j),max(i,j))
        Jd[key]=Jd.get(key,0.0)+w/4.0
        h[idx[a]]+=w/4.0; h[idx[b]]+=w/4.0
# constant offset
const=0.0
for (a,b),w in Q.items():
    if a==b: const+=w/2.0
    else: const+=w/4.0

def ising_E(s):  # s in {-1,+1}
    e=const
    for (i,j),J in Jd.items(): e+=J*s[i]*s[j]
    for i in range(n): e+=h[i]*s[i]
    return e

# check all assignments
maxerr=0.0
for bits in itertools.product([0,1],repeat=n):
    sol={S[i]:bits[i] for i in range(n)}
    eq=qubo_energy(Q,sol,S)
    s=[2*b-1 for b in bits]
    ei=ising_E(s)
    maxerr=max(maxerr,abs(eq-ei))
print(f"max |QUBO - Ising| over all {2**n} assignments: {maxerr:.6f}")
print("CONVERSION CORRECT" if maxerr<1e-6 else "CONVERSION WRONG - formula bug")

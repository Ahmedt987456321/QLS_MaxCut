import numpy as np, json, subprocess
from src.graph import load_gset
from src.local_search import random_cut, one_flip_ls
from src.gain_cache import GainCache
from src.selectors import select_frustrated_connected
from src.qubo import build_local_qubo, qubo_energy
from src.backends import backend_exact

G=load_gset("data/gset/G11.txt"); rng=np.random.default_rng(0)
x=random_cut(G,rng); gc=GainCache(); x,gc,_,_=one_flip_ls(G,x,gc); gc.update(G,x)
S=select_frustrated_connected(G,gc,14,rng=rng,x=x)
Q=build_local_qubo(G,x,S)
S=list(S); idx={v:i for i,v in enumerate(S)}; n=len(S)
e_exact=qubo_energy(Q,backend_exact(Q,S),S)
print(f"exact: {e_exact:.1f}")

h=np.zeros(n); Jd={}
for (a,b),w in Q.items():
    if a==b: h[idx[a]]+=w/2.0
    else:
        i,j=idx[a],idx[b]; key=(min(i,j),max(i,j))
        Jd[key]=Jd.get(key,0.0)+w/4.0
        h[idx[a]]+=w/4.0; h[idx[b]]+=w/4.0
edges=[[i+1,j+1] for (i,j) in Jd]
Jbase=[Jd[(i,j)] for (i,j) in Jd]

for js in (-1,1):
    for hs in (-1,1):
        J=[js*v for v in Jbase]; hh=[hs*v for v in h]
        json.dump({"n":n,"edges":edges,"J":J,"h":hh}, open("tn_in.json","w"))
        subprocess.run(["julia","solve_tn_worker.jl"],check=True,
                       capture_output=True)
        out=json.load(open("tn_out.json")); sp=out["config"]
        for fl in (0,1):
            sol={S[i]:(1-sp[i] if fl else sp[i]) for i in range(n)}
            e=qubo_energy(Q,sol,S)
            if abs(e-e_exact)<1e-6:
                print(f"MATCH: J*{js}, h*{hs}, flip={fl}, energy={e:.1f}")

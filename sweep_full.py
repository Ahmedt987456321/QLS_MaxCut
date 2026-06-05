import numpy as np, networkx as nx
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.stats import mannwhitneyu
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.selectors import select_fiedler, select_frustrated_connected
from src.adaptive_qls import adaptive_qls
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

def load_dimacs(path):
    G=nx.Graph()
    with open(path) as f: lines=f.readlines()
    for line in lines[1:]:
        p=line.split()
        if len(p)==3: G.add_edge(int(p[0]),int(p[1]),weight=float(p[2]))
    return G

def load_mtx(path):
    G=nx.Graph()
    with open(path) as f: lines=f.readlines()
    dl=[l.strip() for l in lines if not l.startswith('%') and l.strip()]
    for line in dl[1:]:
        p=line.split()
        if len(p)>=2:
            u,v=int(p[0]),int(p[1])
            if u!=v: G.add_edge(u,v)
    return nx.convert_node_labels_to_integers(G)

# full set spanning beta_e; k ~ min(default, n//2)
INSTANCES=[
 ("G11",          load_gset,  "data/gset/G11.txt",            400),
 ("G12",          load_gset,  "data/gset/G12.txt",            400),
 ("G13",          load_gset,  "data/gset/G13.txt",            400),
 ("G32",          load_gset,  "data/gset/G32.txt",            400),
 ("G33",          load_gset,  "data/gset/G33.txt",            400),
 ("G34",          load_gset,  "data/gset/G34.txt",            400),
 ("G48",          load_gset,  "data/gset/G48.txt",            400),
 ("G49",          load_gset,  "data/gset/G49.txt",            400),
 ("G50",          load_gset,  "data/gset/G50.txt",            400),
 ("delaunay_n10", load_mtx,   "data/delaunay/delaunay_n10.mtx",500),
 ("delaunay_n11", load_mtx,   "data/delaunay/delaunay_n11.mtx",500),
 ("toruspm3-8-50",load_dimacs,"data/dimacs/toruspm3-8-50.dat", 256),
 ("torusg3-8",    load_dimacs,"data/dimacs/torusg3-8.dat",     256),
 ("toruspm3-15-50",load_dimacs,"data/dimacs/toruspm3-15-50.dat",640),
 ("torusg3-15",   load_dimacs,"data/dimacs/torusg3-15.dat",    640),
 ("G14",          load_gset,  "data/gset/G14.txt",            400),
 ("G15",          load_gset,  "data/gset/G15.txt",            400),
 ("G18",          load_gset,  "data/gset/G18.txt",            400),
 ("G1",           load_gset,  "data/gset/G1.txt",             400),
 ("G22",          load_gset,  "data/gset/G22.txt",            640),
]
SEEDS=list(range(8)); BUDGET=25; ALPHA=0.05

def beta_e(G):
    nodes=list(G.nodes()); idx={v:i for i,v in enumerate(nodes)}
    r,c,val=[],[],[]; deg=np.zeros(len(nodes))
    for u,v in G.edges():
        i,j=idx[u],idx[v]; r+=[i,j];c+=[j,i];val+=[-1.,-1.];deg[i]+=1;deg[j]+=1
    L=sp.diags(deg)+sp.csr_matrix((val,(r,c)),shape=(len(nodes),)*2)
    _,vec=spla.eigsh(L,k=2,which="SM",tol=1e-3,maxiter=3000)
    order=np.argsort(vec[:,1]); S=set(nodes[i] for i in order[:len(nodes)//2])
    return sum(1 for u,v in G.edges() if (u in S)^(v in S))/G.number_of_edges()

def run(G, sel, k):
    cuts, stabs = [], []
    for s in SEEDS:
        log=[]
        def w(G,gc,kk,pool=None,rng=None,x=None):
            S=sel(G,gc,kk,pool=pool,rng=rng,x=x); log.append(frozenset(S)); return S
        w.__name__=sel.__name__
        xb,_=adaptive_qls(G,budget_seconds=BUDGET,selector=w,backend=NEAL,
                          k_min=k,k_max=k,n_reads=100,best_known=None,
                          seed=s,acceptance="lookahead")
        cuts.append(compute_cut_value(G,xb))
        js=[len(a&b)/len(a|b) for a,b in zip(log[:-1],log[1:]) if a|b]
        stabs.append(np.mean(js) if js else float("nan"))
    return np.array(cuts,float), np.array(stabs,float)

print(f"{'instance':<15}{'n':>6}{'beta_e':>8}{'F_mob':>7}{'F_med':>9}{'C_med':>9}{'result':>16}")
print("-"*70)
recs=[]
for name,loader,path,k in INSTANCES:
    try:
        G=loader(path)
    except Exception as e:
        print(f"{name:<15}  LOAD FAILED: {e}"); continue
    k=min(k, G.number_of_nodes()//2)
    be=beta_e(G)
    fc,fs=run(G,select_fiedler,k)
    gc_,gs=run(G,select_frustrated_connected,k)
    fmed, cmed = float(np.median(fc)), float(np.median(gc_))
    if np.array_equal(fc,gc_):
        p=1.0
    else:
        try: _,p=mannwhitneyu(fc,gc_,alternative="two-sided")
        except ValueError: p=1.0
    if p<ALPHA:
        win=("Fiedler" if fmed>cmed else "FConn"); res=f"{win} (p={p:.3f})"
    else:
        win="tie"; res=f"tie (p={p:.2f})"
    recs.append(dict(name=name,n=G.number_of_nodes(),be=be,fmob=fs.mean(),
                     fmed=fmed,cmed=cmed,win=win,p=p))
    fm = f"{fs.mean():.2f}" if np.isfinite(fs.mean()) else " nan"
    print(f"{name:<15}{G.number_of_nodes():>6}{be:>8.3f}{fm:>7}{fmed:>9.0f}{cmed:>9.0f}{res:>16}")

print(f"\nfallback_count: {getattr(select_fiedler,'_fallback_count',0)}")

# analysis on SIGNIFICANT decisions only (ties excluded)
sig=[r for r in recs if r["win"]!="tie"]
fied=[r for r in sig if r["win"]=="Fiedler"]; fcon=[r for r in sig if r["win"]=="FConn"]
print(f"\nsignificant decisions: {len(sig)}/{len(recs)}  (ties dropped: {len(recs)-len(sig)})")
if fied:
    print(f"  Fiedler-wins: beta_e {min(r['be'] for r in fied):.3f}-{max(r['be'] for r in fied):.3f} | "
          f"mob {min(r['fmob'] for r in fied):.2f}-{max(r['fmob'] for r in fied):.2f}")
if fcon:
    print(f"  FConn-wins:   beta_e {min(r['be'] for r in fcon):.3f}-{max(r['be'] for r in fcon):.3f} | "
          f"mob {min(r['fmob'] for r in fcon):.2f}-{max(r['fmob'] for r in fcon):.2f}")
# does either predictor SEPARATE the significant winners? (overlap = fails)
if fied and fcon:
    be_overlap = max(r['be'] for r in fied) >= min(r['be'] for r in fcon)
    mob_vals_f=[r['fmob'] for r in fied if np.isfinite(r['fmob'])]
    mob_vals_c=[r['fmob'] for r in fcon if np.isfinite(r['fmob'])]
    mob_overlap = (max(mob_vals_f) >= min(mob_vals_c)) if mob_vals_f and mob_vals_c else True
    print(f"  beta_e separates winners? {'NO (ranges overlap)' if be_overlap else 'YES'}")
    print(f"  mobility separates winners? {'NO (ranges overlap)' if mob_overlap else 'YES'}")

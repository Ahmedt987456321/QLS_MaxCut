import numpy as np
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls, compute_cut_value
from src.adaptive_qls import adaptive_qls, _overlap_components
from src.selectors import select_fiedler, select_frustrated_connected
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

CONFIG = {
    "G11": dict(bks=564,  selector=select_fiedler,             k=400, budget=30),
    "G14": dict(bks=3064, selector=select_frustrated_connected, k=400, budget=60),
}
M = 50   # poor local optima to sample per instance
summary = {}

for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt")
    nodes = list(G.nodes())

    # ?? strong reference solution (the "escape target") ???????????
    ref, _ = adaptive_qls(G, budget_seconds=cfg["budget"],
                          selector=cfg["selector"], backend=NEAL,
                          k_min=cfg["k"], k_max=cfg["k"], n_reads=100,
                          best_known=cfg["bks"], seed=0, acceptance="lookahead")
    ref_cut = compute_cut_value(G, ref)

    # ?? sample poor local optima, measure disagreement structure ??
    rng = np.random.default_rng(1)
    ncomps, fracs, largest, totdis, big = [], [], [], [], []
    for _ in range(M):
        x = random_cut(G, rng); gc = GainCache()
        x, gc, _, _ = one_flip_ls(G, x, gc)
        comps = _overlap_components(G, x, ref)   # connected disagreement comps, largest first
        if not comps:
            continue
        sizes = [len(c) for c in comps]; td = sum(sizes)
        ncomps.append(len(sizes)); largest.append(sizes[0])
        fracs.append(sizes[0] / td); totdis.append(td)
        big.append(sum(1 for s in sizes if s >= 5))

    med = lambda a: float(np.median(a))
    summary[inst] = dict(ref=ref_cut, n=len(nodes), k=cfg["k"],
                         td=med(totdis), nc=med(ncomps), lg=med(largest),
                         fr=med(fracs), big=med(big))
    s = summary[inst]
    print(f"\n=== {inst}  (ref cut {ref_cut:.0f} / BKS {cfg['bks']}, "
          f"{cfg['selector'].__name__}, k={cfg['k']}) ===")
    print(f"  median total disagreement   : {s['td']:.0f} vtx "
          f"({100*s['td']/s['n']:.0f}% of graph)")
    print(f"  median # disagreement comps : {s['nc']:.0f}   "
          f"(# with >=5 vtx: {s['big']:.0f})")
    print(f"  median largest comp size    : {s['lg']:.0f}   "
          f"(<= k? {'YES' if s['lg'] <= cfg['k'] else 'NO'})")
    print(f"  COHERENCE (frac in largest) : {s['fr']:.3f}   "
          f"(1.0 = one coherent blob, ~0 = fragmented)")

print("\n--- CONTRAST ---")
for inst in summary:
    s = summary[inst]
    print(f"  {inst}: coherence={s['fr']:.3f}  #comps={s['nc']:.0f}  largest={s['lg']:.0f}")

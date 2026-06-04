import itertools, numpy as np
from src.graph import load_gset
from src.local_search import compute_cut_value
from src.adaptive_qls import adaptive_qls, _overlap_components
from src.selectors import select_frustrated_connected
try:
    from src.backends import backend_neal as NEAL
except ImportError:
    from src.backends import get_backend; NEAL = get_backend("neal")

CONFIG = {"G11": dict(bks=564, k=400), "G14": dict(bks=3064, k=400)}
N = 8            # good solutions per instance (same selector)
BUDGET = 20
SEL = select_frustrated_connected   # MATCHED: same selector on both instances

summary = {}
for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt")
    nodes = list(G.nodes())

    sols, cuts = [], []
    for s in range(N):
        x, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=SEL, backend=NEAL,
                           k_min=cfg["k"], k_max=cfg["k"], n_reads=100,
                           best_known=cfg["bks"], seed=s, acceptance="lookahead")
        sols.append(x); cuts.append(compute_cut_value(G, x))

    ncomps, fracs, largest, totdis = [], [], [], []
    for a, b in itertools.combinations(range(N), 2):
        comps = _overlap_components(G, sols[a], sols[b])   # spin-flip-aligned
        if not comps:                       # identical solutions
            ncomps.append(0); fracs.append(0.0); largest.append(0); totdis.append(0)
            continue
        sizes = [len(c) for c in comps]; td = sum(sizes)
        ncomps.append(len(sizes)); largest.append(sizes[0])
        fracs.append(sizes[0]/td); totdis.append(td)

    med = lambda a: float(np.median(a))
    summary[inst] = dict(cutmin=min(cuts), cutmax=max(cuts), bks=cfg["bks"],
                        n=len(nodes), td=med(totdis), nc=med(ncomps),
                        lg=med(largest), fr=med(fracs))
    s = summary[inst]
    print(f"\n=== {inst}  (FConn, {N} sols, cut range {s['cutmin']:.0f}-{s['cutmax']:.0f} "
          f"/ BKS {cfg['bks']}) ===")
    print(f"  median pairwise disagreement : {s['td']:.0f} vtx "
          f"({100*s['td']/s['n']:.0f}% of graph)")
    print(f"  median # disagreement comps  : {s['nc']:.0f}")
    print(f"  median largest comp size     : {s['lg']:.0f}")
    print(f"  COHERENCE (frac in largest)  : {s['fr']:.3f}")

print("\n--- CONTRAST (same selector, matched quality tier) ---")
for inst in summary:
    s = summary[inst]
    print(f"  {inst}: coherence={s['fr']:.3f}  #comps={s['nc']:.0f}  "
          f"largest={s['lg']:.0f}  cut={s['cutmin']:.0f}-{s['cutmax']:.0f}")

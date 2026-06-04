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
N = 6; BUDGET = 20; SEL = select_frustrated_connected

for inst, cfg in CONFIG.items():
    G = load_gset(f"data/gset/{inst}.txt")
    nodes = list(G.nodes())
    sols = []
    for s in range(N):
        x, _ = adaptive_qls(G, budget_seconds=BUDGET, selector=SEL, backend=NEAL,
                           k_min=cfg["k"], k_max=cfg["k"], n_reads=100,
                           best_known=cfg["bks"], seed=s, acceptance="lookahead")
        sols.append(x)

    comp_costs, neutral_frac, comp_sizes = [], [], []
    for a, b in itertools.combinations(range(N), 2):
        xa, xb = sols[a], sols[b]
        # align spin-flip (mirror xb if it disagrees with xa on majority)
        if sum(1 for v in nodes if xa[v]==xb[v]) < len(nodes)/2:
            xb = {v: 1-xb[v] for v in nodes}
        base = compute_cut_value(G, xa)
        comps = _overlap_components(G, xa, xb)
        for c in comps:
            xc = dict(xa)
            for v in c: xc[v] = 1 - xc[v]
            delta = compute_cut_value(G, xc) - base   # cost of flipping THIS comp alone
            comp_costs.append(delta); comp_sizes.append(len(c))
            neutral_frac.append(1.0 if abs(delta) <= 2 else 0.0)  # near-neutral

    cc = np.array(comp_costs); sz = np.array(comp_sizes)
    print(f"\n=== {inst} ({len(cc)} components across {N} solutions) ===")
    print(f"  per-component flip cost: median={np.median(cc):.1f}  "
          f"mean={cc.mean():.1f}  min={cc.min():.0f}  max={cc.max():.0f}")
    print(f"  fraction near-neutral (|delta|<=2): {np.mean(neutral_frac):.2f}")
    print(f"  component sizes: median={np.median(sz):.0f}  max={sz.max():.0f}")
    pos = np.mean(cc > 0); zero = np.mean(np.abs(cc) <= 2); neg = np.mean(cc < -2)
    print(f"  improving: {pos:.2f}   neutral: {zero:.2f}   worsening: {neg:.2f}")

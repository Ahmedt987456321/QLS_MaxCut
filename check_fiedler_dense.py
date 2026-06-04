"""Does select_fiedler run on dense/random graphs? Quick probe before full grid."""
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
for name in ["G1", "G22"]:
    G = load_gset(f"data/gset/{name}.txt")
    try:
        x, m = adaptive_qls(G, budget_seconds=10, selector=select_fiedler,
                            backend=neal, k_min=160, k_max=160, n_reads=100,
                            seed=0, best_known=None)
        print(f"{name}: Fiedler RAN, cut={m.best_cut}, calls={m.qls_calls}, "
              f"fallbacks={getattr(adaptive_qls,'_fallback_count',0)}")
    except Exception as e:
        print(f"{name}: Fiedler ERRORED -> {type(e).__name__}: {e}")

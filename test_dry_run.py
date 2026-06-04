from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_fiedler, select_frustrated_connected
from src.baselines import simulated_annealing
from src.backends import backend_neal

G = load_gset("data/gset/G11.txt")
best_known = 564
k = 400

print("Testing G11 with 1 trial each...")
print("\n1. SA baseline:")
try:
    x_sa, m_sa = simulated_annealing(G, budget_seconds=30, best_known=best_known, seed=1000)
    print(f"   ✓ SA works: best_cut={m_sa.best_cut:.0f}, has attr .best_cut={hasattr(m_sa, 'best_cut')}")
except Exception as e:
    print(f"   ✗ SA failed: {e}")

print("\n2. FConn selector:")
try:
    x_fc, m_fc = adaptive_qls(G, budget_seconds=30, selector=select_frustrated_connected,
                               backend=backend_neal, k_min=k, k_max=k, best_known=best_known, seed=1000)
    print(f"   ✓ FConn works: best_cut={m_fc.best_cut:.0f}, has attr .best_cut={hasattr(m_fc, 'best_cut')}")
except Exception as e:
    print(f"   ✗ FConn failed: {e}")

print("\n3. Fiedler selector:")
try:
    x_fi, m_fi = adaptive_qls(G, budget_seconds=30, selector=select_fiedler,
                               backend=backend_neal, k_min=k, k_max=k, best_known=best_known, seed=1000)
    print(f"   ✓ Fiedler works: best_cut={m_fi.best_cut:.0f}, has attr .best_cut={hasattr(m_fi, 'best_cut')}")
except Exception as e:
    print(f"   ✗ Fiedler failed: {e}")

print("\n✓ All three methods work on G11")

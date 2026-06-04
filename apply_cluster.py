with open("src/adaptive_qls.py", encoding="utf-8") as f:
    s = f.read()

# 1. add networkx import (needed for components) if not present
if "import networkx" not in s:
    s = s.replace("import numpy as np",
                  "import numpy as np\nimport networkx as nx")

# 2. add cluster-move helpers before the adaptive_qls def
helpers = '''
def _overlap_components(G, xA, xB):
    """Connected components of the disagreement set between xA, xB,
    after spin-reversal alignment. Largest-first list of vertex sets."""
    nodes = list(G.nodes())
    agree = sum(1 for v in nodes if xA[v] == xB[v])
    if agree < len(nodes) / 2:
        xB = {v: 1 - xB[v] for v in nodes}
    N = [v for v in nodes if xA[v] != xB[v]]
    if not N:
        return []
    HN = G.subgraph(N)
    return sorted((set(c) for c in nx.connected_components(HN)),
                  key=len, reverse=True)


def _cluster_move(G, x, x_other):
    """Flip the best-improving disagreement component of x vs x_other.
    Returns (new_x, gain). gain=0 and x unchanged if no improvement."""
    comps = _overlap_components(G, x, x_other)
    if not comps:
        return dict(x), 0.0
    base = compute_cut_value(G, x)
    best_x, best_gain = dict(x), 0.0
    for comp in comps[:8]:   # try the largest few components
        xc = dict(x)
        for v in comp:
            xc[v] = 1 - xc[v]
        g = compute_cut_value(G, xc) - base
        if g > best_gain:
            best_gain, best_x = g, xc
    return best_x, best_gain


'''
s = s.replace("def adaptive_qls(", helpers + "def adaptive_qls(", 1)

# 3. add cluster_moves param to signature
s = s.replace(
    "                 acceptance='improvement',\n                 T_initial=2.0, T_min=0.001, cooling=0.995):",
    "                 acceptance='improvement',\n                 T_initial=2.0, T_min=0.001, cooling=0.995,\n                 cluster_moves=False, cluster_interval=20):")

# 4. add cluster-move trigger inside the main loop, right after the EMA update
#    (Phase 7). We use a call counter on the function.
anchor = "        # \u2500\u2500 Phase 7: EMA update \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        ema_esc = alpha * float(delta > 0) + (1 - alpha) * ema_esc"
if anchor not in s:
    # fallback: match on the EMA update line alone
    anchor = "        ema_esc = alpha * float(delta > 0) + (1 - alpha) * ema_esc"
cluster_block = anchor + '''

        # \u2500\u2500 Phase 7b: overlap cluster move (toggle) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if cluster_moves:
            adaptive_qls._cm_counter = getattr(
                adaptive_qls, '_cm_counter', 0) + 1
            if adaptive_qls._cm_counter % cluster_interval == 0 and len(pool) >= 2:
                # best pool member distinct from current x
                x_other = max(pool, key=lambda p: compute_cut_value(G, p))
                x_cm, g_cm = _cluster_move(G, x, x_other)
                if g_cm > 0:
                    x = x_cm
                    gc.invalidate()
                    cut_cm = compute_cut_value(G, x)
                    if cut_cm > best_cut:
                        best_cut = cut_cm
                        x_best = dict(x)
                        if best_known:
                            metrics.check_time_to_target(best_cut, best_known)'''
s = s.replace(anchor, cluster_block, 1)

with open("src/adaptive_qls.py", "w", encoding="utf-8") as f:
    f.write(s)
print("integrated cluster_moves toggle")

with open("src/adaptive_qls.py", encoding="utf-8") as f:
    s = f.read()

old = """                # best pool member distinct from current x
                x_other = max(pool, key=lambda p: compute_cut_value(G, p))"""
new = """                # MOST-DIFFERENT pool member (max disagreement) -- supplies
                # the most exploitable overlap structure for the cluster move
                def _disagree(p):
                    return sum(1 for v in nodes if p[v] != x[v])
                x_other = max(pool, key=_disagree)"""

if old not in s:
    print("ANCHOR NOT FOUND -- pasting current trigger block for inspection")
else:
    s = s.replace(old, new)
    with open("src/adaptive_qls.py", "w", encoding="utf-8") as f:
        f.write(s)
    print("switched cluster move to most-different pool member")

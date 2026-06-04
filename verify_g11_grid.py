import numpy as np
from collections import Counter
from src.graph import load_gset

G = load_gset("data/gset/G11.txt")
n = G.number_of_nodes()
print(f"n={n}  m={G.number_of_edges()}  4-regular={all(d==4 for _,d in G.degree())}")

# neighbour offsets (node ids are 1..n); the small offset is the row step (1),
# the large offset is the column count. Wraparound shows up as n-1 and n-cols.
offs = Counter()
for v in G.nodes():
    for u in G.neighbors(v):
        offs[abs(u - v)] += 1
common = offs.most_common(6)
print("most common |neighbour offsets|:", common)

small = sorted(o for o,_ in common if 0 < o <= n//2)
if len(small) >= 2:
    cols = small[1]                 # 1 = row step, second-smallest = column count
    rows = n // cols
    print(f"=> inferred grid: {cols} cols x {rows} rows  (cols*rows={cols*rows})")
    # verify: every node v has neighbours v?1 (mod row) and v?cols (mod n)
    ok = True
    for v in list(G.nodes())[:50]:
        nb = set(G.neighbors(v))
        exp = {((v-1+d) % n)+1 for d in (1,-1,cols,-cols)}
        if nb != exp: ok = False; break
    print("toroidal-offset structure verified:", ok)

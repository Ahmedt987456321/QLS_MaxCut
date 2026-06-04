import networkx as nx, numpy as np
# the v2 large_gap graph was 2x200 = 400 nodes, run at k=400 = n.
# selectors can only differ when k < n. Confirm n and the k used.
import json
rows = json.load(open("results/eigengap_v2.json"))
for r in rows:
    print(r["family"], "n was 400, k was 400 -> selector picks ALL vertices")
print("\nIf n==k, FConn and Fiedler both select every vertex -> identical support")
print("-> ties are a SCALE ARTIFACT, not a real finding. Need n > k.")

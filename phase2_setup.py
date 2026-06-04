"""Phase 2 setup check: verify minimum-weight perfect matching is available
and works on a small torus. We need this to:
1. Verify the matching representation of 2D Ising ground states
2. Compute the actual faithfulness gap empirically
3. Identify the minimum-weight defect path crossing the Fiedler boundary"""
import networkx as nx
import numpy as np

# test 1: networkx minimum weight matching
print("Test 1: networkx min-weight matching")
G = nx.Graph()
G.add_edges_from([(0,1,{"weight":3}),(1,2,{"weight":1}),(0,2,{"weight":2})])
matching = nx.min_weight_matching(G)
print(f"  min_weight_matching available: True")
print(f"  result on triangle: {matching}")

# test 2: build a small 4x4 torus with +-1 weights
print("\nTest 2: small 4x4 torus with +-1 weights")
L, M = 4, 4
T = nx.grid_2d_graph(L, M, periodic=True)
T = nx.convert_node_labels_to_integers(T)
rng = np.random.default_rng(42)
for u,v in T.edges():
    T[u][v]["weight"] = int(rng.choice([-1,1]))
print(f"  n={T.number_of_nodes()}, edges={T.number_of_edges()}")
print(f"  edge weights (first 8): "
      f"{[(u,v,T[u][v]['weight']) for u,v in list(T.edges())[:8]]}")

# test 3: can we identify frustrated plaquettes?
# a plaquette (face) is frustrated if the product of its 4 edge signs is -1
print("\nTest 3: frustrated plaquettes on 4x4 torus")
frustrated = []
for i in range(L):
    for j in range(M):
        # four edges of the plaquette at (i,j)
        corners = [i*M+j, i*M+(j+1)%M, ((i+1)%L)*M+j, ((i+1)%L)*M+(j+1)%M]
        # four edges: bottom, right, top, left
        e1 = T[corners[0]][corners[1]]["weight"]
        e2 = T[corners[1]][corners[3]]["weight"]
        e3 = T[corners[2]][corners[3]]["weight"]
        e4 = T[corners[0]][corners[2]]["weight"]
        product = e1 * e2 * e3 * e4
        if product == -1:
            frustrated.append((i,j))
print(f"  frustrated plaquettes: {len(frustrated)} of {L*M}")
print(f"  positions: {frustrated}")
print(f"  (expected ~{L*M//2} for 50% random signs)")

print("\nPhase 2 setup: OK")
print("Next: build the dual lattice and match frustrated plaquettes")

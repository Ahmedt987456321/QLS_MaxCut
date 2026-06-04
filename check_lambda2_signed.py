import networkx as nx, numpy as np
# build one toroidal-like graph, then flip some edge signs, check if
# nx.algebraic_connectivity (what the router uses) changes.
G = nx.grid_2d_graph(20, 40, periodic=True)
G = nx.convert_node_labels_to_integers(G)
lam2_a = nx.algebraic_connectivity(G, method="lanczos")
# assign random +-1 weights (signs) -- does algebraic_connectivity see them?
rng = np.random.default_rng(0)
for u,v in G.edges():
    G[u][v]["weight"] = int(rng.choice([-1, 1]))
lam2_b = nx.algebraic_connectivity(G, weight="weight", method="lanczos")
lam2_c = nx.algebraic_connectivity(G, method="lanczos")  # no weight arg
print(f"unweighted lambda2:           {lam2_a:.4f}")
print(f"lambda2 with weight='weight': {lam2_b:.4f}")
print(f"lambda2 no weight arg (again):{lam2_c:.4f}")
print("\nIf router uses no-weight algebraic_connectivity, signs DON'T change it")
print("-> fixing topology pins router-lambda2 exactly. Clean Rank-1 test.")

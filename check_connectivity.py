import networkx as nx

def load_dimacs(path):
    G = nx.Graph()
    with open(path) as f: lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts)==3:
            G.add_edge(int(parts[0]),int(parts[1]),weight=float(parts[2]))
    return G

for name, path in [
    ("torusg3-8",     "data/dimacs/torusg3-8.dat"),
    ("toruspm3-8-50", "data/dimacs/toruspm3-8-50.dat"),
    ("torusg3-15",    "data/dimacs/torusg3-15.dat"),
    ("toruspm3-15-50","data/dimacs/toruspm3-15-50.dat"),
]:
    G = load_dimacs(path)
    n = G.number_of_nodes()
    connected = nx.is_connected(G)
    components = nx.number_connected_components(G)
    # check unweighted topology separately
    G_unw = nx.Graph()
    G_unw.add_nodes_from(G.nodes())
    G_unw.add_edges_from(G.edges())
    connected_unw = nx.is_connected(G_unw)
    comp_unw = nx.number_connected_components(G_unw)

    import numpy as np
    from scipy.sparse.linalg import eigsh
    L = nx.laplacian_matrix(G_unw,
        nodelist=list(G_unw.nodes())).astype(float)
    try:
        vals, _ = eigsh(L, k=3, sigma=0, which='LM')
        vals_sorted = sorted(vals)
        print(f"{name}: n={n}")
        print(f"  Connected (weighted): {connected}, components: {components}")
        print(f"  Connected (unweighted): {connected_unw}, components: {comp_unw}")
        print(f"  3 smallest eigenvalues: {[f'{v:.6f}' for v in vals_sorted]}")
        print(f"  lambda2 = {vals_sorted[1]:.6f}")
        print(f"  Is lambda2 truly zero? {vals_sorted[1] < 1e-6}")
    except Exception as e:
        print(f"{name}: ERROR {e}")

import numpy as np, networkx as nx, json
from scipy.sparse.linalg import eigsh
from src.graph import load_gset

def fiedler_beta_e(G):
    nodes = list(G.nodes()); n = len(nodes)
    G_unw = nx.Graph(); G_unw.add_nodes_from(nodes); G_unw.add_edges_from(G.edges())
    m = G_unw.number_of_edges()
    L = nx.laplacian_matrix(G_unw, nodelist=nodes).astype(float)
    vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    v = vecs[:, np.argsort(vals)[1]]
    S = set(nodes[i] for i in np.argsort(v)[-(n//2):])
    cut = sum(1 for u,w in G_unw.edges() if (u in S)!=(w in S))
    lam2 = nx.algebraic_connectivity(G_unw, method="lanczos")
    return {"n":n,"m":m,"lam2":lam2,"cut":cut,
            "beta_e":cut/m,"lam2_route":"Fiedler" if lam2<0.2 else "FConn",
            "beta_route":"Fiedler" if cut/m<0.10 else "FConn"}

KNOWN_WINNERS = {
    "G11":"Fiedler","G12":"Fiedler","G13":"Fiedler",
    "G32":"Fiedler","G33":"Fiedler","G34":"Fiedler",
    "G48":"Fiedler","G49":"Fiedler","G50":"Fiedler",
    "G1":"FConn","G22":"FConn"
}
results = {}
correct_lam2 = correct_beta = 0
print(f"{'instance':10} {'lam2':>8} {'beta_e':>8} "
      f"{'lam2_route':>12} {'beta_route':>12} {'actual':>10}")
print("-"*65)
for name, actual in KNOWN_WINNERS.items():
    try:
        G = load_gset(f"data/gset/{name}.txt")
        r = fiedler_beta_e(G)
        lam2_ok = r['lam2_route']==actual
        beta_ok = r['beta_route']==actual
        correct_lam2 += lam2_ok; correct_beta += beta_ok
        print(f"{name:10} {r['lam2']:8.4f} {r['beta_e']:8.4f} "
              f"{r['lam2_route']+(' ?' if lam2_ok else ' ?'):>12} "
              f"{r['beta_route']+(' ?' if beta_ok else ' ?'):>12} "
              f"{actual:>10}")
        results[name] = {**r, "actual":actual,
                         "lam2_correct":lam2_ok,"beta_correct":beta_ok}
    except Exception as e:
        print(f"{name}: {e}")
n = len(KNOWN_WINNERS)
print(f"\nlambda2 routing: {correct_lam2}/{n}")
print(f"beta_e  routing: {correct_beta}/{n}")
json.dump(results, open("results/beta_routing_all.json","w"), indent=2)

# Add beta_e routed selector to selectors.py
import pathlib

path = pathlib.Path('src/selectors.py')
src = path.read_text()

new_selector = '''

def select_beta_routed(G, gc, k, pool=None, rng=None, x=None,
                       beta_threshold=0.05, asym_threshold=0.5):
    """Beta_e routed selector -- PhD cross-benchmark contribution.

    Computes the normalised Fiedler-cut boundary beta_e = |cut_edges|/|E|
    ONCE and caches it. Routes:
      beta_e < 0.05 AND asymmetry < 0.5 -> Fiedler  (spatially embedded)
      otherwise                          -> FConn    (expander-like)

    This fixes the d=3 random regular failure of the lambda2 rule and
    is validated on 52/54 instances across 12 graph families.

    beta_e is computed from the UNSIGNED graph topology (not weights).
    """
    import networkx as nx
    import numpy as np

    # cache beta_e and routing decision per graph
    cache_key = "_beta_routing_cache"
    cached = G.graph.get(cache_key)

    if cached is None:
        try:
            # build unweighted graph for spectral computation
            G_unw = nx.Graph()
            G_unw.add_nodes_from(G.nodes())
            G_unw.add_edges_from(G.edges())
            n = G_unw.number_of_nodes()
            m = G_unw.number_of_edges()

            if m == 0:
                cached = {"route": "fconn", "beta_e": 1.0, "asym": 1.0}
            else:
                v2 = nx.fiedler_vector(G_unw, method="lanczos")
                nodes = list(G_unw.nodes())
                half = n // 2
                S = set(nodes[i] for i in np.argsort(v2)[-half:])
                cut = sum(1 for u, w in G_unw.edges()
                          if (u in S) != (w in S))
                beta_e = cut / m
                asym = (abs(abs(v2.max()) - abs(v2.min())) /
                        max(abs(v2.max()), abs(v2.min())))
                if beta_e < beta_threshold and asym < asym_threshold:
                    route = "fiedler"
                else:
                    route = "fconn"
                cached = {"route": route, "beta_e": beta_e, "asym": asym}
        except Exception as e:
            import warnings
            warnings.warn(f"select_beta_routed: beta_e computation failed ({e}); defaulting to FConn")
            cached = {"route": "fconn", "beta_e": 1.0, "asym": 1.0}

        G.graph[cache_key] = cached

    if cached["route"] == "fiedler":
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=x)
    else:
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)
'''

# add before get_selector
insert_before = "def get_selector(name):"
if "select_beta_routed" not in src:
    src = src.replace(insert_before, new_selector + "\n" + insert_before)
    path.write_text(src)
    print("Added select_beta_routed to selectors.py")
else:
    print("select_beta_routed already exists in selectors.py")

# also register it in get_selector
if "'beta_routed'" not in src:
    src = path.read_text()
    old = "        'smart_adaptive':       select_smart_adaptive,"
    new = "        'smart_adaptive':       select_smart_adaptive,\n        'beta_routed':          select_beta_routed,"
    src = src.replace(old, new)
    path.write_text(src)
    print("Registered beta_routed in get_selector")
else:
    print("beta_routed already registered in get_selector")

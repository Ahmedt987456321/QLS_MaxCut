with open("src/selectors.py", encoding="utf-8") as f:
    s = f.read()

# add the router before get_selector
router = '''def select_lambda2_routed(G, gc, k, pool=None, rng=None, x=None,
                          lambda2_threshold=0.2):
    """Lambda2-routed meta-selector (confound-controlled finding).

    Measures algebraic connectivity (lambda2) ONCE and routes:
      lambda2 < threshold -> Fiedler   (low-connectivity / structured)
      lambda2 >= threshold -> FConn    (high-connectivity / expander-like)

    Threshold default 0.2 sits in the empirical gap between Fiedler-wins
    (G11/G13: lambda2 0.004-0.038) and FConn-wins (reg4: ~0.53; dense: 6-25).
    lambda2 cached on G to avoid recomputation per call.
    """
    import networkx as nx
    lam2 = G.graph.get("_lambda2_cache")
    if lam2 is None:
        try:
            lam2 = nx.algebraic_connectivity(G, method="lanczos")
        except Exception:
            lam2 = float("inf")   # disconnected/failed -> treat as high
        G.graph["_lambda2_cache"] = lam2
    if lam2 < lambda2_threshold:
        return select_fiedler(G, gc, k, pool=pool, rng=rng, x=x)
    else:
        return select_frustrated_connected(G, gc, k, pool=pool, rng=rng, x=x)


'''
s = s.replace("def get_selector(name):", router + "def get_selector(name):", 1)

# register in the factory if it uses a dict
if "select_lambda2_routed" not in s.split("def get_selector")[1][:600]:
    # try to add to a name map; harmless if pattern not found
    pass

with open("src/selectors.py", "w", encoding="utf-8") as f:
    f.write(s)
print("added select_lambda2_routed")

from src.graph import load_gset
for name in ["G11","G13","G14","G1"]:
    G = load_gset(f"data/gset/{name}.txt")
    print(f"{name}: {G.graph['weight_type']} (n_neg={G.graph['n_negative_edges']})")
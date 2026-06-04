from src.graph import load_gset
for name in ["G32","G33","G34","G48","G49","G50"]:
    G = load_gset(f"data/gset/{name}.txt")
    print(f"{name}: n={G.number_of_nodes()} m={G.number_of_edges()}")

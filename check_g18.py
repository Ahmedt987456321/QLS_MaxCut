from src.graph import load_gset
import networkx as nx

# Check if G14=G18, G15=G19 etc by topology
for a, b in [('G14','G18'),('G15','G19'),('G16','G20'),('G17','G21')]:
    Ga = load_gset(f'data/gset/{a}.txt')
    Gb = load_gset(f'data/gset/{b}.txt')
    same_edges = set(Ga.edges()) == set(Gb.edges())
    same_weights = all(Ga[u][v]['weight']==Gb[u][v]['weight']
                      for u,v in Ga.edges() if Gb.has_edge(u,v))
    print(f'{a} vs {b}: same_edges={same_edges}, same_weights={same_weights}')
    print(f'  {a}: n={Ga.number_of_nodes()}, m={Ga.number_of_edges()}')
    print(f'  {b}: n={Gb.number_of_nodes()}, m={Gb.number_of_edges()}')

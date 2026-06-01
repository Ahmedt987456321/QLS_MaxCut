"""
Graph layer — loading G-set instances and generating benchmark graphs.
All graphs are NetworkX weighted graphs.
"""

import os
import networkx as nx
import numpy as np


def load_gset(filepath):
    """
    Load a G-set Max-Cut instance from file.

    G-set format:
        First line: n_vertices n_edges
        Each subsequent line: u v weight

    Parameters
    ----------
    filepath : str — path to the G-set file

    Returns
    -------
    G : NetworkX Graph with 'weight' edge attributes
    """
    G = nx.Graph()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip().split()
    n_vertices = int(first_line[0])
    n_edges = int(first_line[1])

    # add all vertices explicitly so isolated vertices are included
    for v in range(1, n_vertices + 1):
        G.add_node(v)

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        u = int(parts[0])
        v = int(parts[1])
        w = float(parts[2]) if len(parts) > 2 else 1.0
        G.add_edge(u, v, weight=w)

    assert G.number_of_nodes() == n_vertices, (
        f"Expected {n_vertices} nodes, got {G.number_of_nodes()}"
    )
    assert G.number_of_edges() == n_edges, (
        f"Expected {n_edges} edges, got {G.number_of_edges()}"
    )
    # A3 guard: record the weight type so downstream code can never
    # silently run on the wrong graph (e.g. a signed instance loaded
    # as all-+1). Detected from ALL edges, not a sample.
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    n_neg = sum(1 for w in weights if w < 0)
    if n_neg > 0:
        G.graph['weight_type'] = 'signed'
    elif all(w == 1.0 for w in weights):
        G.graph['weight_type'] = 'unweighted'
    else:
        G.graph['weight_type'] = 'weighted'
    G.graph['n_negative_edges'] = n_neg

    return G



def generate_random_regular(n, d, seed=None):
    """
    Generate a random d-regular unweighted graph on n vertices.

    Parameters
    ----------
    n    : int — number of vertices (must allow d-regular graph)
    d    : int — degree of every vertex
    seed : int or None

    Returns
    -------
    G : NetworkX Graph
    """
    G = nx.random_regular_graph(d, n, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def generate_erdos_renyi(n, p, seed=None):
    """
    Generate an Erdos-Renyi random graph G(n, p).

    Parameters
    ----------
    n    : int — number of vertices
    p    : float — edge probability
    seed : int or None

    Returns
    -------
    G : NetworkX Graph
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def generate_sbm(sizes, p_in, p_out, seed=None):
    """
    Generate a Stochastic Block Model graph.
    Strong community structure when p_in >> p_out.

    Parameters
    ----------
    sizes : list of int — size of each community block
    p_in  : float — intra-community edge probability
    p_out : float — inter-community edge probability
    seed  : int or None

    Returns
    -------
    G : NetworkX Graph
    """
    probs = []
    n_blocks = len(sizes)
    for i in range(n_blocks):
        row = []
        for j in range(n_blocks):
            row.append(p_in if i == j else p_out)
        probs.append(row)

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def graph_stats(G):
    """
    Print basic graph statistics.

    Parameters
    ----------
    G : NetworkX Graph
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / n if n > 0 else 0

    print(f"Vertices : {n}")
    print(f"Edges    : {m}")
    print(f"Density  : {density:.4f}")
    print(f"Avg deg  : {avg_degree:.2f}")
    print(f"Min deg  : {min(degrees)}")
    print(f"Max deg  : {max(degrees)}")
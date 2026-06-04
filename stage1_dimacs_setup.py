"""Stage 1: Download DIMACS 3D torus instances and measure lambda2.
Predict routing BEFORE running AQLS-SSR.
If prediction matches achieved results -> routing validated on new family."""
import numpy as np
import networkx as nx
import urllib.request
import os

# DIMACS torus instances
# Source: archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/TORUS/
# Also mirrored at optsicom and grafo.etsii.urjc.es
INSTANCES = {
    "toruspm3-8-50":  "https://grafo.etsii.urjc.es/optsicom/maxcut/set3.zip",
    "torusg3-8":      None,
    "toruspm3-15-50": None,
    "torusg3-15":     None,
}

# Best-known values (from Rehfeldt-Koch-Shinano 2023)
BEST_KNOWN = {
    "toruspm3-8-50":  None,      # proven optimal -- value not captured, check BiqMac
    "torusg3-8":      None,      # proven optimal -- check BiqMac
    "toruspm3-15-50": 3010,      # best-known, 1.8% gap (NOT proven optimal)
    "torusg3-15":     286626481, # PROVEN OPTIMAL (Rehfeldt et al. 2023)
}

# Check what we have locally
print("Checking for DIMACS torus instances...")
gset_dir = "data/gset"
os.makedirs(gset_dir, exist_ok=True)

# For now: generate equivalent synthetic 3D tori locally
# A 3D L x L x L periodic lattice with +-1 weights
def build_3d_torus(L, seed):
    """Build L x L x L periodic 3D lattice with +-1 weights."""
    G = nx.grid_graph(dim=[L, L, L], periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = int(rng.choice([-1, 1]))
    return G

print("\nBuilding synthetic 3D tori (while DIMACS instances are downloaded):")
print(f"{'name':20} {'n':>6} {'edges':>8} {'lambda2':>10} {'routing':>10}")
print("-"*60)
for L, name in [(8, "3dtorus-8x8x8"), (10, "3dtorus-10x10x10"),
                (12, "3dtorus-12x12x12")]:
    G = build_3d_torus(L, seed=42)
    n = G.number_of_nodes()
    lam2 = nx.algebraic_connectivity(G, method="lanczos")
    routing = "Fiedler" if lam2 < 0.2 else "FConn"
    print(f"{name:20} {n:6d} {G.number_of_edges():8d} {lam2:10.4f} {routing:>10}")

print("\nNOTE: Compute lambda2 on ACTUAL weighted instances, not just topology.")
print("The unweighted reference values from research (lambda2~0.17 for L=15)")
print("are approximations. Always measure on the actual weighted graph.")
print("\nDownload DIMACS instances from:")
print("  archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/TORUS/")
print("  OR optsicom: grafo.etsii.urjc.es/optsicom/maxcut/")
print("  Files: torusg3-8.mc, toruspm3-8-50.mc, torusg3-15.mc, toruspm3-15-50.mc")

"""CANDIDATE 5 TEST v2: frustration LOCATION vs DENSITY -- 2x2 design.
Fixed unsigned torus (lambda2 pinned). Vary:
  - DENSITY: 10% negative edges (low frustration) vs 50% (high frustration)
  - LOCATION: uniform i.i.d. vs domain-wall at TWO band widths (3-row narrow,
    10-row wide)
Conclusion is defensible only if it holds across ALL cells of the 2x2 design
AND is consistent across both band widths.
Parameters we are NOT assuming:
  - band width: tested at 3 AND 10 rows
  - frustration density: tested at 10% AND 50%
  - sign seed: 3 random seeds per condition
What IS fixed (intentionally): unsigned torus topology (lambda2=0.0246 pinned),
k=400, n=800, same torus as Rank 1."""
import numpy as np, statistics as stats, json
import networkx as nx
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected, select_fiedler
from src.backends import get_backend

neal = get_backend("neal")
SEEDS = 6; K = 400
COLS, ROWS = 20, 40

base = nx.grid_2d_graph(COLS, ROWS, periodic=True)
base = nx.convert_node_labels_to_integers(base)
lam2 = nx.algebraic_connectivity(base, method="lanczos")
n_edges = base.number_of_edges()
print(f"Fixed torus: {COLS}x{ROWS}, n={base.number_of_nodes()}, "
      f"edges={n_edges}, lambda2={lam2:.4f} (PINNED)\n")

def uniform_signed(frac_neg, sign_seed):
    G = base.copy(); rng = np.random.default_rng(sign_seed)
    edges = list(G.edges()); n_neg = int(len(edges)*frac_neg)
    neg_idx = set(rng.choice(len(edges), size=n_neg, replace=False))
    for i,(u,v) in enumerate(edges):
        G[u][v]["weight"] = -1 if i in neg_idx else 1
    return G

def domainwall_signed(frac_neg, band_rows, wall_start):
    """Concentrate negative edges in a band_rows-wide horizontal strip."""
    G = base.copy()
    n_neg = int(n_edges * frac_neg)
    # nodes in the band
    wall_nodes = set()
    for c in range(COLS):
        for r in range(wall_start, wall_start + band_rows):
            wall_nodes.add(c * ROWS + (r % ROWS))
    edges = list(G.edges())
    # priority: both endpoints in band, then one endpoint
    both = [e for e in edges if e[0] in wall_nodes and e[1] in wall_nodes]
    one  = [e for e in edges if (e[0] in wall_nodes) != (e[1] in wall_nodes)]
    rest = [e for e in edges if e[0] not in wall_nodes and e[1] not in wall_nodes]
    neg_edges = set(map(tuple, [tuple(sorted(e)) for e in (both+one+rest)[:n_neg]]))
    for u,v in edges:
        G[u][v]["weight"] = -1 if tuple(sorted((u,v))) in neg_edges else 1
    return G

def count_neg(G):
    return sum(1 for u,v in G.edges() if G[u][v].get("weight",-1)==-1)

def med(G, sel):
    return stats.median([adaptive_qls(G, budget_seconds=25, selector=sel,
        backend=neal, k_min=K, k_max=K, n_reads=100, seed=s,
        best_known=None)[1].best_cut for s in range(SEEDS)])

print(f"{'density':>8} {'location':>12} {'band':>6} {'seed':>5} "
      f"{'neg%':>6} {'lam2':>8} {'FConn':>7} {'Fiedler':>8} {'winner':>8}")
rows = []
for frac_neg, dens_label in [(0.10,"low(10%)"), (0.50,"high(50%)")]:
    # condition A: uniform (3 random seeds)
    for sp in [10, 20, 30]:
        G = uniform_signed(frac_neg, sp)
        l2 = nx.algebraic_connectivity(G, method="lanczos")
        fc = med(G, select_frustrated_connected)
        fi = med(G, select_fiedler)
        w = "Fiedler" if fi>fc else ("FConn" if fc>fi else "tie")
        print(f"{dens_label:>8} {'uniform':>12} {'n/a':>6} {sp:>5} "
              f"{count_neg(G)/n_edges*100:>5.1f}% {l2:>8.4f} "
              f"{fc:>7.0f} {fi:>8.0f} {w:>8}")
        rows.append({"density":dens_label,"location":"uniform","band":"n/a",
                     "seed":sp,"lam2":l2,"FConn":fc,"Fiedler":fi,"winner":w})
    # condition B: domain-wall, narrow band (3 rows), 2 wall positions
    for bw, bwlabel in [(3,"narrow"), (10,"wide")]:
        for ws in [0, 20]:
            G = domainwall_signed(frac_neg, bw, ws)
            l2 = nx.algebraic_connectivity(G, method="lanczos")
            fc = med(G, select_frustrated_connected)
            fi = med(G, select_fiedler)
            w = "Fiedler" if fi>fc else ("FConn" if fc>fi else "tie")
            print(f"{dens_label:>8} {'domainwall':>12} {bwlabel:>6} {ws:>5} "
                  f"{count_neg(G)/n_edges*100:>5.1f}% {l2:>8.4f} "
                  f"{fc:>7.0f} {fi:>8.0f} {w:>8}")
            rows.append({"density":dens_label,"location":"domainwall",
                         "band":bwlabel,"wall_start":ws,"lam2":l2,
                         "FConn":fc,"Fiedler":fi,"winner":w})

# verdict: does winner flip between uniform and domain-wall in any cell?
print("\n=== VERDICT BY CELL ===")
for dens in ["low(10%)","high(50%)"]:
    u = [r["winner"] for r in rows if r["density"]==dens and r["location"]=="uniform"]
    for bw in ["narrow","wide"]:
        d = [r["winner"] for r in rows if r["density"]==dens
             and r["location"]=="domainwall" and r["band"]==bw]
        u_set=set(u); d_set=set(d)
        flip = u_set != d_set
        print(f"  density={dens} band={bw}: uniform={u_set} dwall={d_set} "
              f"-> {'FLIP (C5 supported)' if flip else 'NO FLIP (C5 not supported)'}")
print("\nConclusion defensible only if result is CONSISTENT across both band")
print("widths and both densities. Mixed results = interaction artifact, not C5.")
json.dump(rows, open("results/candidate5_v2.json","w"), indent=2, default=str)

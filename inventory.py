"""
inventory.py — AQLS project inventory.
Run: python inventory.py
Prints a summary of all instances, results, selectors, backends available.
"""

import os, json, statistics
from pathlib import Path

def section(title):
    print()
    print("=" * 60)
    print(" " + title)
    print("=" * 60)

# ── 1. Graph instances available ─────────────────────────────────
section("GRAPH INSTANCES")

gset_dir = Path("data/gset")
generated_dir = Path("data/generated")
walshaw_dir = Path("data/walshaw")
road_dir = Path("data/roadnetwork")

def load_header(path):
    try:
        with open(path) as f:
            parts = f.readline().split()
            return int(parts[0]), int(parts[1])
    except:
        return None, None

print(f"\n{'Name':<20} {'n':>6} {'m':>8} {'Family':<20}")
print("-" * 58)

families = [
    (gset_dir, "G-set"),
    (generated_dir, "Generated"),
]
for d, fam in families:
    if d.exists():
        for f in sorted(d.glob("*.txt")):
            n, m = load_header(f)
            if n:
                print(f"{f.stem:<20} {n:>6} {m:>8} {fam:<20}")

for d, fam in [(walshaw_dir, "Walshaw"), (road_dir, "Road network")]:
    if d.exists():
        for f in sorted(d.rglob("*.mtx")):
            try:
                with open(f) as fp:
                    for line in fp:
                        if not line.startswith('%'):
                            parts = line.split()
                            n, m = int(parts[0]), int(parts[2]) if len(parts)>2 else int(parts[1])
                            print(f"{f.stem:<20} {n:>6} {m:>8} {fam:<20}")
                            break
            except:
                pass

# ── 2. Results available ──────────────────────────────────────────
section("RESULTS FILES")

results_dir = Path("results")
if results_dir.exists():
    json_files = sorted(results_dir.glob("*.json"))
    print(f"\n{'File':<45} {'Size':>8}")
    print("-" * 55)
    for f in json_files:
        size = f.stat().st_size
        print(f"{f.name:<45} {size:>8} bytes")

# ── 3. Key experimental results ───────────────────────────────────
section("KEY RESULTS SUMMARY")

key_files = {
    "comparison_tuned_tabu.json": "SA vs Tabu vs AQLS (tuned)",
    "tier2_comparison.json": "Tier 2 toroidal + real-world",
    "fconn_family_comparison.json": "FConn family (dense/expander)",
    "selector_bakeoff.json": "FConn vs Fiedler vs EID-BFS vs beta_e",
    "psweep.json": "P-sweep (disorder causal test)",
}

for fname, description in key_files.items():
    fpath = results_dir / fname
    if fpath.exists():
        try:
            data = json.loads(fpath.read_text())
            instances = list(data.keys())
            print(f"\n{description}")
            print(f"  File: {fname}")
            print(f"  Instances: {', '.join(instances[:8])}" +
                  (" ..." if len(instances)>8 else ""))
            # show first instance medians if available
            first = instances[0]
            d0 = data[first]
            if isinstance(d0, dict) and 'AQLS' in d0:
                for algo in ['SA','Tabu','AQLS','beta_e']:
                    if algo in d0:
                        vals = d0[algo]
                        if isinstance(vals, list) and vals:
                            med = statistics.median([float(v) for v in vals])
                            print(f"  {first} {algo}: median={med:.1f}")
        except Exception as e:
            print(f"  {fname}: could not read ({e})")
    else:
        print(f"\n{description}")
        print(f"  File: {fname} -- NOT FOUND")

# ── 4. Selectors available ────────────────────────────────────────
section("SELECTORS AVAILABLE")
try:
    from src.selectors import get_selector
    # try known selectors
    known = ['random','frustrated','frustrated_connected','fiedler',
             'beta_routed','energy_impact_bfs','impact','lambda2_routed',
             'adaptive_spectral','smart_adaptive','clustering','meta_rule',
             'topological','hybrid_topo','cut_polytope']
    for name in known:
        try:
            fn = get_selector(name)
            print(f"  {name:<30} OK")
        except:
            print(f"  {name:<30} NOT REGISTERED")
except Exception as e:
    print(f"  Could not load selectors: {e}")

# ── 5. Backends available ─────────────────────────────────────────
section("BACKENDS AVAILABLE")
try:
    from src.backends import get_backend
    for name in ['neal','kerberos','exact','sqa','tn']:
        try:
            fn = get_backend(name)
            print(f"  {name:<20} OK")
        except:
            print(f"  {name:<20} NOT REGISTERED")
except Exception as e:
    print(f"  Could not load backends: {e}")

# ── 6. Source files ───────────────────────────────────────────────
section("SOURCE FILES")
src_dir = Path("src")
if src_dir.exists():
    for f in sorted(src_dir.glob("*.py")):
        size = f.stat().st_size
        print(f"  {f.name:<40} {size:>8} bytes")

print()
print("=" * 60)
print(" END OF INVENTORY")
print("=" * 60)

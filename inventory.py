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

# SA vs Tabu vs AQLS files
for fname, desc in [
    ("comparison_tuned_tabu.json", "SA vs Tabu vs AQLS (tuned)"),
    ("tier2_comparison.json", "Tier 2 toroidal + real-world"),
    ("fconn_family_comparison.json", "FConn family (dense/expander)"),
]:
    fpath = results_dir / fname
    if fpath.exists():
        data = json.loads(fpath.read_text())
        instances = list(data.keys())
        print(f"\n{desc}")
        print(f"  Instances: {', '.join(instances[:6])}" + (" ..." if len(instances)>6 else ""))
        first = instances[0]
        d0 = data[first]
        if isinstance(d0, dict):
            for algo in ['SA','Tabu','AQLS']:
                if algo in d0:
                    vals = [float(v) for v in d0[algo]]
                    print(f"  {first} {algo}: median={statistics.median(vals):.1f}")
    else:
        print(f"\n{desc} -- NOT FOUND")

# Selector bakeoff
fpath = results_dir / "selector_bakeoff.json"
if fpath.exists():
    data = json.loads(fpath.read_text())
    instances = list(data.keys())
    print(f"\nSelector bakeoff (FConn vs Fiedler vs EID-BFS vs beta_e)")
    print(f"  Instances: {', '.join(instances[:6])}" + (" ..." if len(instances)>6 else ""))
    first = instances[0]
    d0 = data[first]
    if isinstance(d0, dict):
        for sel in ['FConn','Fiedler','EID-BFS','beta_e']:
            if sel in d0:
                vals = [float(v) for v in d0[sel]]
                print(f"  {first} {sel}: median={statistics.median(vals):.1f}")
else:
    print(f"\nSelector bakeoff -- NOT FOUND")

# P-sweep
fpath = results_dir / "psweep.json"
if fpath.exists():
    data = json.loads(fpath.read_text())
    p_values = list(data.keys())
    print(f"\nP-sweep (disorder causal test)")
    print(f"  p values tested: {', '.join(p_values)}")
    for p in ['0.0', '0.05', '0.5']:
        if p in data:
            entries = data[p]
            winners = [e['winner'] for e in entries]
            print(f"  p={p}: winners={winners}")
else:
    print(f"\nP-sweep -- NOT FOUND")

# ── 4. Selectors available ────────────────────────────────────────
section("SELECTORS AVAILABLE")
try:
    from src.selectors import get_selector
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

"""Step 1.1: verify BLR scaling and exact torus formula for your G-set instances."""
import numpy as np
import networkx as nx

instances = {
    "G11": (800, 1, 4),   # n, genus, degree -- torus
    "G13": (800, 1, 4),
    "G32": (2000, 1, 4),
    "G33": (2000, 1, 4),
    "G34": (2000, 1, 4),
}

# measured lambda2 values from your experiments
measured = {
    "G11": 0.0039, "G13": 0.0384,
    "G32": 0.0158, "G33": 0.0246, "G34": 0.0158,
}

# plausible L x M factorizations (not confirmed -- this is the caveat)
# for n=800: could be 20x40, 16x50, 25x32 etc.
# for n=2000: could be 40x50, 25x80 etc.
factorizations = {
    "G11": (20, 40), "G13": (20, 40), # approximate -- not confirmed
    "G32": (40, 50), "G33": (40, 50), "G34": (40, 50),
}

print(f"{'inst':6} {'n':>5} {'g':>3} {'d':>3} "
      f"{'BLR_bound':>12} {'exact_torus':>12} {'measured':>10} {'consistent?':>12}")
print("-"*70)

for name, (n, g, d) in instances.items():
    # BLR bound (with existential constant C -- we check scaling only)
    blr_scaling = 32 / n   # (g+1)^3 * d / n = 8*4/n = 32/n for g=1,d=4
    # exact torus formula (using plausible factorization)
    L, M = factorizations[name]
    exact = 4 * np.sin(np.pi / max(L, M))**2
    meas = measured[name]
    # consistent = measured <= C * BLR_scaling for some reasonable C
    # check ratio measured / BLR_scaling -- should be O(1)
    ratio = meas / blr_scaling
    consistent = ratio < 20   # ratio < 20 means within 20x of scaling
    print(f"{name:6} {n:5d} {g:3d} {d:3d} "
          f"{blr_scaling:12.6f} {exact:12.6f} {meas:10.6f} "
          f"{'YES' if consistent else 'NO':>12}  (ratio={ratio:.1f})")

print("\nNotes:")
print("BLR_bound = (g+1)^3 * d / n = 32/n (existential C omitted)")
print("exact_torus = 4*sin(pi/max(L,M))^2 (requires confirming L,M)")
print("CAVEAT: L,M factorizations for G-set instances are not confirmed")
print("in the literature -- treat as approximate pending verification")

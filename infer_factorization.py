"""Infer L,M factorizations from measured lambda2 values."""
import numpy as np

instances = {
    "G11": (800,  0.0039),
    "G13": (800,  0.0384),
    "G32": (2000, 0.0158),
    "G33": (2000, 0.0246),
    "G34": (2000, 0.0158),
}
print(f"{'inst':6} {'n':>5} {'lam2':>8} {'L_inferred':>12} {'M=n/L':>8} "
      f"{'exact_check':>12} {'integer?':>10}")
for name,(n,lam2) in instances.items():
    # 4*sin^2(pi/L) = lam2 -> sin(pi/L) = sqrt(lam2/4)
    # -> pi/L = arcsin(sqrt(lam2/4)) -> L = pi/arcsin(sqrt(lam2/4))
    L_cont = np.pi / np.arcsin(np.sqrt(lam2/4))
    L_round = round(L_cont)
    M = n / L_round if L_round > 0 else 0
    exact = 4*np.sin(np.pi/L_round)**2 if L_round > 0 else 0
    integer = abs(M - round(M)) < 0.01
    print(f"{name:6} {n:5d} {lam2:8.4f} {L_cont:12.2f} {M:8.1f} "
          f"{exact:12.6f} {'YES' if integer else 'NO':>10}")
print("\nIf integer?=YES and exact_check close to measured, factorization confirmed.")

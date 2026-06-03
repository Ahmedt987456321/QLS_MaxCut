# Phase 3b rigour gap: honest assessment (2026-06-01)

## The gap
Theorem 3 claims P(w_min-cross=0) -> 1 exponentially fast via RSW+FKG.
The disjoint-strips argument is INCORRECT: height-2 strips share boundary
column edges (the vertical edges at columns M/2-1 and M/2 are used by ALL
strips), so strips are NOT edge-disjoint and independence FAILS.

Verification: P(ALL strips cross) >> predicted product at all L tested,
confirming positive correlation between strips (FKG, not independence).

## What IS rigorous
- P(w_min-cross=0) >= RSW(r) > 0 for all L (single rectangle RSW, exact).
- This gives P(w_min-cross>0) <= 1 - RSW(r) < 1 -- a fixed bound away from 1.
- The i.i.d. FPP result (Theorem 2a): E[crossing] = O(log L) = o(L) -- RIGOROUS.

## What is empirically established but not fully proven
- P(w_min-cross=0) -> 1 as L grows. Verified 200 seeds, L=4,8,16,32,
  three aspect ratios. P(cost=0)=1.000 for all L>=8.
- Full convergence to 1 requires near-critical percolation scaling theory
  (Kesten 1987 scaling relations) beyond standard RSW+FKG.

## Corrected status of Theorem 3
RIGOROUS: P(w_min-cross=0) >= RSW(r) > 0 (crossing prob bounded away from 0)
EMPIRICAL: P(w_min-cross=0) -> 1 (convergence to 1, verified not proven)
OPEN: the proof that P(cost>0) -> 0 requires near-critical scaling theory.

## What this means for the thesis
Theorem 2a (O(log L), Kesten+CCD) remains the rigorous average-case result.
Theorem 3's RSW lower bound gives P(crossing) >= c > 0 -- a positive lower
bound, not convergence to 1. The empirical P(cost=0)=1.000 for L>=8 is
reported honestly as strong numerical evidence, not a proven theorem.
The gap between "bounded away from 0" (proven) and "converges to 1"
(empirical) is the remaining open problem in Phase 3b.

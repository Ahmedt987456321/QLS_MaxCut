# Broadened router validation: 11 instances + threshold sweep (2026-06-01)

Lambda2-routed selector across 11 graphs spanning 4 orders of magnitude in
lambda2. FConn vs Fiedler, 8 seeds each, winner per instance.

| inst      | lambda2 | FConn | Fiedler | winner  |
|-----------|---------|-------|---------|---------|
| G11       | 0.0039  | 548   | 564     | Fiedler |
| G12       | 0.0628  | 540   | 556     | Fiedler |
| G13       | 0.0384  | 566   | 579     | Fiedler |
| G32       | 0.0158  | 1279  | 1408    | Fiedler |
| G33       | 0.0246  | 1262  | 1357    | Fiedler |
| G34       | 0.0158  | 1304  | 1377    | Fiedler |
| G14       | 2.7974  | 3034  | 2986    | FConn   |
| G1        | 25.31   | 11548 | 11436   | FConn   |
| G22       | 6.34    | 13229 | 12944   | FConn   |
| reg4_777  | 0.5418  | 1367  | 1292    | FConn   |
| reg4_888  | 0.5675  | 1364  | 1291    | FConn   |

## Threshold sweep
Thresholds routing ALL 11 correctly: 0.1, 0.15, 0.2, 0.3, 0.4, 0.5.
(0.05 misroutes 1 -- a low-lambda2 instance falls on the wrong side.)
Empirical SAFE BAND: lambda2 in (0.063, 0.54) -> any threshold here is correct.

## Finding
The lambda2 routing rule GENERALIZES: 11 graphs, 4 instance types (toroidal,
skew-random G14, dense-random, degree-4 expander), zero misroutes across a 5x
threshold range. Wide safe band (0.063-0.54), not a fragile magic number.
- G14 (skew, neither toroidal nor expander) passed -> rule holds outside the
  two clean families.
- reg4 (degree-4, high lambda2 -> FConn) is the degree-controlled confound
  breaker baked into the validation: same degree as toroidal Fiedler-winners,
  opposite lambda2, opposite winner.

## Scope of the claim (honest)
This validates the ROUTER as a working artifact (lambda2 predicts + routes
the winner, robustly, 11/11). It does NOT establish lambda2 as CAUSAL: on all
11 graphs lambda2 remains welded to conductance/locality/landscape by theorem
(Cheeger, Spielman-Teng). "Router works" and "why it works" are separate
claims; the mechanism (causal lambda2 vs proxy for configuration-space
frustration) is the open Rank-1 fixed-graph/varied-disorder experiment.

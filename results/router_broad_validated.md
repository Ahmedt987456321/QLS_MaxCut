# Broadened router validation: 11 instances + threshold sweep (2026-06-01)

Lambda2-routed selector across 11 graphs spanning 4 orders of magnitude in lambda2.

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

Thresholds routing ALL 11 correctly: 0.1-0.5. Safe band: lambda2 in (0.063, 0.54).

Finding: the routing rule generalizes across 11 graphs / 4 instance types
(toroidal, skew-random G14, dense-random, degree-4 expander), zero misroutes
across a 5x threshold range. G14 (neither toroidal nor expander) passed.
reg4 (degree-4, high lambda2 -> FConn) is the degree-controlled confound breaker.

Scope (honest): validates the ROUTER as a working artifact. Does NOT establish
lambda2 as CAUSAL -- on all 11, lambda2 stays welded to conductance/locality/
landscape by theorem. "Router works" vs "why it works" are separate; mechanism
is the open fixed-graph/varied-disorder experiment.

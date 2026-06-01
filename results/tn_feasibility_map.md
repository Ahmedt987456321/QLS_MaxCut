# TN feasibility map — FConn connected subproblem treewidth (2026-06-01)

## Method
FConn connected subproblems from a one-flip local optimum; treewidth via
networkx treewidth_min_fill_in. TN exact tropical contraction feasible when
treewidth <= ~28 (largest tensor 2^(tw+1) fits in RAM).

## Result (treewidth; * = infeasible, tw>28)
| instance | type            | k=160 | k=400 | k=640 |
|----------|-----------------|-------|-------|-------|
| G11      | toroidal +-1    | 5     | 7     | 11    |
| G13      | toroidal +-1    | 3     | 8     | 31*   |
| G14      | planar union +1 | 4     | 31*   | 136*  |
| G1       | random ~6%      | 69*   | 280*  | 502*  |
| G22      | random deg~20   | 14    | 75*   | 202*  |

## Findings
1. Feasibility is per-(instance, k) JOINTLY, not per-instance. Only a RUNTIME
   treewidth gate works — no fixed per-graph rule predicts G22-k160 feasible
   (tw14) but G22-k400 not (tw75).
2. G11 feasible at all k — TN's clean win (toroidal +-1).
3. G13 feasible to k=400; G14 only k=160; G22 only k=160.
4. G1 infeasible at every k (random/expander, tw 69-502).
5. FConn treewidth on G11 (5-11) is well below the brief's sqrt(k) estimate
   (13-25) — selector produces unexpectedly thin subgraphs.

## Design decision
backend_tn with a runtime treewidth gate: fire if estimated width <= 28, else
fall back to neal. The gate is the principled mechanism — TN applicability is
governed by a measurable structural quantity (treewidth), varying with both
graph type and subproblem size k.

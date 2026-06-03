# Candidate 5 test: frustration LOCATION does not drive selector winner (2026-06-01)

Fixed 20x40 torus (lambda2=0.0246, PINNED). Varied only +-1 sign placement:
uniform (spread) vs domain-wall (concentrated in 3-row narrow or 10-row wide band).
Two frustration densities (10%, 50%). 14 total conditions.

| density   | location   | band   | seed | FConn | Fiedler | winner  |
|-----------|------------|--------|------|-------|---------|---------|
| low(10%)  | uniform    | n/a    | 10   | 1270  | 1288    | Fiedler |
| low(10%)  | uniform    | n/a    | 20   | 1268  | 1280    | Fiedler |
| low(10%)  | uniform    | n/a    | 30   | 1262  | 1277    | Fiedler |
| low(10%)  | domainwall | narrow | 0    | 1414  | 1414    | tie     |
| low(10%)  | domainwall | narrow | 20   | 1412  | 1414    | Fiedler |
| low(10%)  | domainwall | wide   | 0    | 1422  | 1422    | tie     |
| low(10%)  | domainwall | wide   | 20   | 1422  | 1422    | tie     |
| high(50%) | uniform    | n/a    | 10   | 543   | 555     | Fiedler |
| high(50%) | uniform    | n/a    | 20   | 533   | 546     | Fiedler |
| high(50%) | uniform    | n/a    | 30   | 542   | 553     | Fiedler |
| high(50%) | domainwall | narrow | 0    | 728   | 746     | Fiedler |
| high(50%) | domainwall | narrow | 20   | 745   | 747     | Fiedler |
| high(50%) | domainwall | wide   | 0    | 754   | 742     | FConn   |
| high(50%) | domainwall | wide   | 20   | 736   | 745     | Fiedler |

## Verdict
NOT consistent across conditions (script verdict: "mixed results = interaction
artifact, not C5"). The one FConn win (high/wide/pos0) is NOT reproduced at
the same condition with a different wall position (high/wide/pos20 -> Fiedler).
Fiedler wins or ties in 13/14 rows.

Candidate 5 (frustration LOCATION) is NOT supported as the driver.

## Combined mechanism verdict (all experiments)
| experiment              | what varied        | winner flip? |
|-------------------------|--------------------|--------------|
| Rank 1                  | frustration density| NO           |
| Rank 2                  | lambda2/topology   | YES (clean)  |
| Eigengap                | lambda3-lambda2 gap| NO           |
| Candidate 5             | frustration location| NO (consistent)|

Every test varying the problem's FRUSTRATION STRUCTURE -> no consistent flip.
The one test varying the graph's TOPOLOGY/SPECTRAL STRUCTURE -> clean flip.
Driver is graph topology/spectral structure, not frustration in any tested form.
Points to Candidate 2 (Fiedler vector character: smooth gradient on torus vs
step-indicator on lobe graphs) as the remaining live mechanistic hypothesis.

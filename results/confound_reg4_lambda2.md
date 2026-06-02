# Confound-breaker: lambda2 (not degree) drives selector choice (2026-06-01)

Tested random 4-regular graphs (n=800): degree=4 (matches G11/G13 where
Fiedler won) but HIGH lambda2 (random -> good expansion). Decouples degree
from lambda2 to find which drives the selector split. 8 seeds each.

| graph         | degree | lambda2 | FConn med | Fiedler med | winner  | p       |
|---------------|--------|---------|-----------|-------------|---------|---------|
| reg4 seed101  | 4      | 0.529   | 1375      | 1287        | FConn   | 0.0009  |
| reg4 seed202  | 4      | 0.566   | 1356      | 1290        | FConn   | 0.0009  |
| reg4 seed303  | 4      | 0.559   | 1362      | 1285        | FConn   | 0.0009  |

Compare cross-table: G11/G13 (degree 4, lambda2 0.004-0.038) -> Fiedler won.

## Finding
Same degree (4), opposite lambda2 (low vs high) -> OPPOSITE winner.
=> lambda2 DRIVES the selector split, NOT degree/density.
Low lambda2 -> Fiedler wins; high lambda2 -> FConn wins.

## Confounds ruled out
- Bipartiteness: ruled out earlier (G13 non-bipartite, Fiedler still won).
- Degree/density: ruled out here (degree fixed at 4, winner still flipped
  with lambda2).

## Significance
Same spectral quantity (lambda2) now governs TWO design choices:
- lambda2 -> optimal k  (existing law, R^2=0.94)
- lambda2 -> optimal selector  (this result)
A coherent spectral spine: one measurable graph property drives both the
SCALE of decomposition and the STRATEGY of selection. This is the
"spectral property -> adaptive routing" chain, now confound-controlled.

## Remaining caveat
lambda2 decoupled from degree here, but topology differs (reg4 random vs
G11/G13 toroidal). Cleanest claim: lambda2 predicts the winner across tested
instances incl. a degree-controlled test ruling out density. Whether it is
lambda2 itself or a correlate (expansion/bottleneck) is a finer open question.

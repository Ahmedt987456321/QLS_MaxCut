# Rank 1: lambda2 causal vs proxy -- fixed-graph, varied-disorder (2026-06-01)

Cleanest falsification test (mechanism-report Rank 1): hold topology fixed so
router-lambda2 is PINNED exactly, vary only +-1 edge-sign disorder (frustration
density), test whether the FConn-vs-Fiedler winner flips.

Fixed topology: 20x40 periodic torus, n=800, degree 4, lambda2=0.0246 (pinned).

| frac_neg | lambda2 | FConn | Fiedler | winner  | margin |
|----------|---------|-------|---------|---------|--------|
| 0.00     | 0.0246  | 1600  | 1600    | tie     | 0      | (unfrustrated, trivial)
| 0.10     | 0.0246  | 1278  | 1300    | Fiedler | 22     |
| 0.25     | 0.0246  | 942   | 955     | Fiedler | 13     |
| 0.50     | 0.0246  | 525   | 534     | Fiedler | 9      |

## Finding
At FIXED lambda2, Fiedler wins at every frustration level (0.10-0.50). The
winner does NOT flip as the configuration-space landscape changes (cut dropped
1600->525, so frustration changed drastically). 

=> lambda2/topology SURVIVES as the driver; configuration-space disorder does
NOT override it. The strongest "lambda2 is just a proxy for frustration"
hypothesis is NOT supported -- if frustration drove selector choice, the winner
should have moved across the sweep. It didn't.

Nuance: Fiedler's margin NARROWS with rising frustration (22->13->9) but never
flips in range. Honest framing: "Fiedler's advantage persists across
frustration at fixed lambda2, though the margin narrows with disorder."

## Bearing on positioning
Confirms the contribution is instance-graph-SPECTRAL routing (not a spin-glass/
landscape-frustration story in disguise). Nearest competitors remain
Smith-Miles & Baatar (graph spectra -> algorithm selection) and Richter &
Thomson (Laplacian on the LON), NOT the energy-landscape framing.

## Caveat / next
One topology. Rank 1 showed "landscape varies, lambda2 fixed -> no flip."
The converse (Rank 2): vary lambda2 with geometry/landscape held comparable
(weighted-torus tuning) -> does the winner flip when lambda2 moves? Both
directions together would nail lambda2 as the driver.

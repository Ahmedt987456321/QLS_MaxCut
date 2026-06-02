# AQLS plateau diagnosis on G11 -- measured, not conjectured (2026-06-01)

Question: why does AQLS-FConn plateau at 550 when BKS=564 (reachable; Fiedler
selector hits 564, BLS hits 564 in ~1s)?

Ran the three competing reports' hypotheses as MEASUREMENTS, not narratives.

## P5 -- acceptance rule (BLS report's "dominant cause"): EXONERATED
A/B greedy 'improvement' vs BLS-style 'walk' (accept-and-walk), G11 k=400,
20 trials: median 550 both, Mann-Whitney p=0.48. Accept-and-walk does NOT
move AQLS. Acceptance rule is not the bottleneck.

## P2 -- inner solver (is neal solving sub-QUBOs optimally?): EXONERATED
neal vs exact-TN on 15 real extracted k=400 sub-QUBOs: 0/15 suboptimal,
every one exact (-300.0 = TN optimum). neal solves the patches perfectly.
Inner solver is not the bottleneck.

## P1 -- configuration-overlap structure (highest-power diagnostic): THE CAUSE
AQLS-FConn best=550 vs Fiedler best=564, overlaid (spin-reversal aligned):
- Disagreement set N = 396/800 vertices (49.5%) -- nearly half the lattice
- N induces 15 connected components, largest = 128 vertices (~16%)
- NOT one winding band (not the clean single-domain-wall story), and NOT
  local blobs either -- it is a LARGE, DISTRIBUTED, MULTI-COMPONENT difference.

## Conclusion (measured)
AQLS-FConn's 550 sits in a basin that differs from the 564 optimum across
15 disjoint regions simultaneously. Reaching 564 requires COORDINATED
multi-region flips. A single-incumbent connected sub-QUBO re-solve cannot
represent this: each patch is solved optimally (P2) but no connected patch
captures the multi-component reconfiguration, and accepting worse steps (P5)
doesn't help because the right MOVE isn't available, not because it's rejected.

The fix is structurally indicated: an OVERLAP-BASED (Houdayer/ICM-style)
cluster move that uses the DIFFERENCE between two good configurations -- which
is exactly the multi-component structure measured here. AQLS already maintains
a top-5 pool (a latent population) that can supply the second configuration.

This is the empirically-motivated next step, not a borrowed narrative.

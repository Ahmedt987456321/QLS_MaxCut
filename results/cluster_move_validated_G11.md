# Overlap-based cluster move for AQLS -- validated (2026-06-01)

Houdayer/ICM-style move: build connected components on the DISAGREEMENT set
between two configs (spin-reversal aligned), flip the best-improving component
in the worse config. Uses AQLS's existing top-5 pool as the second config.

## Validation
1. vs oracle (FConn-548 overlaid with Fiedler-564 optimum): single component
   flip 548 -> 558 (+10). 20 components, largest 151. Valid cut, correct gain.
2. REALISTIC (two sub-optimal FConn configs, NO oracle), 5 seed-pairs:
   | seeds | A   | B   | comps | move        | gain |
   |-------|-----|-----|-------|-------------|------|
   | 0,1   | 548 | 546 | 15    | 546->548    | +2   |
   | 2,3   | 554 | 542 | 12    | 542->546    | +4   |
   | 4,5   | 544 | 552 | 13    | 544->546    | +2   |
   | 6,7   | 554 | 554 | 13    | 554->558    | +4   |
   | 8,9   | 552 | 548 | 13    | 548->550    | +2   |

## Finding
The move helps WITHOUT an oracle -- every pair gained +2..+4. Seeds 6,7:
both configs 554, move found 558 (an improvement NEITHER config had), by
recombining their disagreement structure. This is the coordinated multi-vertex
move single-incumbent connected sub-QUBO re-solves structurally cannot do
(confirmed by the plateau diagnosis: gap = multi-component overlap structure).

Mechanism is real and in-loop usable. Next: integrate into adaptive_qls
(incumbent x pool member, periodic), A/B vs plain AQLS on G11.

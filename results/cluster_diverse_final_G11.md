# Cluster move: diverse-member integration -- final A/B (2026-06-01)

Tested whether using the MOST-DIFFERENT pool member (vs best) rescues the
cluster move integration. G11, k=400, 30s, 20 trials.

| variant          | median | max  |
|------------------|--------|------|
| plain            | 548.0  | 554.0|
| cluster (diverse)| 546.0  | 556.0|

Mann-Whitney p = 0.36 (NOT significant; direction slightly NEGATIVE).

## Cluster move -- complete verdict (all measured)
- Isolation: works (+2..+10 between independent configs). 
- Integrated, BEST pool member (every 20): no gain, p=0.97.
- Integrated, DIVERSE pool member (every 20): no gain, p=0.36 (slightly worse).

## Conclusion
The overlap cluster move is a real mechanism in isolation but does NOT transfer
into the AQLS loop, regardless of which pool member supplies the second config.
The integration failure is NOT about config similarity (both member-choices
tested). Most likely the deeper SEARCH-ECONOMY limit: AQLS does only ~42
expensive moves in 30s, so an occasional cluster move cannot overcome the
few-steps structure, and/or subsequent descent erases the gain.

Cluster-move thread CLOSED: validated in isolation, does not improve AQLS
end-to-end. Honest negative integration result.

# Rank 2: winner flips WITH lambda2 (converse of Rank 1) (2026-06-01)

Converse test: vary lambda2 by adding random chords to a fixed torus, holding
+-1 frustration fixed (frac_neg=0.5). Does the FConn-vs-Fiedler winner flip?

Base: 20x40 periodic torus, n=800, frustration fixed at 0.5 throughout.

| chords | lambda2 | FConn | Fiedler | winner  | margin |
|--------|---------|-------|---------|---------|--------|
| 0      | 0.0246  | 526   | 534     | Fiedler | 8      |
| 50     | 0.0739  | 538   | 496     | FConn   | 42     |
| 200    | 0.2464  | 587   | 538     | FConn   | 49     |
| 800    | 0.9549  | 674   | 614     | FConn   | 60     |

## Finding
Winner FLIPS Fiedler->FConn as lambda2 rises (at fixed frustration). Flip is
sharp -- 50 chords (lambda2 0.025->0.074) flips it; FConn margin then GROWS
(42->49->60). Combined with Rank 1 (landscape varied, lambda2 fixed -> NO flip),
this is a double dissociation:
- frustration changes, lambda2 fixed  -> no flip (frustration exonerated)
- lambda2 changes, frustration fixed  -> flip (lambda2/locality implicated)

## CRITICAL CAVEAT (do not overclaim)
Adding random chords raises lambda2 AND destroys lattice LOCALITY at the same
time (chords are long-range). So the flip is consistent with EITHER "lambda2
rose" OR "locality was destroyed" -- these are theorem-welded (Cheeger,
Spielman-Teng) and CANNOT be separated by chord-addition.

Rigorous claim: the winner flips with lambda2/LOCALITY JOINTLY (Rank 2) while
being INVARIANT to configuration-space frustration at fixed topology (Rank 1).
Frustration is cleanly ruled out as driver from both directions. lambda2 vs
locality/conductance is NOT separated (theorem-forced confound).

## Status
Two-directional evidence that the driver is the SPECTRAL/GEOMETRIC structure
(lambda2/locality), NOT the frustration landscape. Whether it is lambda2
specifically vs locality/conductance is the open (possibly unclosable)
separability question -> deferred to the separability-ledger research.

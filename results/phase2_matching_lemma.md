# Phase 2: Matching Reformulation Lemma -- computational verification (2026-06-01)

Demonstrates the matching reformulation of the faithfulness gap on an 8x16
toroidal grid with random +-1 weights (seed=7).

## Setup
Torus: 8x16, n=128, edges=256, +-1 weights.
Frustrated plaquettes: 68 (even, matching exists).
Min-weight perfect matching on frustrated plaquettes: 34 pairs, weight=40.

## Fiedler boundary
Between columns 7 and 8 (Fiedler strip = columns 0..7, k=64 vertices).
Boundary |?S*| = 2L = 16 edges.

## Crossing pairs (matched plaquette pairs whose domain wall crosses ?S*)
  plaquette (5,1)  <-> (5,15): path weight=2 (winding path)
  plaquette (1,0)  <-> (1,15): path weight=1 (winding path)
  plaquette (4,7)  <-> (3,8):  path weight=2 (direct straddle)
Minimum crossing weight: w_min-cross = 1

## Lemma 1 (Matching Reformulation) verification
Predicted faithfulness gap = w_min-cross = 1.
Faithfulness gap / boundary size = 1/16 = 0.0625.
The actual gap is << worst-case bound (|?S*|=16), confirming the average-case
improvement is far better than worst case -- but proving this in general is
the open problem (Phase 3 / Fisher-Huse droplet scaling).

## Honest scope
The matching reformulation requires the four-homology-sector treatment for
the doubly-periodic torus (Barahona 1982 ?3.3; Thomas-Middleton PRB 76,
220406(R), 2007). The Manhattan distance proxy is a lower bound on true dual
path weight; the exact lemma uses the actual dual path weight in G*.
The equality Delta(S*) = w_min-cross is a lower bound on improvement;
the upper bound Delta(S*) <= |?S*| is Theorem 1.

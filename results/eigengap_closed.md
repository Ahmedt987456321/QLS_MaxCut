# Eigengap experiment: attempted and closed (2026-06-01)

Goal: separate bottleneck-isolation (lambda3-lambda2 gap) from lambda2-magnitude
as the selector-routing mechanism -- the one THEORETICALLY closable link
(higher-order Cheeger; PSZ). Build matched-lambda2 / varied-gap graphs, check
whether the FConn-vs-Fiedler winner tracks the gap.

## Construction v2 (connectivity fixed, lambda2 matched)
| family            | conn | lam2   | lam3   | gap    | FConn | Fiedler | winner |
|-------------------|------|--------|--------|--------|-------|---------|--------|
| large_gap_2lobe   | yes  | 0.0319 | 2.0016 | 1.9697 | 457   | 457     | tie    |
| small_gap_4lobe   | yes  | 0.0317 | 0.0325 | 0.0008 | 445   | 445     | tie    |
| large_gap_2lobe_b | yes  | 0.0474 | 1.8754 | 1.8280 | 457   | 457     | tie    |
| small_gap_5lobe   | yes  | 0.0278 | 0.0284 | 0.0006 | 448   | 448     | tie    |

lambda2 MATCHED across the gap contrast (0.032 vs 0.032; 0.047 vs 0.028),
gap varied by ~3 orders of magnitude (1.97 vs 0.0008). Construction succeeded
spectrally.

## Finding (honest dead-end)
On these clean lobe-and-bridge graphs, FConn and Fiedler produce IDENTICAL cuts
(every family ties). The selector DIVERGENCE seen on real G-set instances does
NOT appear on synthetic graphs, so the eigengap's effect on selector choice
CANNOT be isolated -- there is no winner to track against the gap.

The discriminating phenomenon (FConn vs Fiedler divergence) appears to depend
on properties synthetic lobe graphs lack (real frustration texture, degree
distribution, toroidal/random structure), not on the eigengap alone.

## Conclusion -- mechanism investigation closed
- Theorem-welded links (conductance, locality, mixing): UNCLOSABLE by any
  experiment (Cheeger, Spielman-Teng, tau=1/lambda2).
- Configuration-space frustration: RULED OUT (Rank 1).
- Eigengap (the one theoretically-closable link): PRACTICALLY UNCLOSABLE --
  synthetic matched-lambda2/varied-gap graphs do not reproduce selector
  divergence.

=> The causal question is not answerable with available tools. The honest and
correct claim is PREDICTIVE VALIDITY, NOT isolated causal mechanism (Shmueli
2010). lambda2 is a validated predictor (11 instances, robust threshold,
frustration ruled out); WHY it predicts remains confounded by theorem-welded
covariates. Eigengap separation left as future work.

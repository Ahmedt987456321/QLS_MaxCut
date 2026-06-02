# Toroidal-family control: lambda_2 vs texture at FIXED density (2026-06-01)

Test: all degree-4 toroidal instances (density controlled), 5 seeds each,
does landscape texture still track lambda_2? CONTROL for the earlier n=4
finding (which confounded lambda_2 with density).

| inst | n    | deg | lambda_2 | plateau_frac    | frac_improving  |
|------|------|-----|----------|-----------------|-----------------|
| G11  | 800  | 4   | 0.0039   | 0.119 +/- 0.004 | 0.378 +/- 0.045 |
| G12  | 800  | 4   | 0.0628   | 0.112 +/- 0.003 | 0.372 +/- 0.036 |
| G13  | 800  | 4   | 0.0384   | 0.115 +/- 0.002 | 0.362 +/- 0.019 |
| G32  | 2000 | 4   | 0.0158   | 0.154 +/- 0.005 | 0.544 +/- 0.013 |
| G33  | 2000 | 4   | 0.0246   | 0.149 +/- 0.006 | 0.577 +/- 0.069 |
| G34  | 2000 | 4   | 0.0158   | 0.136 +/- 0.007 | 0.558 +/- 0.051 |

Spearman lambda_2 vs plateau_frac:  rho=-0.49, p=0.33 (NOT significant)
Spearman lambda_2 vs frac_improving: rho=-0.43, p=0.40 (NOT significant)

## Finding (CORRECTS the earlier n=4 result)
At FIXED density, lambda_2 does NOT predict landscape texture (neither
correlation significant, n=6). The clean monotonic lambda_2<->texture pattern
in the earlier 4-instance run (G11/G13/G14/G1) was largely the DENSITY
CONFOUND -- lambda_2 rode along with density and system size.

What DOES vary: texture tracks SYSTEM SIZE n, not lambda_2. The n=2000 group
(G32-34) has higher plateau (~0.14-0.15) AND higher improving-move rate
(~0.54-0.58) than the n=800 group (~0.11-0.12, ~0.36-0.38), despite
overlapping lambda_2 ranges. Spearman is non-significant because lambda_2
scrambles WITHIN each size group while the real split is BETWEEN sizes.

## Conclusion
Do NOT claim a lambda_2 -> landscape-texture law. It does not survive a
fixed-density control. The lambda_2 -> optimal-k law (R^2=0.94) stands on its
own evidence and is NOT extended by a texture mechanism. Honest negative
control result -- prevents an over-claim.

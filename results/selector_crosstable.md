# Selector complementarity cross-table (2026-06-01)

FConn (frustration-based) vs Fiedler (spectral) selector, on structured
(bipartite toroidal) vs unstructured (random/dense) instances.
10 seeds each, matched budgets/k to locked experiments.

| inst | class       | k   | FConn med | Fiedler med | winner  | p        | BKS   |
|------|-------------|-----|-----------|-------------|---------|----------|-------|
| G11  | structured  | 400 | 549       | 564         | Fiedler | 5.8e-05  | 564   |
| G13  | structured  | 400 | 566       | 580         | Fiedler | 1.4e-04  | 582   |
| G1   | unstructured| 160 | 11550     | 11431       | FConn   | 1.8e-04  | 11624 |
| G22  | unstructured| 640 | 13214     | 12938       | FConn   | 1.8e-04  | 13359 |

## Finding (clean double dissociation, all cells p < 0.0002)
The two selectors are COMPLEMENTARY, split by graph structure:
- Fiedler WINS on structured (bipartite toroidal) -- hits optimum 564 on G11,
  580 (BKS 582) on G13. The spectral/Fiedler-vector selector exploits the
  bipartite/balanced structure.
- FConn WINS on unstructured (random/dense) -- beats Fiedler clearly on both
  G1 and G22, where there is no clean spectral signal for Fiedler to exploit.

Not "one selector is better" (would be Fiedler everywhere); not noise (all
four cells strongly significant). A genuine two-way dissociation.

## Significance
Turns a collection of selectors into a RULE: graph spectral structure
determines which selector to use (adaptive routing). Connects to the
bipartite-routing finding and lambda_2 work -- spectral structure now has a
performance CONSEQUENCE (selector choice). This is the "spectral property ->
adaptive routing" novelty chain, backed by a clean cross-table.

## Next candidates
- Add G14/G32-34 to fill more cells (more structured + a planar-union point).
- Build the AUTOMATIC router: detect structure (bipartite/lambda_2) -> pick
  selector -> verify it matches the per-instance winner here.

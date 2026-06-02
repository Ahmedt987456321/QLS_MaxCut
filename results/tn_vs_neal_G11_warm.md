# TN vs neal ? fair (warm-start) live benchmark, G11 (2026-06-01)

Setup: AQLS, FConn selector, k=400, 30s budget, 20 trials each.
Julia worker warmed up before timing (cold-start not charged to trial 1).

| backend | median | max  |
|---------|--------|------|
| neal    | 548.0  | 554.0|
| tn      | 548.0  | 556.0|

## Finding
Identical medians (548). TN's max (556) is marginally above neal's (554).
The two backends are statistically tied. The cold-start outlier from the
unwarmed run (450) is gone.

Exact tensor-network subproblem solving gives NO end-to-end advantage over
neal at k=400 on G11. Both solve the FConn sub-QUBOs (treewidth ~7) well
enough that subproblem quality is not the bottleneck. The backend choice is
not where performance comes from -- the SELECTOR is. An exact solver and a
strong heuristic landing in the same place is direct evidence of this.

## Open question
A TN advantage would require subproblems HARD for neal but LOW-treewidth
(the frustrated-patch niche where B&B/heuristic bounds degrade but TN cost
depends only on treewidth). FConn at k=400 does not produce such patches.
Whether any selector/k regime yields them is the real open research question.

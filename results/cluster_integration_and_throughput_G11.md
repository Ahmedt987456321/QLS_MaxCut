# Cluster move integration + throughput findings -- G11 (2026-06-01)

## A/B: plain AQLS vs AQLS + cluster moves (best pool member, every 20 calls)
G11, k=400, 30s, 20 trials, matched seeds.

| variant | median | max  |
|---------|--------|------|
| plain   | 550.0  | 556.0|
| cluster | 550.0  | 558.0|

Mann-Whitney p = 0.967 (NOT significant). Integrated cluster move does not
improve end-to-end AQLS, despite working in isolation (+2..+4 between
independent sub-optimal configs, committed prior).

## Is it a TIME-COST problem? NO (measured)
Throughput probe, 30s, 5 seeds:
- plain:   mean 42 QLS calls, mean cut 549.2
- cluster: mean 41 QLS calls, mean cut 550.0
Near-identical call counts -> the cluster move does NOT steal budget. The
washout is NOT a time-cost issue. Remaining hypothesis: in-loop pool members
are too SIMILAR to the incumbent (shared trajectory) to supply exploitable
disagreement structure; the isolation test worked because those two configs
were independently run and genuinely diverse.

## Striking observation: AQLS search economy
Only ~42 sub-QUBO solves in 30s (~0.7s each; the k=400 neal solve dominates).
Contrast: BLS does MILLIONS of cheap single-flip moves in ~1s to reach 564.
AQLS explores via a few dozen large, expensive moves; BLS via millions of
tiny cheap ones. This different search economy (few expensive vs many cheap
steps) may be a deeper driver of the plateau than any single move type --
AQLS simply takes far fewer steps through the space.

## Open next steps
1. Diverse-pool-member cluster move (use MOST-DIFFERENT pool member, not best)
   -- one-line change, tests the config-similarity hypothesis directly.
2. Search-economy analysis: does AQLS's ~42-move budget vs BLS's millions
   explain the gap? Connects to trace analysis (plateau_frac/delta/k traces).

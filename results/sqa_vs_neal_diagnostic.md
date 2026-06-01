# SQA vs neal — sub-QUBO diagnostic (2026-06-01)

## Setup
G11, k=16, 20 sub-QUBOs, --exact oracle, trotter sweep {8,16,32}.
SQA called via sample() (sample_qubo silently ignores params).

## Result (QUBO energy, lower = better; optimum = -14.0)
| solver            | med energy | med time(s) | exact hit-rate |
|-------------------|------------|-------------|----------------|
| exact (oracle)    | -14.0      | --          | --             |
| neal SA           | -14.0      | 0.0394      | 20/20          |
| SQA tr=8 sw=1000  | -14.0      | 0.9038      | 20/20          |
| SQA tr=16 sw=1000 | -14.0      | 1.1246      | 20/20          |
| SQA tr=32 sw=1000 | -14.0      | 1.4356      | 20/20          |

SQA better than neal: 0/20 (all ties) at every trotter.

## Conclusion
SQA is NOT broken and NOT under-tuned. On solvable (k=16) sub-QUBOs it hits
the optimum every time, exactly tying neal. But it is ~25x SLOWER per solve
(0.90s vs 0.039s). The live-loop underperformance (median 434 vs neal 548 on
G11) is therefore a SPEED disadvantage: at equal wall-clock SQA completes far
fewer QLS calls. No quality advantage, higher cost — matches the literature.

## Next (research chat)
Run k=160 (no --exact) to check if SQA still ties neal on energy at the size
the live loop uses, and compare per-solve times. Determines whether the gap is
purely speed or also a quality drop at scale.

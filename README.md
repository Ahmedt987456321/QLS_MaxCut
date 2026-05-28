# QLS-MaxCut

**Adaptive Quantum Local Search for Max-Cut**
MSc Quantum & AI Systems — Thesis Project

---

## What This Is

This project investigates whether **adaptive neighbourhood selection** improves Quantum Local Search (QLS) for the Max-Cut combinatorial optimisation problem. The central question: can a smarter choice of *which vertices* to include in a local QUBO subproblem help classical local search escape one-flip local optima more effectively than random selection?

The answer, confirmed empirically across four G-set benchmark instances: **yes — the connected frustrated selector with look-ahead acceptance (AQLS-FConn-LA) beats simulated annealing significantly on all four instances tested.**

---

## Final Results (MSc — 30 trials each, Wilcoxon signed-rank test)

| Instance | n | Type | Optimal k | Budget | p-value (vs SA) | Result |
|---|---|---|---|---|---|---|
| G1 | 800 | Dense ER (deg≈48) | 160 | 30s | p=0.0000 | **AQLS beats SA** |
| G11 | 800 | Toroidal 4-reg (±1) | 400 | 30s | p=0.0000 | **AQLS beats SA** |
| G14 | 800 | Planar overlay (deg≈12) | 400 | 60s | p=0.0001 | **AQLS beats SA** |
| G22 | 2000 | Spin glass (ER) | 640 | 30s | p=0.0000 | **AQLS beats SA** |

**Key finding:** Optimal k scales inversely with λ₂ (Fiedler value). Dense graphs with large λ₂ need smaller k; sparse graphs with small λ₂ need larger k. This is consistent with RSB theory, the Overlap Gap Property, and spectral graph theory simultaneously.

---

## Novel Contributions

1. **Connected frustrated selector** — seeds from the most frustrated vertex (|gain|≈0) and grows a connected subgraph, producing 2.75× more internal QUBO edges than plain frustrated selection
2. **Look-ahead acceptance** — runs one-flip local search on the QUBO proposal before evaluating delta, comparing local optima to local optima rather than raw proposals
3. **EMA-based adaptive trigger** — replaces fixed stagnation threshold with an exponential moving average of escape probability, eliminating a hyperparameter
4. **Momentum-damped adaptive k** — 10-call window with consistent-trend detection and 5-call cooldown, preventing oscillation
5. **k scales with 1/λ₂** — first systematic empirical confirmation of spectral-guided neighbourhood sizing for QLS/LNS on G-set
6. **Escape rate as a process metric** — first measurement of per-call escape rates across G-set instances; novel contribution regardless of cut quality

---

## Project Structure

```
QLS/
├── src/
│   ├── gain_cache.py       # GainCache with explicit invalidation
│   ├── local_search.py     # OneFlipLS, compute_cut_value, random_cut
│   ├── qubo.py             # build_local_qubo, merge_proposal
│   ├── graph.py            # load_gset, generators, graph_stats
│   ├── backends.py         # backend_exact, backend_neal, backend_sqa
│   ├── selectors.py        # 6 selectors + plateau_aware_rank
│   ├── qls.py              # QLS baseline (random selector, fixed k)
│   ├── adaptive_qls.py     # Adaptive QLS (main contribution)
│   ├── baselines.py        # Improved SA + Tabu search
│   ├── metrics.py          # 9 outcome + process metrics
│   └── experiment.py       # run_experiment, run_gset_experiment, run_k_sweep
├── tests/
│   ├── test_qubo.py        # K4 known-optimum sign test + 3 others
│   └── test_local_search.py # 13 tests including connectivity + 2.75× edge test
├── data/gset/              # G1, G11, G14, G22 benchmark instances
├── results/                # All experiment JSON outputs
├── Literature Review/      # Chapter 2 documents
├── Methodology/            # Algorithm design documents
└── docs/                   # Handoff documents and code reference
```

---

## Quick Start

### Requirements

- Python 3.10
- Windows (tested) or Linux

### Setup

```bash
# Clone
git clone https://github.com/Ahmedt987456321/QLS_MaxCut.git
cd QLS_MaxCut

# Virtual environment
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install
pip install -e .
pip install networkx numpy scipy scikit-learn dimod dwave-neal openjij pytest
```

### Run Tests (17 should pass)

```bash
python -m pytest tests/ -v
```

### Run a Quick Experiment (5 methods, 50-vertex graph, 10s each)

```bash
python -m src.experiment
```

### Run Full G-set Experiment

```bash
# G11 — 7 methods, 10 trials, 30s each (~25 min)
python -c "from src.experiment import run_gset_experiment; run_gset_experiment('G11', budget=30, n_trials=10)"

# k-sweep — find optimal k for an instance
python -c "from src.experiment import run_k_sweep; run_k_sweep('G11', budget=30, n_trials=10)"
```

---

## Algorithm: Adaptive QLS

The main algorithm runs in 8 phases per iteration:

```
1. Classical descent (OneFlipLS) to a local optimum
2. EMA trigger — only invoke solver if stagnating (ema_esc < 0.2)
3. Plateau detection — compute frac_zero for selector awareness
4. Neighbourhood selection — pluggable selector picks k vertices
5. Local QUBO solve — backend solves subproblem over selected vertices
5b. Look-ahead — optionally run OneFlipLS on proposal before evaluating
6. Acceptance + pool update — accept if improving; restart with pool persistence if ema_esc < 0.05
7. EMA update — ema_esc = 0.3 * (delta > 0) + 0.7 * ema_esc
8. Adaptive k — momentum-damped update based on 10-call window
```

### Selectors

| Selector | Description | Source |
|---|---|---|
| `random` | Uniform sample | Tomesh et al. 2022 baseline |
| `frustrated` | Top-k by \|gain\| ascending | Novel |
| `frustrated_connected` | Seed + grow connected subgraph by \|gain\| | **Novel — main contribution** |
| `impact` | Top-k by \|gain\| descending | Atobe et al. 2022 |
| `clustering` | Spectral clustering of solution pool correlations | Zhao & Tang 2025 |
| `meta_rule` | Rule-based switching | Novel |

### Backends

| Backend | Description | Phase |
|---|---|---|
| `backend_neal` | D-Wave simulated annealing (dwave-samplers) | Phase 1 ✅ |
| `backend_sqa` | Simulated quantum annealing (OpenJij) | Phase 1 ✅ |
| `backend_exact` | Brute-force exact (\|S\| ≤ 20) | Phase 1 ✅ |
| SBM | Simulated bifurcation (OpenJij) | Phase 2 — not yet implemented |
| GPU SA | PyTorch 1000 parallel chains | Phase 2 — not yet implemented |
| QAOA | PennyLane gate-model | Phase 2 — not yet implemented |

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ema_esc` init | 0.5 | Initial EMA escape probability |
| `alpha` | 0.3 | EMA smoothing factor |
| EMA threshold | 0.2 | Trigger QLS call when below this |
| k window | 10 calls | Window for adaptive k update |
| k cooldown | 5 calls | Prevent k oscillation |
| k step | ±2 | k update increment |
| Pool max size | 20 | Maximum solution pool entries |
| Pool warm-start | 5 | Diversified cuts before main loop |
| `chi0` (SA) | 0.80 | Ben-Ameur target initial acceptance |
| `beta_final_multiplier` | 5.0 | β_final = β_initial × 5 |
| `reheat_threshold` | 10 | Reheat after 10×N non-improving flips |
| `max_reheats` | 3 | Maximum reheats per run |

---

## G-set Best-Known Values

| Instance | BKS | n | m | Type |
|---|---|---|---|---|
| G1 | 11,624 | 800 | 19,176 | Dense ER, +1 weights |
| G11 | 564 | 800 | 1,600 | Toroidal 4-reg, ±1 weights |
| G14 | 3,064 | 800 | 4,694 | Planar overlay, +1 weights |
| G22 | 13,359 | 2,000 | 19,990 | Spin glass ER, +1 weights |

All four proven optimal by BiqMac/BiqCrunch/BiqBin/MADAM exact solvers.

---

## Key Citations

- **Tomesh, Saleem & Suchara 2022** (Quantum 6, 781) — canonical QLS baseline
- **Liu & Goan 2023** (arXiv:2304.06473) — RL-QLS with fixed-T Metropolis acceptance
- **Atobe, Tawada & Togawa 2022** — Impact-Indexing for sub-QUBO
- **Zhao & Tang 2025** (arXiv:2502.16212) — clustering > impact > random selector
- **Benlic & Hao 2013** — BLS, strong classical baseline
- **Ben-Ameur 2004** (Comp. Optim. Appl. 29) — SA T_initial calibration
- **Myklebust 2015** (arXiv:1505.03068) — linear-in-β schedule, G11=564 BKS
- **Goemans & Williamson 1995** — α_GW ≈ 0.87856 SDP bound
- **Parisi RSB / FRSB** — spin glass theory underpinning funnel structure
- **Chen, Gamarnik, Panchenko & Rahman** — Overlap Gap Property

---

## Branches

| Branch | Contents |
|---|---|
| `main` | MSc submission — final state, frozen |
| `phd-extensions` | PhD research questions: cut polytope selector, λ₂ k-scaling theory, Phase 2 backends |

---

## Implementation Notes

**Critical traps:**

1. **QUBO sign** — `build_local_qubo` builds Q such that *minimising* Q maximises Max-Cut. K4 unit test verifies this.
2. **GainCache invalidation** — every accepted proposal calls `gc.invalidate()`. Every selector calls `gc.assert_valid()`. Stale gains hard-crash.
3. **Zero-bias QUBO skip** — if Q is trivial (all values near 0), the iteration is skipped.
4. **`neal` import** — installed via `pip install dwave-neal`, imported from `dwave.samplers`.
5. **Tabu tenure** — auto-scales: `max(10, len(nodes) // 20)`.

---

## Author

Ahmed Thotlapalli — MSc Quantum & AI Systems

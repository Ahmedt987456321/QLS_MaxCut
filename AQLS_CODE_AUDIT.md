# AQLS Code Audit — Consolidated Reference
2026-06-01 progress: File #1 verified (invariant tests green). File #2 qubo + backend_exact F5 fixed, proven by QUBO=cut test. selectors.py: F9 fixed+verified, F52 selector tests added, F25 fallback made visible. adaptive_qls.py: F25 fallback visible, F22 resolved (deliberate fixed-k). All committed to phd-extensions.

**Scope:** Full read of the AQLS/QLS Max-Cut codebase — `gain_cache.py`, `local_search.py`, `qubo.py`, `selectors.py`, `backends.py`, `adaptive_qls.py`, `qls.py`, `baselines.py`, `graph.py`, `experiment.py`, `metrics.py`, plus `tests/test_qubo.py` and `tests/test_local_search.py`.

**Purpose:** A single source of truth so the thesis prose, the code, and the reported numbers cannot drift. Every formula the code computes is written out (E1–E34). Every concern is logged (F1–F54), tiered by impact, with the exact change for each must-fix.

**Headline verdict:** The core result-producing chain is mathematically correct:
`load (signs preserved) → gain (E2) → QUBO (E7/E8, incl. signed weights) → neal (E15) → cut (E1) → median/ratio (correct) → Wilcoxon (correct arithmetic)`.
The exposure is **not** in the MSc result arithmetic. It is in (a) the Fiedler/balance theory-vs-code gap, (b) silent fallbacks that can make a labelled selector run a blend, (c) reporting/test-methodology choices a reviewer would scrutinise, and (d) a test suite that confirms *runs* but does not *prove* the invariants.

---

## Part 1 — Equations & Calculations Ledger

Notation: `x_v ∈ {0,1}` binary assignment; `w_uv` edge weight (signed for ±1 instances); `N(v)` neighbours of v; `g_v` flip gain; `C(x)` cut value.

### Foundation (`gain_cache.py`, `local_search.py`)

- **E1 — Cut value.** `C(x) = Σ_{(u,v)∈E} w_uv · 1[x_u ≠ x_v]`. Sums *signed* weights, so cutting a −1 edge subtracts. Correct for weighted/signed Max-Cut. *Verified.*
- **E2 — Flip gain.** `g_v(x) = Σ_{u∈N(v)} w_uv · (+1 if x_u=x_v else −1)`. Verified algebraically: `g_v(x) ≡ C(flip v) − C(x)` exactly. This is the load-bearing identity. *Verified.*
- **E3 — Descent step.** `v* = argmax_v g_v`; accept iff `g_{v*} > 0`; else one-flip local optimum. Best-improvement. *Verified.*
- **E4 — Incremental update set.** After flipping v, recompute `g_w` for `w ∈ {v} ∪ N(v)` only. Complete and sufficient. *Verified.*
- **E5 — Plateau move** (`one_flip_ls_with_plateau`, BLS only). When `g_{v*} ≤ 0`, take first `g_v = 0` non-tabu vertex, up to `max_plateau`. Uses exact `== 0` (integer-weight assumption).

### QUBO (`qubo.py`)

- **E6 — Binary cut identity.** `1[x_u ≠ x_v] = x_u + x_v − 2 x_u x_v`.
- **E7 — Internal edge** (u,v ∈ S). `Q_uu += −w; Q_vv += −w; Q_uv += +2w`, giving per-edge energy `−w·1[x_u≠x_v]`. Minimising energy = maximising internal cut. *Verified at all corners, incl. w<0.*
- **E8 — Boundary edge** (u∈S, v fixed). `Q_uu += −w if x_v=0 else +w`. *Verified.*
- **E9 — QUBO energy.** `E = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j`. *Note:* not the cut value — only the S-dependent part up to a constant. Never compare to `compute_cut_value`.

### Selectors (`selectors.py`)

- **E10 — Signed Laplacian (balance test).** `σ_uv = sign(w_uv); L = D − A_σ`, `balanced ⟺ λ_min(L) ≈ 0` (Zaslavsky). PSD operator. **Tests the wrong signature for Max-Cut — see F9.**
- **E11 — Frustration-weighted Laplacian (Fiedler).** cut edge → `+1`, non-cut → `−1`; `ew = |w|·sign`; `L = D − A`. Code takes **second-smallest** eigenvector (`eigsh(L,k=2,'SM')[:,1]`). **This is the ordinary Fiedler vector of D−A, not the signless-Laplacian indicator the thesis cites — see F7.**
- **E12 — Combined score.** `gain_score = 1/(1+|g_v|) ∈ (0,1]`; `fiedler_score = 1/(0.005+|f_v|) ∈ (0,200]`; `score = 0.6·gain + 0.4·fiedler`. **Scales mismatched; Fiedler term dominates — see F8.**
- **E13 — λ₂ routing.** Unsigned Laplacian, `eigsh(k=2,'SM')`, take `sorted[1]`.
- **E14 — Odd-cycle violation (cut_polytope).** Triangle violated iff `n_cut ∈ {0,2}`; `violation = (w_uv+w_vw+w_uw)/3`.

### Backends (`backends.py`)

- **E15 — neal QUBO→BQM.** Diagonal→linear, off-diagonal→interaction; dimod symmetrises internally (immune to key-order). *Verified; this is the path for all headline results.*
- **E16 — SBM QUBO→Ising.** `h_i −= Q_ii; J_ij −= Q_ij/2` then `maximize(domain='spin')`; spin +1→1, −1→0. **Incomplete substitution — see F17.**
- **E17 — exact.** Brute force over 2^|S|, minimise inline energy. *Correct objective, modulo F5 key-order.*

### Adaptive loop (`adaptive_qls.py`)

- **E18 — EMA trigger.** Skip solver while `not at_opt and ema_esc ≥ 0.2`.
- **E19 — delta.** `delta = C(x_prop) − current_cut` (post-look-ahead if `acceptance='lookahead'`). *Uses verified E1.*
- **E20 — acceptance.** improvement: `delta>0`; sa_cooling: also `exp(delta/T)`, `T←max(T_min,T·cooling)`; fixed_temp: `exp(−|delta|/T_initial)` (**direction-blind — F26**).
- **E21 — EMA update.** `ema_esc ← 0.3·1[delta>0] + 0.7·ema_esc`. *Verified.*
- **E22 — momentum-damped k.** Means `t0,t1,t2` of last 9 deltas; `t0<t1<t2 → k−=2`; `t0>t1>t2 → k+=2`; cooldown 5. **Trend sign suspect — F22.**

### Experiment / metrics (`experiment.py`, `metrics.py`)

- **E23 — seed schedule.** `seed = trial*1000 + 42` → 30 distinct seeds. *Verified; closes F3.*
- **E24 — approx ratio.** `best_cut / best_known`. BKS: G1 11624, G11 564, G14 3064, G22 13359.
- **E25 — escape rate.** `escape_successes / qls_calls`, success = `delta>0`.
- **E26 — Wilcoxon.** `stats.wilcoxon(a,b)` on paired `best_cut` arrays; sig at `p<0.05`. **Pairing/test-choice — F29.**

### Baselines (`baselines.py`)

- **E27 — T calibration (Ben-Ameur).** `T = −mean(|Δ⁻|)/ln(χ₀)`, χ₀=0.80. *Verified.*
- **E28 — Cooling (linear-in-β).** `β(t) = β_init + (β_final−β_init)·(elapsed/budget)`, `β_final = 5·β_init`. *Verified.*
- **E29 — SA acceptance.** improving always; worsening `exp(Δ/T)`. *Verified.*
- **E30 — Incremental cut tracking.** `current_cut += g_v`. Valid because E2 = exact delta. *Verified.*
- **E31 — BLS perturbation prob.** `p_directed = max(0.8, exp(−ω/T_stag))`.
- **E32 — BLS L adaptation.** same attractor → `L=min(L+1,L_max)`; new basin → `L=L0`.

### Loader / baseline-QLS (`graph.py`, `qls.py`)

- **E33 — G-set parse.** `w = float(parts[2])` stored verbatim. **Signs preserved** — not normalised/dropped. Header asserts on node/edge counts. *Verified; closes F-loader.*
- **E34 — QLS core.** Same primitives as AQLS without adaptation; improvement-only; full random restart on failure (**F47**).

---

## Part 2 — Findings, Tiered

Impact tags carried from the per-file review. "Affects MSc result?" answers whether the locked G1/G11/G14/G22 numbers are touched.

### TIER 1 — MUST FIX BEFORE WRITING
*(changes what can be claimed, can produce/hide a wrong number, or a reviewer will catch)*

**F44 — Resolve bipartite vs balance (the keystone — do this FIRST).**
The whole Fiedler theory tangle (F7/F9) hinges on what G11's edge signs actually are. One diagnostic settles it and likely *simplifies* the theory.
*Action:* run the weight-sign/bipartite diagnostic (Part 4 #1). If G11 is all-`+1` and `nx.is_bipartite=True` (G13 False), the correct predictor is **plain bipartiteness**, the Desai–Rao-on-the-unsigned-graph story holds cleanly, and the balance machinery should be deleted in favour of `nx.is_bipartite`. If G11 has genuine `−1` edges, keep signed-balance framing but fix the signature (F9).

**F7 — Fiedler selector does not implement the cited Desai–Rao mechanism. (CRITICAL, affects the *explanation* of the Fiedler result, not the empirical 564.)**
Thesis narrates "smallest eigenvector of signless Laplacian Q=D+A *is* the optimal bipartition." Code builds **D−A** and takes its **second-smallest** eigenvector, used only as a **selection score** (neal does the optimisation). Three mismatches: operator (D−A vs Q=D+A), eigenvector index (2nd-smallest vs smallest-of-Q), and role (neighbourhood heuristic vs cut-recovery operator).
*Fix:* Either (a) rewrite Section 4 to describe the honest mechanism — "Fiedler band concentrates the k-neighbourhood; neal solves it" — and drop the exact-recovery-via-Desai–Rao causal claim; or (b) change the code to actually compute the signless-Laplacian indicator if you want to keep that claim. Decide after F44.

**F8 — The α=0.6/0.4 score blend is illusory; Fiedler term silently dominates. (CRITICAL calculation.)**
`gain_score ∈ (0,1]` vs `fiedler_score ∈ (0,200]` (because `eps=0.005`). The 0.4-weighted term swamps the 0.6-weighted one. "60% gain / 40% Fiedler" is not what runs.
*Fix:* min-max (or z-) normalise each score to [0,1] before the convex blend. Until then make no statement about the gain/Fiedler balance.

**F9 — Balance computed under the wrong sign convention for Max-Cut. (CRITICAL, affects routing claims.)**
`is_signed_graph_balanced` uses `σ = sign(w)`. For Max-Cut the relevant signature is the negated (anti-ferromagnetic) one. Under the code's convention, all-`+1` graphs (G14/G22/G1) are trivially balanced → tier-1 rule routes them to **Fiedler** — the opposite of the empirical truth. Likely masked in runs only because `which='SM'` fails → `nan` → falls through (i.e. "correct" routing by accident, not design).
*Fix:* use bipartiteness under the all-negative signature, i.e. `nx.is_bipartite(G)` for the unsigned graph (the originally commented-out line). Confirm against F44.

######### updated F9 is below ##################
**F9 — Balance routing CONFIRMED ANTI-CORRELATED — RESOLVED (2026-06-01).**
Diagnostic (check_signs.py + check_balance.py): G11/G12/G13 are genuinely signed (±1, real negative edges); G14/G22/G1 are all-+1. The old is_signed_graph_balanced was anti-correlated — it said False for the bipartite ±1 graphs where Fiedler WINS (G11/G12) and True for the +1 graphs where Fiedler is HARMFUL (G14/G22/G1). nx.is_bipartite gives the correct split (True/True/False/False/False/False).
FIX APPLIED + committed: select_smart_adaptive now routes by nx.is_bipartite (bipartite -> Fiedler, else -> FConn). Verified on all six (check_route.py): G11/G12 -> fiedler, G13/G14/G22/G1 -> fconn.
Re-run flag: any result via the smart_adaptive / adaptive_spectral ROUTER used the inverted predictor and needs re-checking. Pure select_fiedler results (564 on G11) are unaffected. Locked MSc FConn-LA results are unaffected (never used the router).

**F25 — Silent selector fallback can make a labelled run a blend. (HIGH.)**
Two layers of `except → fallback`: Phase 4 in `adaptive_qls` (`except → random`) and inside `select_fiedler` (`except → frustrated_connected`). An "AQLS-Fiedler" run could silently contain random + FConn selections with no record.
*Fix:* add a fallback counter/log to both paths; during validation runs, raise instead of swallow. Re-run one G11 Fiedler trial and confirm fallback count = 0 before claiming "Fiedler achieves 564."

**F5 / F50 — Fix `backend_exact` (the oracle) and stop the K4 test laundering it. (CRITICAL for verification integrity.)**
`qubo_energy` and `backend_exact` look up off-diagonal `Q[(vi,vj)]` in one key order; `build_local_qubo` stores in `G.edges()` order. On disagreement the term is silently counted 0. The K4 test passes *through* the buggy `qubo_energy` and only because K4 is tiny/symmetric — so the foundational correctness test certifies less than it appears to. The "2.75× internal edges" claim is a *printed* value, asserted only as `>=`, never as a ratio.
*Fix:* in both `qubo_energy` and `backend_exact`, read `Q.get((vi,vj),0)+Q.get((vj,vi),0)`. Then the oracle is trustworthy and can certify the pipeline. Separately, assert the internal-edge ratio (or report it as observed-on-one-seed, not as a tested invariant).

**F51 — Add the two invariant tests that actually certify the pipeline. (HIGH.)**
Missing: (1) `gain = delta` — `g_v(x) == C(flip v) − C(x)` over random (x,v); (2) `QUBO drop = cut gain` via a *fixed* exact oracle over random S. These prove E2≡E1 and E7/E8≡E1.
*Fix:* add both (Part 4 #3). Want these green before trusting any median.

**F29 — Wilcoxon pairing validity / test choice. (HIGH, every p-value.)**
`stats.wilcoxon(a,b)` treats trial i of A and B as a matched pair. They share the initial `random_cut` seed but diverge in rng consumption, so the pairing is weak. If paired (CRN) is intended, justify it and ensure identical seeding; otherwise use **Mann–Whitney U**.
*Fix:* decide the design; use the matching test. Recompute one p-value both ways (Part 4 #5) — if significance agrees, the worry is academic; if it diverges, justify before quoting.

**F30 — `p={p:.4f}` prints `0.0000`, violating your own rule. (HIGH reporting.)**
The "p<0.0001, never 0.0000" rule is currently a manual post-hoc correction, not enforced in code.
*Fix:* formatter emits `p < 0.0001` when `p < 1e-4`. Also: 3-trial pilots (e.g. `test_3inst`) carry **no** valid p-value — report directional only.

### TIER 2 — SHOULD FIX
*(matters for robustness, reproducibility, or comparator strength; does not bias the MSc result)*

- **F10 — Eigensolver fragility.** `which='SM'`, `tol=1e-3`, `maxiter=1000`, bare `except`. Switch to shift-invert (`sigma=0, which='LM'`) or `which='SA'`, tighten tol, and log non-convergence. (Underlies F9 masking.)
- **F11 — Gate vs selector use different matrices.** Router decides on static weight-sign signed Laplacian (E10); `select_fiedler` acts on dynamic cut-state frustration Laplacian (E11). Make the deciding operator and the acting operator consistent.
- **F12 — Magic thresholds.** `adaptive_spectral` routes on `λ₂<1.0`; `smart_adaptive` on `λ₂<2.0`. State one threshold, justify or learn it, use everywhere.
- **F13 — Redundant EMA + double work in `smart_adaptive`.** Its own α=0.3 escape-EMA duplicates the loop EMA and computes `compute_cut_value` twice per call. Pick one EMA as source of truth.
- **F14 — `id(G)` cache key unsafe.** CPython can reuse freed addresses. Use a stable key (instance name or `(n,m,hash(sorted edges))`).
- **F22 — Adaptive-k trend sign suspect.** "improvement trend → shrink k" reads as a non-sequitur; mixing improving/worsening deltas on one scale is ambiguous. **First confirm whether finals used fixed `k_min==k_max`** (Part 4 #4) — if so this controller is dormant in the headline runs and only needs a docstring/claim correction.

#############updated F22 is below #########

**F22 — Adaptive-k controller: RESOLVED, no fix needed (deliberate design).**
Confirmed from Implementation-chat history: final 30-trial runs used FIXED k per instance (k_min==k_max: G1=160, G11=400, G14=400, G22=640), chosen from a k-sweep. This was deliberate — fixing k is what made the spectral law (optimal k proportional to 1/lambda2, R^2=0.94) measurable; an active controller would have masked that effect. The controller's suspect trend-sign logic never fired in any reported result.
Action: no code change. Thesis framing: "We swept k, found optimal k proportional to 1/lambda2 (structural finding); finals use the empirically-optimal fixed k per instance; adaptive self-tuning of k is future work." Fixed-k is a stated strength, not a missing feature.

- **F23 — Budget straddle / overshoot.** `while time()<budget` gates entry but a long descent can overshoot. Symmetric across AQLS/QLS/SA, but confirm per-trial elapsed ≈ budget so "equal wall-clock" holds. (Same issue F49.)
- **F24 — Fiedler eigenvector recomputed every call.** Confounds Fiedler-vs-FConn *timing*; also runs F8 every call. Cache eigenvector once per graph (Research-chat recommendation, never applied).
- **F31 — `Metrics.best_cut = 0.0` silent zero.** A never-recorded run yields `best_cut=0.0` flowing in as a real datum. Init to `-inf`; assert `cut_trace` non-empty before trusting `best_cut`.
- **F32 — `if approx_ratio:` / `if best_known:` truthiness.** Skips a legitimate 0.0. Use `is not None`; else the ratio list can desync from `best_cut`.
- **F33 — k-sweep range vs reported optima.** Sweep tests `[10,20,40,80,160]`; finals use k=400 (G11), 640 (G22) — outside the grid. Confirm an extended sweep brackets the reported optima, or the 1/λ₂ scaling (R²=0.94) is extrapolated.
- **F34 — time-to-target nan propagation.** Misses store `nan`; aggregating with `np.mean` → nan. Filter nans in any reported mean-time-to-target.
- **F37 — SA reheat overwrite.** Top-of-loop recompute of `β/T` discards reheated values unless `remaining>0.05`. Reheating is intermittently a no-op. *Does not bias results* (only weakens SA, making your win easier) — fix so prose matches behaviour.
- **F38 — BLS M2 selects on stale gains.** v1 chosen from a pre-v0-flip snapshot. Cut tracking stays exact; only BLS quality affected. Re-select v1 after v0's incremental update.
- **F39 — SA `_calibrate_T_initial` unseeded rng.** Breaks within-seed reproducibility of SA. Thread the seeded rng.
- **F40 — BLS warm-up counted inside budget.** Small asymmetry *against* BLS (not SA). Move warm-up outside the timer or note BLS is slightly disadvantaged.
- **F17 — SBM QUBO→Ising mapping incomplete.** `h−=Q_ii; J−=Q_ij/2` drops the `x_ix_j → (s_i+s_j)/4` field cross-terms of the full substitution; silent `except → neal` + `avg_degree<5 → neal` mean "SBM" runs can be entirely neal. **Do not cite SBM numbers** until verified against `backend_exact` on a small dense QUBO. (Affects only the shelved SBM PhD claim.)
- **F18 — neal warm-start format.** `initial_states=[single dict]` with `num_reads=100` may be silently ignored. Verify the warm start is applied (relevant to PhD Q5).
- **F52 — Fiedler/adaptive/smart selectors untested.** Add unit tests (valid subset, connectivity, fallback-not-silently-taken) — currently zero coverage on the PhD-extension path.

### TIER 3 — COSMETIC / LOW
*(no correctness or claim impact)*

- **F1** plateau `==0` float-fragility (safe for integer weights). **F2** docstring complexity wording. **F4** cut float vs gain int. **F6** silent backend dropout (add `assert set(result)==set(S)`). **F15** toroidal-grid detection assumes contiguous labels (topological selectors shelved). **F16** mixed determinism in clustering/random selectors — ensure seeded rng always passed. **F19** top-level SBM imports break neal/exact if package missing — move inside function. **F20** `get_backend` omits `sbm`. **F21** sqa vs neal dropout default inconsistency. **F26** `fixed_temp` direction-blind & accepts delta=0 (only if cited). **F27** epsilon zoo (1e-6 / 1e-10 / 0.005) — document each. **F28** pool warm-start timing consistency vs SA. **F35** escape_rate prose definition must equal `escape_successes/qls_calls`. **F36** duplicate `import numpy`; `__main__` between defs. **F42** BLS weight detection samples only 20 edges — check all (fix via F46). **F45** generators all-+1 (collapses bipartite-vs-balance on generated graphs). **F46** loader stores no `weight_type` — add `G.graph['weight_type']`. **F47** QLS full-restart-on-failure → escape-rate cross-comparison with AQLS needs a prose caveat (cut-quality comparison is clean). **F48** QLS no warm start (baseline, fine). **F49** QLS budget straddle (symmetric). **F53** `test_qubo_energy_zero_assignment` is a tautology. **F54** tests use only +1 generated graphs — signed path never exercised.

### CLEARED (checked, no issue)
- **F3** determinism — resolved by E23 (30 distinct seeds).
- **F41** SA gain-weighted selection — *strengthens* SA; makes the comparison harder to win, i.e. in your favour. Label it "improved/gain-guided SA," not "standard SA."
- **F43** Tabu aspiration logic — correct.
- **F-loader** — closed by E33: signs preserved, header asserts present.
- **BLS prior bugs** (perturb-before-search, blank-cache reset) — appear genuinely fixed; FIX SUMMARY matches code.

---

## Part 3 — Assumptions Ledger

1. **Binary {0,1} encoding** throughout the classical path; flip = `1−x[v]`. No spin {−1,+1} except inside SBM (E16), where the conversion is suspect (F17).
2. **`x` is total** (keyed by every node). `random_cut` guarantees it; any partial builder elsewhere risks `KeyError` in `_compute_gain`.
3. **Weights via `'weight'`, default 1.0.** Unweighted load silently becomes all-+1.
4. **Objective = maximise signed sum over cut edges** (E1). For G11 must match the convention under which BKS=564 — confirmed sign-preserving at load (E33); confirm the *value* via F44.
5. **`best_flip` tie-break = node insertion order** (deterministic, not random).
6. **Plateau/degenerate tests assume exact-zero gains** (`==0`) — true for ±1/+1, breaks on fractional weights.
7. **dimod symmetrises Q** — why neal (E15) is immune to the key-order bug that bites `backend_exact`/`qubo_energy` (F5).
8. **id(G) stable within a run** — assumed by selector caches (F14 says not guaranteed).
9. **Single seeded rng per run** threads through selectors, restarts, acceptance — good for reproducibility, but streams are entangled (can't reason about one in isolation).

---

## Part 4 — Open Runtime Questions (diagnostics to run)

These cannot be settled by static reading; each is a short paste-and-run. Listed in priority order — #1 may reclassify F7/F9.

1. **Weight-sign / bipartite (settles F44/F7/F9 — the keystone).** For G11/G12/G13/G14/G22/G1 print `min_w, max_w, n_neg, nx.is_bipartite`. Expected: G11/G12 bipartite, G13 not; if all-+1, switch routing to `nx.is_bipartite` and the theory simplifies.
2. **Fallback counter (settles F25/F52).** Instrument both `except` paths; run one G11 Fiedler trial; confirm fallback count = 0.
3. **Invariant tests (settles F51).** Add gain=delta and QUBO-drop=cut-gain (through fixed oracle); run on a ±1 graph too (closes F54).
4. **Adaptive-k on/off in finals (settles F22/F33).** Print `k_min,k_max` for the locked G1/G11/G14/G22 configs. If `k_min==k_max`, the k-controller is dormant for headline results.
5. **p-value both ways (settles F29).** From saved JSON, run `wilcoxon` and `mannwhitneyu` on G11 SA-vs-FConn `best_cut`. Compare significance.
6. **Ratio re-derivation (sanity).** Recompute each table ratio as median ÷ BKS (e.g. 2980/3064 = 0.9726, not 0.9736) to catch arithmetic slips.

---

## Part 5 — Bottom Line by Original Question

1. **Is the code wrong?** The MSc result chain is correct. Real bugs: F5 (oracle key-order — fix before using it to verify), F37 (SA reheat — weakens a feature, doesn't bias), F22 (adaptive-k trend sign — likely dormant in fixed-k finals). The Fiedler selector computes a different object than its narration (F7), with a broken score blend (F8) and wrong-signature routing (F9).
2. **Are we missing theory?** Yes — F44 is the keystone; resolving it most likely *simplifies* the story to clean bipartiteness and makes the Desai–Rao citation correct.
3. **Did a calculation get confused?** Yes — F8 (scale-mismatched blend), F30 (`0.0000`), F29 (test choice), and "2.75×" being printed-not-tested (F50).

**The reassuring synthesis:** everything producing the locked MSc and Fiedler numbers runs through verified math (`backend_neal` + `compute_cut_value`). The work before writing is (i) run the six diagnostics, (ii) clear Tier 1, (iii) reconcile the Fiedler prose with what the code actually does.

File #6 baselines.py / qls.py — DONE (2026-06-01): comparators confirmed FAIR and correct. SA is strong (auto-calibrated, gain-guided) — label "gain-guided SA" not "standard SA" (F41). Tabu standard. QLS correct (note full-restart-on-failure vs AQLS pool, F47, only matters for escape-rate comparison). BLS bugs fixed; warm-up counted in budget slightly disadvantages BLS (F40) — works against our favour, so win is not inflated. No code changes needed.

AUDIT FILE PASS COMPLETE: all files audited. Result-affecting bugs fixed (F5, F9, F25), equations proven by invariant tests, p-value reporting enforced (F30), metrics silent-zero closed (F31/F32). Remaining: F8 (parked, only if Fiedler scoring cited), F7 (thesis prose), F29 (test-choice diagnostic), assumption guards A2/A3/A7 (optional hardening).


F29 — RESOLVED (2026-06-01, check_test_choice.py on G11_k400_final_30trials).
SA vs AQLS-FConn-LA-k400, 30 trials each:
  Wilcoxon (paired):       p = 2.46e-06
  Mann-Whitney (unpaired): p = 9.71e-10
Both significant. The conclusion is independent of the paired-vs-unpaired choice. Thesis can state significance under BOTH tests, which forecloses the test-choice objection entirely. No code change; reporting note only.


A3 — RESOLVED (2026-06-01). load_gset now records G.graph['weight_type'] (signed/unweighted/weighted) and ['n_negative_edges'], detected from ALL edges. Verified: G11/G13=signed, G14/G1=unweighted. Protects against silently running on a wrong-typed graph.
F29 — RESOLVED (see entry). Significant under both Wilcoxon (p=2.46e-06) and Mann-Whitney (p=9.71e-10) on G11 30-trial. Conclusion independent of test choice.

CODE SIDE OF AUDIT COMPLETE. All result-affecting findings fixed and verified by tests. Remaining items are writing-time only: F7 (Fiedler prose), F8 (only if Fiedler scoring cited), plus framing notes (gain-guided SA label, deliberate fixed-k, benchmark-selection rule).

A3 — RESOLVED (2026-06-01). load_gset now records G.graph['weight_type'] (signed/unweighted/weighted) and ['n_negative_edges'], detected from ALL edges. Verified: G11/G13=signed, G14/G1=unweighted. Protects against silently running on a wrong-typed graph.

F29 — RESOLVED (2026-06-01, check_test_choice.py on G11_k400_final_30trials). SA vs AQLS-FConn-LA-k400, 30 trials: Wilcoxon (paired) p=2.46e-06, Mann-Whitney (unpaired) p=9.71e-10. Both significant. Conclusion is independent of paired-vs-unpaired choice — thesis can state significance under BOTH tests. No code change; reporting note only.

CODE SIDE OF AUDIT COMPLETE (2026-06-01). All result-affecting findings fixed and verified by tests (F5, F9, F25, F30, F31, F32, A3). Equations proven (gain=delta, QUBO=cut invariant tests green). Remaining items are writing-time only: F7 (Fiedler prose honesty), F8 (only if Fiedler internal scoring is cited). Framing notes to state in thesis: gain-guided SA label (not "standard SA"), deliberate fixed-k (enables the λ₂ law), benchmark-selection rule (one instance per structural family).

G22 PREMISE CORRECTION (2026-06-01). G22 is NOT toroidal/spin-glass — it is a
RANDOM graph (n=2000, 19,990 edges, avg degree ~20), the n=2000 analogue of G1.
Per the Bonn BQP library: G22-G42 mirror G1-G21 types at n=2000. The toroidal
n=2000 instances are G32-G34. Action: anywhere the thesis/notes describe G22 as
"toroidal" or "spin-glass-like," correct to "random graph." This affects framing
only — the locked G22 result (SA 13128 vs AQLS 13198.5) is unaffected; it's a
valid comparison regardless of structural label. G11/G12/G13 remain the genuine
toroidal ±1 spin-glass instances.
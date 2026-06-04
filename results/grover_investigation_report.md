# Grover / Sub-Problem Investigation ? Findings Log (2026-06)
## Origin & scope
Began as "can Grover help AQLS?"; pivoted to "characterize the sub-QUBOs directly."
Instances: G11 (toroidal, n=800, m=1600, deg-4); G22 (dense random, n=2000, m=19990).
Recurring mechanism, established repeatedly: NEAL ABSORBS INNER-LOOP CLEVERNESS at k=100-600.
All numbers below are exact run outputs. Headline finding (Sec 5) is genuine and neal-confirmed;
most intermediate threads are clean NEGATIVE results.

## 1. Quantum threads (all CLOSED)
1.1 Grover as backend: not viable. At k=160-640, GAS needs sqrt(2^k) coherent oracle
    iterations (2^80 at k=160), fault-tolerant only, beats brute force not neal.
    Largest better-than-classical Grover demo to date: 3-4 qubits.
1.2 Short-path (beyond-Grover): exists (Hastings 2018; Dalzell-Pancotti-Campbell-Brandao
    STOC 2023) but irrelevant at scale. Proven constants microscopic:
    c = 5.22e-7 (3-SAT), 2.7e-5 (SK), 2.24e-4/k^3 (k-spin).
    2^{cn} reaches factor 2 only near n ~ 1.9 million. JPMorganChase numerics (2410.23270)
    on MIS <=21 vertices: O*(2^{0.400n}) -- SLOWER than classical.
    Le Gall-Tamaki dequantization (Apr 2026, 2604.12131): classical 2^{(1-c')n}, c'>c,
    for MAX-k-CSP/QUBO -> kills the super-quadratic claim.
1.3 Quantum-relevance condition (Dalzell dichotomy): need rare/isolated near-optima,
    C((1-eta)E*) <= 2^{(1-gamma)n}. Empirical regime test (k=200):
      G11: 0 case-b, 5 mixed, 3 case-a; mean near95 frac 0.082; rand-config ratio 0.037
      G22: 8 case-b, 0 mixed, 0 case-a; mean near95 frac 0.595; rand-config ratio 0.608
    G22 saturated with near-optima (classical wins). G11's 3 "case-a" later shown finite-size noise.

## 2. Borrowed-technique threads (both NULL)
2.1 Delocalisation routing: synthetic strong (Spearman rho=+0.85, p=3.6e-8; separation
    term rho=-0.48 dropped). Real instances collapsed:
      G11 k400 neal  rho=+0.37 p=0.019
      G11 k120 neal  rho=-0.18 p=0.27
      G22 k640 neal  rho=+0.19 p=0.25
      G11 k400 SD    rho=+0.47 p=0.002
      G11 k120 SD    rho=+0.15 p=0.36
      G22 k640 SD    rho=+0.01 p=0.93
      G22 k200 SD    rho=-0.05 p=0.75
    Sign flips with size/instance -> noise around zero. DEAD.
2.2 Spectral warm-start saving neal budget: gap (spectral vs cold neal) by sweeps:
      sweeps 1   : G11 +55.5%  G22 +7.4%
      sweeps 5   : G11 +1.1%   G22 +0.4%
      sweeps 10  : G11 +1.6%   G22 -0.4%
      sweeps 20  : G11 +0.3%   G22 +0.2%
      sweeps 40-160: ~0 both
    Advantage only at 1 sweep, gone by 5. AQLS runs >>5 sweeps. DEAD. (neal fast-mixing.)

## 3. Static landscape (calibrated): "twins + rugged"
3.1 Exact, k<=18 (canonical optima, +/- flip folded):
      k   G11_nlo gdeg FDC   | G22_nlo gdeg FDC
      10  11      1   0.420  | 34      1   0.413
      12  89      1   0.378  | 95      3   0.449
      14  128     1   0.326  | 157     3   0.457
      16  116     10  0.572  | 843     3   0.369
      18  183     1   0.324  | 1597    1   0.309
    Both rugged; both moderate positive FDC ~0.3-0.6 (real funnel); statistically TWINS.
3.2 Sampling k=100-600: saturation linear 50:50 100:100 200:200 400:400 600:600 on every
    support, both instances -> every restart finds a NEW optimum -> optima count effectively
    unbounded; global optimum non-degenerate (gopt-deg=1) at scale.
3.3 FDC at large k: NOT measurable by sampling. 3 estimators failed calibration vs exact +0.4-0.55:
      over-local-optima : sampled ~0
      random-perturb    : sampled NEGATIVE (-0.31..-0.16)
      descent-trajectory: sampled ~0 (-0.06..+0.07)
    Honest: funnel real at k<=18, UNMEASURABLE (not absent) at k>=100.
Net: uniformly rugged at scale; toroidal vs random indistinguishable by static geometry.

## 4. THE FINDING: dynamical signature of toroidal structure (VALIDATED)
Pivot: measure the landscape THROUGH solver behavior (no calibration wall; direct observation).
4.1 Initial (k=400, 5 reps, Metropolis):
      metric          G11    G22
      frac2plateau    0.577  0.548
      late-gain%      6.74   2.84
      late-accept     0.077  0.035
    AQLS-connected vs random support: nearly identical (Q2 NULL -- selection != easier sub-QUBO).
4.2 Q3 confirmation (k=400, 30 reps, Mann-Whitney):
      metric        G11     G22     p
      late-gain%    5.763   2.338   6.1e-11
      late-gain raw 14.833  21.367  2.8e-5
      late-accept   0.071   0.036   9.0e-11
      frac2plateau  0.606   0.544   6.0e-9
      down-steps    0.263   0.286   4.8e-6
    All significant. Toroidal plateaus later, accepts ~2x more late moves, improves more as
    fraction of optimum; dense commits faster with larger raw swings. Artifact-free metric:
    late acceptance (0.071 vs 0.036).
4.3 Confound broken -- STRUCTURE not sparsity (k=200, 30 reps):
      arm                      edge-dens  late-accept
      G11 toroidal (sparse)    0.017      0.068
      random-4-regular sparse  0.012      0.056
      G22 random dense         0.019      0.052
      toroidal vs sparse-random (matched density): p=0.018 DIFF
      sparse-random vs dense:                       p=0.46  ns
    Density does NOT drive it; lattice structure does.
4.4 Bulletproof -- 5 control seeds + REAL NEAL (k=200):
      G11 Metropolis late-accept = 0.0676, ABOVE all 5 controls (means 0.051-0.058):
        seed7  0.0559 p=0.018 DIFF
        seed13 0.0569 p=0.054 ns
        seed21 0.0566 p=0.009 DIFF
        seed42 0.0513 p=0.001 DIFF
        seed99 0.0579 p=0.047 DIFF
      REAL NEAL (late-activity proxy, 10 reps): G11=0.555 vs rand4reg=0.389, p=0.009 DIFF.

## 5. Bottom line
(1) No quantum method (Grover or short-path) gives usable advantage on AQLS sub-QUBOs at
    k=100-600. Established rigorously.
(2) Sub-QUBOs uniformly rugged at scale; moderate funnel at small k; toroidal vs random are
    static-geometry twins.
(3) VALIDATED POSITIVE FINDING: toroidal/lattice sub-QUBOs produce a distinct annealing
    signature -- solver sustains ~2x higher late-stage acceptance (Metropolis 0.068 vs 0.056,
    4/5 controls p<0.05; NEAL 0.555 vs 0.389, p=0.009) -- driven by LATTICE STRUCTURE not
    sparsity or size, confirmed on the production solver.

## 6. Caveats (do not overstate)
- neal confirmation uses a late-activity PROXY (spins still flipping in 2nd half), not true
  per-sweep acceptance -> corroboration, not direct measurement.
- Metropolis effect is MODEST (0.068 vs 0.056); one control at p=0.054.
- Demonstrated on ONE toroidal instance (G11); "lattice structure" needs G12/G13/G32.
- "Toroidal" is more precisely "2D lattice geometry."
- Several cited 2025-2026 papers (dequantization, QOGP) are recent preprints.

## 7. Next step to make Sec 4 thesis-grade
Repeat 4.4 on G12/G13/G32; replace neal proxy with true per-sweep acceptance trace
(neal lower-level interface or schedule-matched custom annealer). Converts finding from
"G11, neal-proxy-confirmed" to "lattice structure, neal-confirmed" -- a dynamical companion
to the existing Fiedler toroidal result.

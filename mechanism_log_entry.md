# Research Log — Mechanism of FConn-vs-Fiedler Selector Advantage (Toroidal Case)

**Date:** 2026-06-05
**Branch:** phd-extensions
**Status:** Causal result on toroidal family — VERIFIED across disorder patterns. Cross-family question — CLOSED as negative (no single observational predictor).

---

## 1. Question

Why does the Fiedler-based sub-QUBO selector beat the gain-based (FConn) selector on some
instances (e.g. G11) and not others? Is there a single mechanism / predictor?

## 2. What was tested and FALSIFIED (observational, across instances)

Five candidate cross-family predictors of the winner were tested and **all failed** — each
separates the coupled/dense regime (high beta_e -> FConn) but **none separates winners within
the low-beta_e regime**, where Fiedler-wins, ties, and FConn-wins are interleaved:

1. **Alignment** (Fiedler support vs disagreement components) — bestJaccard ~0.20, scattered. Rejected.
2. **Boundary quality** (|dS|) — Fiedler ~4x tighter on G11 but corr(|dS|, gain) wrong-signed (+0.48); inert. Rejected.
3. **beta_e** (Fiedler boundary fraction) — Fiedler-wins beta_e 0.020-0.044; FConn-wins 0.017-0.458. Ranges OVERLAP. Rejected as cross-family predictor.
4. **Mobility** (support stability) — Fiedler-wins mob 0.11-0.36; FConn-wins 0.07-1.00. Ranges OVERLAP. Confirmed mechanism on G11 only, NOT cross-family.
5. **Fragmentation** (disagreement-component structure) — Delaunay is MORE fragmented than G11 yet FConn wins; G13/G32 coupled yet Fiedler wins. Rejected.

**20-instance sweep (8 seeds, Mann-Whitney, fallback_count=0):** "beta_e separates winners? NO.
mobility separates winners? NO." Both confirmed by the data, not asserted.

**Convergent conclusion:** No single STATIC or DYNAMIC structural/spectral property predicts the
winner across instance families. This is a genuine negative result, not a failure to search.
(NOTE: torusg3-8 / torusg3-15 rows in that sweep were CORRUPTED — Gaussian-weighted instances
produced garbage cut values [~3.8e7]. Excluded. The +-1 toruspm instances were fine.)

## 3. The causal experiment (p-sweep) — KEY RESULT

Design: ONE fixed 20x40 periodic torus (n=800, deg 4, lambda2 = 0.0246 PINNED). Vary ONLY the
+-1 edge-sign disorder fraction (frac_neg). Generator: `make_signed` in rank1_disorder.py
(verified: copies fixed `base`, flips only signs). Torus re-asserted at runtime
(n=800, m=1600, 4-regular). 8 seeds/condition, fallback_count=0.

Result — the winner FLIPS with disorder while lambda2 is constant:

| frac_neg | winner        | Fiedler margin |
|----------|---------------|----------------|
| 0.00     | tie           | —              |
| 0.02     | tie           | —              |
| 0.05     | tie           | —              |
| 0.08     | Fiedler (p=.009) | grows        |
| 0.103    | Fiedler (p=.002) | (= p_c)      |
| 0.13-0.50| Fiedler (p<=.001)| grows monotonically |

**Interpretation:** lambda2 (and all topology) held EXACTLY constant, yet the winner moves.
Therefore the causal lever is **frustration/disorder density, NOT spectral structure**.
Transition sits at the 2D +-J critical concentration p_c ~ 0.103 (Hartmann et al.).

## 4. Verification (multi-pattern, Test 1+2) — PASSED

Re-ran across 3 independent disorder PATTERNS (sign_seed = 42, 7, 123) per density, 6 seeds each,
fallback_count=0:

- **Mobility is structurally pinned:** F_mob = 0.33-0.37, std <= 0.023 at EVERY density, invariant
  to disorder amount and pattern. Mobility is a property of topology, not frustration.
- **Winner-flip reproduces across patterns:** CONSISTENT at 0.0/0.05 (tie), 0.08/0.15/0.25/0.50
  (Fiedler). The ONLY inconsistent density is p_c=0.103 itself (Fiedler/Fiedler/tie) — pattern-
  sensitivity AT the critical point is expected (it is the phase transition), but is noted, not
  explained-away.

## 5. What this licenses (honest scope)

**PROVEN (toroidal family):**
- Frustration density is the causal driver of the selector winner (winner flips while lambda2 pinned).
- Below threshold: tie. Clearly above (>=0.08): Fiedler. Transition region ~p_c=0.103.
- Mobility is invariant to disorder => the winner-flip is **NOT driven by a change in mobility**.
- Mobility (Fiedler ~0.34 mobile, FConn ~0.80 sticky) is structural scaffolding, reproduced here.

**NOT PROVEN — do NOT write these:**
- "Disorder and mobility are INDEPENDENT" (factorial sense). Only shown: flip not *caused by*
  mobility change. Independence needs a mobility x disorder factorial (vary aspect ratio 20x40 /
  8x100 / 40x20 against frac_neg). UNTESTED.
- Sharp deterministic flip exactly at p_c — data shows a transition REGION with a fuzzy/pattern-
  sensitive critical point, not a step.
- Generality beyond toroidal topology — causal claim is for the torus family only.

## 6. Open items
1. Factorial mobility x disorder (aspect-ratio sweep) — to test true independence.
2. Characterise p_c: 8-10 patterns at frac_neg=0.103 only (~40 min) — is it ~67% Fiedler or noise?
3. Other topologies — does "frustration is the lever" hold off the torus?
4. Why low-beta_e splits across families (G11 Fiedler vs Delaunay FConn) — likely different
   frustration structure, but the cross-family causal test is unbuilt.

## 7. Methodological note (carried from this session)
- A second chat "confirming" the mobility finding was CIRCULAR — it had been shown our result;
  its agreement is an echo, discounted to zero. Only its independently-derived beta_e routing
  work counts, and that is a separate (also-imperfect) predictor, not confirmation of mobility.
- The contribution reframes FROM "spectral Fiedler selection achieves exact recovery via domain-
  wall alignment" (falsified) TO "frustration density drives selector advantage on toroidal
  instances; mobility is the structural enabler; no single observational predictor generalises
  across families." Arrived at by falsifying five mechanisms before the causal test.

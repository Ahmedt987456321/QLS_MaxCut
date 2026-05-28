# PhD Research Log — QLS Max-Cut Project
# Branch: phd-extensions
# Starting point: MSc complete, 4/4 G-set wins, p<=0.0001

---

## STARTING POSITION (from MSc)

AQLS-FrustratedConn-LA beats SA on all 4 G-set instances.
Best method: connected frustrated selector + look-ahead acceptance.
Key finding: optimal k scales inversely with lambda_2.
Novel contributions: connected frustrated selector, escape rate measurement, spectral k-scaling law.

---

## PhD RESEARCH QUESTIONS & CURRENT ANSWERS

### RQ1 — Does a cut polytope selector outperform connected frustrated?
**Status: Answered — No, on current implementation.**

**What we tried:**
Version 1 — triangle-based violation scoring using |gain| as proxy.
Result: median 510 vs connected frustrated 549 on G11 (p=0.002 — significantly worse).
Problem: used |gain| as proxy for odd-cycle violations — not the real mathematical object.

Version 2 — proper odd-cycle violations using actual vertex assignments x.
Added x parameter to all selectors. Computed true triangle violations.
Result: median 526 vs connected frustrated 549 on G11 (p=0.002 — still worse).
Problem: G11 is bipartite with NO triangles. Triangle-based scoring finds nothing meaningful.

Also tested on G1 (dense, many triangles).
Result: median 11425 vs connected frustrated 11537 (p=0.002 — worse).
Problem: max_triangles=1000 limit samples only a tiny fraction of G1's triangles.
Scattered high-violation vertices produce disconnected subproblems neal cannot solve well.

**Root cause:**
Cut polytope selector needs proper Barahona-Mahjoub odd-cycle separator (LP-based).
Triangle sampling is a poor approximation. Needs full polyhedral separation.

**Key theoretical finding:**
G11 is bipartite — no odd cycles in contractible sense.
Active facets on G11 are chordless 4-face cycle inequalities (6400 facets)
and non-contractible homological cycles (not triangles).
Frustrated selector was ALREADY approximating binding facets correctly.

**Future fix:**
Implement proper Barahona-Mahjoub separator.
Or use LP relaxation to find most violated odd-cycle inequality.
Estimated: 2-3 weeks additional work.

---

### RQ2 — Does the topological selector improve on G11?
**Status: Answered — Matches but does not beat connected frustrated.**

**What we tried:**
Implemented topological selector seeding from non-contractible loops.
G11 is 8-column x 100-row toroidal grid (confirmed by neighbour analysis).
8 meridian cycles (length 100) + 100 longitude cycles (length 8).
Selector finds most frustrated loop and seeds BFS from it.

Result: median 549 vs connected frustrated 547 (p=0.44 — not significant).

**Why no improvement:**
Research confirmed frustrated selector already implicitly finds binding facets.
Both methods access the same cut polytope information through different routes.
Topological seeding finds non-contractible loop frustration.
Frustrated selection finds face-cycle binding facets.
At k=400 both are capturing the same structural information.

**Key theoretical finding:**
Connected frustrated selector is an informal approximation of polyhedral active-facet selector.
Making this rigorous is a theoretical PhD contribution.
Galluccio-Loebl-Vondrák theorem: G11 (torus genus-1) is polynomially solvable.
Best known cut 564 is almost certainly the certified global optimum.

**Future direction:**
Topological seeding may help on 3D Ising instances (G55, G60)
where Galluccio-Loebl-Vondrák fails and polynomial algorithms don't exist.

---

### RQ3 — Does the hybrid topological selector improve on G11?
**Status: Answered — No improvement over individual selectors.**

**What we tried:**
Hybrid alternates between connected frustrated and topological seeding.
Every log(n)~10 iterations uses topological seed, rest uses connected frustrated.

Result: median 545 — statistically identical to both individual selectors (p~1.0).
All three beat SA significantly (p=0.002).

**Conclusion:**
All three selectors access the same structural information on G11.
Hybrid adds complexity without benefit on toroidal graphs.
May be useful on non-toroidal graphs where the two signals are independent.

---

### RQ4 — Does SBM beat neal as a subproblem solver?
**Status: Answered — No, on sparse subproblems. Neal is better.**

**What we tried:**

Attempt 1 — SBM with CPU PyTorch (wrong venv).
Result: 6.9 seconds per call vs neal 0.6 seconds. 11x SLOWER.
Problem: PyTorch installed in wrong venv (OneDrive vs C:\dev\QLS).

Attempt 2 — Fixed venv, GPU PyTorch cu124.
GPU detection: CUDA 13.1 driver requires cu124 build (not cu121).
Also needed NVIDIA Control Panel: set python.exe to high-performance GPU.
Result: SBM 0.612 seconds at steps=500 — similar speed to neal.

Attempt 3 — Benchmark SBM vs neal on G14.
Result: neal median 3031, SBM median 3009 (p=0.002 — SBM significantly worse).
ConvergenceWarning: "No agent has converged. Returned signs of final positions instead."

**Root cause — SBM failure on sparse subproblems:**
SBM designed for large dense all-to-all Ising problems.
Your connected frustrated subproblems at k=400 have average degree 1-4.
QUBO density approximately 0.25-0.5%.

Three failure modes:
1. Force starvation — low-degree nodes have too little coupling force to bifurcate.
   Equation: y_dot_i = -[harmonic] + c0 * sum_j J_ij * sign(x_j)
   With 1-4 neighbours, coupling force is tiny. Harmonic term dominates.
   Oscillators never commit to +1 or -1.

2. Mis-calibrated bifurcation point — default c0 formula assumes dense Wigner matrix.
   Sparse bounded-degree graphs have different spectral structure.
   Formula gives wrong c0, bifurcation point mistimed.

3. Swing nodes — sparse frustrated topologies produce oscillators that swing around zero.
   Proved in 2023 Nature Communications paper.

**ConvergenceWarning meaning:**
Not a single agent stabilised within convergence window (50 identical samples, 50 steps apart).
Package returns signs of still-oscillating positions.
For isolated vertices (degree 0 in induced subgraph): completely random output.

**Fix implemented — density routing:**
Added density check to backend_sbm:
edge_count = (J != 0).sum().item() // 2
avg_degree = (2 * edge_count) / max(n, 1)
if avg_degree < 5: return backend_neal(...)

Confirmed working: AQLS-SBM-routing matches AQLS-Neal (p=0.684).

**When SBM IS appropriate:**
Average degree >= 5 (density >= ~1%).
This requires either larger k or a dense-subgraph selector (e.g. Fiedler-guided).
On G1 with avg degree 48, induced subgraphs at k=160 are genuinely dense.

**Correct SBM hyperparameters for sparse subproblems (if needed):**
mode: discrete (dSB) not ballistic
heated: True (HdSB) — adds thermal noise for sparse graphs
time_step: 0.7-1.0 (not default 0.1)
max_steps: 20000-50000 (not 500)
agents: 512+ (not 200)
early_stopping: False

**SBM API notes (version 2.0.0):**
Mode strings: "ballistic" and "discrete" (NOT "bSB" and "dSB")
Required keyword: domain="spin"
Enum values: SimulatedBifurcationEngine.bSB and .dSB exist but passed as strings

---

### RQ5 — Does optimal k scale with inverse lambda_2? (confirmed from MSc)
**Status: Confirmed across 4 instances.**
See MSc log for full details.

---

## IMPLEMENTATIONS COMPLETED (PhD branch)

### New selectors in src/selectors.py

1. select_cut_polytope — odd-cycle violation based
   Uses actual vertex assignments x (passed via x= parameter)
   Computes triangle violations in induced subgraph
   Falls back to frustrated_connected when no triangles found
   Status: implemented, underperforms on G11 and G1
   Reason: needs proper LP-based separator not triangle sampling

2. select_topological — non-contractible loop seeding for toroidal graphs
   Detects toroidal grid structure via _detect_toroidal_grid()
   Scores meridian and longitude loops by total frustration
   Seeds BFS from most frustrated vertex in best loop
   Falls back to frustrated_connected for non-toroidal graphs
   Status: implemented, matches frustrated_connected quality

3. select_hybrid_topo — alternates frustrated and topological
   Every log(n) iterations: topological seed
   All other iterations: connected frustrated
   Status: implemented, identical quality to individual selectors

4. select_fiedler — NOT YET IMPLEMENTED
   Planned: score(v) = alpha * |gain(v)| + (1-alpha) / (eps + |v2(v)|)
   Novel: no prior paper combines Fiedler + gain for Max-Cut sub-QUBO
   Theoretical basis: Trevisan 2009 spectral Max-Cut bound

### Interface change — x parameter added to all selectors
All selectors now accept x=None as optional parameter.
adaptive_qls.py passes x=x to selector call.
Enables cut_polytope selector to compute true violations.

### New backend in src/backends.py

backend_sbm — Simulated Bifurcation Machine
PyTorch-based, GPU-accelerated when available
Density routing: avg_degree < 5 → fall back to neal
Mode selection: n>=300 → discrete, else ballistic
max_steps=500 (sweet spot: 0.612s, good convergence)
Fallback to neal on any exception

### GPU setup notes
CUDA 13.1 driver requires PyTorch cu124 build:
  C:\dev\QLS\venv\Scripts\python.exe -m pip install torch
  --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
Also set python.exe to high-performance GPU in NVIDIA Control Panel.

---

## EXPERIMENTS RUN (PhD branch)

| Experiment | File | Key result |
|---|---|---|
| Cut polytope v1 vs FConn G11 | phd_cut_polytope_vs_fconn_G11.json | CutPoly worse p=0.002 |
| Cut polytope v2 vs FConn G11 | phd_cut_polytope_v2_G11.json | CutPoly worse p=0.002 |
| Cut polytope v2 vs FConn G1 | phd_cut_polytope_G1.json | CutPoly worse p=0.002 |
| Topological vs FConn G11 | phd_topological_vs_fconn_G11.json | Not significant p=0.44 |
| Hybrid topo vs FConn G11 | phd_hybrid_topo_G11.json | Not significant p~1.0 |
| SBM vs Neal G14 (CPU) | phd_sbm_vs_neal_G14.json | SBM worse p=0.002 |
| SBM vs Neal G14 (GPU) | phd_sbm_vs_neal_G14_gpu.json | SBM worse p=0.002 |
| SBM routing vs Neal G14 | phd_sbm_routing_G14.json | Routing works p=0.684 |

---

## RESEARCH REPORTS COMPLETED (other chat)

1. Cut polytope theory — Barahona-Mahjoub odd-cycle separation
   Finding: no published Max-Cut heuristic uses facet violations as destroy rule — novel
   G11 specific: bipartite, no triangles, active facets are 4-face cycles and homological loops

2. G11 topology — Galluccio-Loebl-Vondrák theorem
   Finding: G11 (torus genus-1) polynomially solvable in 4 Pfaffian computations
   Implication: best known 564 is certified global optimum
   Implication: AQLS finding 556 is within 1.4% of provably optimal

3. SBM failure modes on sparse QUBOs
   Finding: force starvation, mis-calibrated bifurcation, swing nodes
   Solution: density routing — use neal for sparse, SBM for dense
   Future: TESB (Tabu-Enhanced SBM) or SBQA for sparse rugged regime

4. QAOA as subproblem solver
   Finding: NOT competitive in 2026 at k=80-400
   QAOA p=1 achieves 0.692 approximation ratio vs GW 0.878 classical
   Circuit depth for k=80 requires ~80,000 two-qubit gates, IBM budget ~5,000
   Exception: Pauli-Correlation Encoding (Sciorilli et al. 2025) on Quantinuum hardware
   Recommendation: small QAOA chapter at k=12-24 showing negative result

5. BLS implementation details
   Finding: no Python BLS exists anywhere — building it is community contribution
   G14: BLS hits best-known only 6/20 times (mean 3062.85)
   G22: BLS hits best-known only 1/20 times (mean 13344.45)
   AQLS already achieves G14 median 3036, G22 median 13198 with neal
   With better solver: predicted G14 mean >= 3063, G22 mean >= 13350
   This beats BLS — publication target

6. Fiedler-augmented selector theory
   Finding: novel — no prior paper combines Fiedler vector + gain for Max-Cut sub-QUBO
   score(v) = alpha * |gain(v)| + (1-alpha) / (eps + |v2(v)|)
   Frustration-weighted Laplacian: cut edges +1, uncut edges -1
   Trevisan 2009 provides theoretical justification
   Predicted gain: modest (+0-5 cuts) but more consistent

---

## REMAINING RESEARCH QUESTIONS

### Priority 1 — Implement and test
RQ-PhD-1: Does AQLS beat BLS on G14 and G22?
  Implementation: 300 lines Python from Benlic & Hao 2013 pseudocode
  Parameters: L0=0.01n, T=1000, tabu=rand(3,n//10)
  Target: beat BLS mean (3062.85 on G14, 13344 on G22)
  Timeline: 2 weeks

RQ-PhD-2: Does Fiedler-augmented selector improve on connected frustrated?
  Implementation: 50 lines in selectors.py
  score(v) = 0.6 * |gain(v)| + 0.4 / (0.005 + |v2(v)|)
  Timeline: 1 day + experiments

### Priority 2 — Extend coverage
RQ-PhD-3: Does spectral k-scaling law hold across G1-G54?
  Plot k vs lambda_2 for all 54 instances, fit curve
  Timeline: 1 week experiments + 2 days analysis

RQ-PhD-4: Does SBM beat neal on dense subproblems (G1)?
  Test on G1 where induced subgraphs avg degree ~48
  Density routing will allow SBM to run on G1 subproblems
  Timeline: 1 experiment run

### Priority 3 — Theoretical
RQ-PhD-5: Formal proof of k_escape = Omega(n / lambda_2)
  Data: G1 k=160 lambda_2~34, G11 k=400 lambda_2~0.05, G22 k=640
  Tools: Cheeger inequality + Cauchy interlacing theorem
  Timeline: 2-4 weeks

RQ-PhD-6: Formal proof that connected frustrated ~ binding facet detection on toroidal graphs
  Research confirmed informally — needs polyhedral theory + topology
  Timeline: 1-2 months

### Priority 4 — Future
RQ-PhD-7: QAOA advantage over neal at k=12-24?
  Small experiment chapter, expected negative result
  Library: Qiskit (has QUBO converters, QAOAAnsatz, WarmStartQAOAOptimizer)
  Timeline: 1 week

RQ-PhD-8: Topological selector on 3D Ising instances (G55, G60)?
  Galluccio-Loebl-Vondrák fails in 3D
  Non-contractible loops in 3D torus vs 2D torus
  Timeline: 2 weeks

---

## IMPORTANT TECHNICAL NOTES

### Two venv problem
C:\dev\QLS\venv — correct PhD venv
C:\Users\thotl\OneDrive\Desktop\QLS\venv — old MSc venv
pip notice always mentions OneDrive path but installs into dev path.
Always use: C:\dev\QLS\venv\Scripts\python.exe -m pip install ...
Verify with: python -c "import sys; print(sys.executable)"

### GPU setup
Driver: NVIDIA 592.27, CUDA 13.1
Requires PyTorch cu124 (NOT cu121, NOT cpu)
Install: C:\dev\QLS\venv\Scripts\python.exe -m pip install torch
         --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
Verify: python -c "import torch; print(torch.cuda.is_available())"
NVIDIA Control Panel: set python.exe to High-performance NVIDIA processor

### SBM API (version 2.0.0)
from simulated_bifurcation import maximize
mode strings: "ballistic" or "discrete" (NOT "bSB"/"dSB")
required: domain="spin"
sweet spot: max_steps=500, agents=200 → 0.612s on GPU
ConvergenceWarning means sparse subproblem — fall back to neal

### G11 grid structure
8 columns x 100 rows toroidal grid, nodes 1-800
node(r,c) = r*8 + c + 1 (0-indexed r,c)
Meridian cycles: 8 loops of length 100 (one per column)
Longitude cycles: 100 loops of length 8 (one per row)
Bipartite: no odd cycles, no triangles

### Selector interface
All selectors: selector(G, gc, k, pool=None, rng=None, x=None)
x=None is safe default — all selectors fall back gracefully
adaptive_qls.py passes x=x in selector call

---

## PUBLICATION ROADMAP

### Paper 1 (ready to write — MSc results)
Title: Adaptive Quantum Local Search with Connected Frustrated Neighbourhood
       Selection for Max-Cut
Venue: GECCO or EvoCOP
Content: 4 G-set instances, spectral k-scaling, escape rate measurement
Status: all experiments done, 30 trials each, p<=0.0001

### Paper 2 (next — needs BLS implementation)
Title: AQLS vs Breakout Local Search on G-set Max-Cut Benchmarks
Venue: Journal of Heuristics or Computers & Operations Research
Content: head-to-head AQLS vs BLS on G14/G22
Target: beat BLS mean on both instances
Status: needs BLS implementation (~2 weeks)

### Paper 3 (future — needs theory)
Title: Spectral Characterisation of Optimal Perturbation Size in
       Quantum Local Search for Max-Cut
Content: formal proof k_escape = Omega(n/lambda_2), Fiedler selector
Status: empirical data ready, theory not yet formalised

---

## HOW TO RESUME THIS PROJECT

1. Activate venv: venv\Scripts\activate
2. Confirm GPU: python -c "import torch; print(torch.cuda.is_available())"
3. Run tests: python -m pytest tests/ -v (expect 17 passed)
4. Check branch: git branch (should be phd-extensions)
5. Next action: implement BLS in src/baselines.py

### BLS implementation starting point
File: src/baselines.py
Function: breakout_local_search(G, budget_seconds, best_known=None, seed=None)
Key components:
  - Steepest descent local search (already have one_flip_ls)
  - Tabu list: tabu[v] = iteration + tenure, tenure = rand(3, n//10)
  - Perturbation operators M1 (best tabu-filtered flip) M2 (best swap) M3 (random)
  - Adaptive L: starts L0=n//100, increments on attractor revisit, resets on improvement
  - Attractor detection: exact configuration equality check O(n)
  - Stop: T=1000 non-improving optima before forced restart
Reference: Benlic & Hao 2013, Engineering Applications of AI 26(3):1162-1173

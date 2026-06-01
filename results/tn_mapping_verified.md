# TN backend ? verified QUBO->GTN mapping (2026-06-01)

Bridge: Python writes tn_in.json, Julia worker (solve_tn_worker.jl, tn_env
project) runs GTN SpinGlass + SingleConfigMax, writes tn_out.json.

VERIFIED mapping (matches backend_exact on G11 k=14, energy -14.0):
- QUBO (minimize) -> Ising J,h:
    diagonal Q_ii -> h[i] += Q_ii/2
    off-diag Q_ij -> J_ij += Q_ij/4; h[i] += Q_ij/4; h[j] += Q_ij/4
- Edges 1-indexed. Worker MUST re-align J to Graphs.edges(g) order
  (GTN sorts edges; insertion order != solve order -- this was the bug).
- SpinGlass(g; J=J, h=h), no negation; read res.c.data bits directly as x.
- Conversion proven exact over all 2^8 assignments (tn_check_map.py).

Files: solve_tn_worker.jl, tn_check_map.py, tn_calibrate.py (throwaway).
GTN 1.4.0 pinned in tn_env (newer versions pull LuxorGraphPlot -> OpenSSL
conflict with juliacall; subprocess bridge avoids in-process juliacall).

import argparse, time, statistics as stats
import numpy as np
from src.graph import load_gset
from src.local_search import compute_cut_value, random_cut, one_flip_ls
from src.qubo import build_local_qubo, qubo_energy
from src.gain_cache import GainCache
from src.backends import backend_neal, backend_exact

def connected_subset(G, k, rng):
    nodes = list(G.nodes())
    seed = nodes[rng.integers(0, len(nodes))]
    S, seen, frontier = [seed], {seed}, [seed]
    while len(S) < k and frontier:
        nxt = []
        for u in frontier:
            for w in G.neighbors(u):
                if w not in seen:
                    seen.add(w); S.append(w); nxt.append(w)
                    if len(S) >= k: break
            if len(S) >= k: break
        frontier = nxt
    return S

def sqa_solve(Q, S, trotter, num_sweeps, beta, num_reads):
    import openjij as oj, dimod
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for (i, j), v in Q.items():
        if i == j: bqm.add_variable(i, v)
        else: bqm.add_interaction(i, j, v)
    sampler = oj.SQASampler()
    resp = sampler.sample(bqm, num_reads=num_reads, trotter=trotter,
                          num_sweeps=num_sweeps, beta=beta)
    best = resp.first.sample
    return {v: best.get(v, 0) for v in S}

def time_call(fn):
    t0 = time.perf_counter(); out = fn(); return out, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gset", required=True)
    ap.add_argument("--k", type=int, default=160)
    ap.add_argument("--n_subqubos", type=int, default=20)
    ap.add_argument("--neal_reads", type=int, default=100)
    ap.add_argument("--sqa_reads", type=int, default=100)
    ap.add_argument("--trotter", type=int, nargs="+", default=[8,16,32])
    ap.add_argument("--sweeps", type=int, nargs="+", default=[1000])
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.gset} ...")
    G = load_gset(args.gset)
    print(f"  n={G.number_of_nodes()} m={G.number_of_edges()} k={args.k} subs={args.n_subqubos}")
    if args.exact and args.k > 18:
        print("  [warn] --exact ignored: k>18"); args.exact = False
    x = random_cut(G, rng=np.random.default_rng(args.seed))
    gc = GainCache(); x, gc, _, _ = one_flip_ls(G, x, gc)
    print(f"  incumbent cut: {compute_cut_value(G, x):.1f}")
    subs = []
    for _ in range(args.n_subqubos):
        S = connected_subset(G, args.k, rng)
        subs.append((S, build_local_qubo(G, x, S)))
    neal_e, neal_t = [], []
    for S, Q in subs:
        sol, dt = time_call(lambda: backend_neal(Q, S, n_reads=args.neal_reads))
        neal_e.append(qubo_energy(Q, sol, S)); neal_t.append(dt)
    exact_e = None
    if args.exact:
        exact_e = [qubo_energy(Q, backend_exact(Q, S), S) for S, Q in subs]
    sqa_res = {}
    for tr in args.trotter:
        for sw in args.sweeps:
            e, t, ok = [], [], True
            for S, Q in subs:
                try:
                    sol, dt = time_call(lambda: sqa_solve(Q, S, tr, sw, args.beta, args.sqa_reads))
                except Exception as ex:
                    print(f"  [SQA tr={tr} sw={sw}] failed: {ex}"); ok = False; break
                e.append(qubo_energy(Q, sol, S)); t.append(dt)
            if ok: sqa_res[(tr, sw)] = (e, t)
    med = stats.median
    print("\n" + "="*64)
    print("RESULTS (QUBO energy: LOWER = better)")
    print("="*64)
    print(f"{'solver':<28}{'med energy':>12}{'med time(s)':>14}")
    print("-"*64)
    print(f"{'neal SA':<28}{med(neal_e):>12.1f}{med(neal_t):>14.4f}")
    for (tr, sw), (e, t) in sorted(sqa_res.items()):
        print(f"{f'SQA tr={tr} sw={sw}':<28}{med(e):>12.1f}{med(t):>14.4f}")
    if exact_e is not None:
        print(f"{'exact (oracle)':<28}{med(exact_e):>12.1f}{'--':>14}")
    print("\n" + "-"*64)
    print("Paired vs neal (per sub-QUBO)")
    print("-"*64)
    for (tr, sw), (e, t) in sorted(sqa_res.items()):
        wins = sum(1 for a, b in zip(e, neal_e) if a < b)
        ties = sum(1 for a, b in zip(e, neal_e) if abs(a-b) < 1e-9)
        print(f"  SQA tr={tr} sw={sw}: SQA better on {wins}/{len(e)} (ties {ties})")
    if exact_e is not None:
        def hit(es): return sum(1 for a, b in zip(es, exact_e) if abs(a-b) < 1e-9)
        print("\n  exact hit-rate:")
        print(f"    neal: {hit(neal_e)}/{len(exact_e)}")
        for (tr, sw), (e, _) in sorted(sqa_res.items()):
            print(f"    SQA tr={tr} sw={sw}: {hit(e)}/{len(exact_e)}")

if __name__ == "__main__":
    main()

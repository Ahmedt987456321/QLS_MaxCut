"""Plain Parallel Tempering for Max-Cut ? dev/validation before porting."""
import time, numpy as np, statistics as stats
from src.gain_cache import GainCache
from src.local_search import compute_cut_value, random_cut
from src.metrics import Metrics

def parallel_tempering(G, budget_seconds, best_known=None, seed=None,
                       n_replicas=8, beta_min=0.1, beta_max=3.0,
                       swap_interval=10):
    """
    Plain Parallel Tempering (replica exchange) for Max-Cut.
    n_replicas at a geometric beta ladder; each does gain-weighted
    Metropolis flips; adjacent replicas swap on the PT criterion.
    """
    rng = np.random.default_rng(seed)
    metrics = Metrics(); metrics.start()
    nodes = list(G.nodes()); n = len(nodes)

    # geometric temperature ladder
    betas = np.geomspace(beta_min, beta_max, n_replicas)

    # one state per replica
    xs = [random_cut(G, rng) for _ in range(n_replicas)]
    gcs = [GainCache() for _ in range(n_replicas)]
    cuts = []
    for r in range(n_replicas):
        gcs[r].update(G, xs[r])
        cuts.append(compute_cut_value(G, xs[r]))

    x_best = dict(xs[0]); best_cut = max(cuts)
    bi = int(np.argmax(cuts)); x_best = dict(xs[bi])
    metrics.record_cut(best_cut)

    step = 0
    while time.time() - metrics.start_time < budget_seconds:
        step += 1
        # one Metropolis sweep-ish per replica (a batch of flips)
        for r in range(n_replicas):
            beta = betas[r]
            for _ in range(max(1, n // 10)):
                v = nodes[int(rng.integers(0, n))]
                delta = gcs[r].gain[v]   # gain = +cut change if flipped
                if delta >= 0 or rng.random() < np.exp(beta * delta):
                    xs[r][v] = 1 - xs[r][v]
                    gcs[r].incremental_update(G, xs[r], v)
                    cuts[r] += delta
                    if cuts[r] > best_cut:
                        best_cut = cuts[r]; x_best = dict(xs[r])
                        if best_known:
                            metrics.check_time_to_target(best_cut, best_known)

        # replica exchange on adjacent pairs
        if step % swap_interval == 0:
            for r in range(n_replicas - 1):
                d_beta = betas[r] - betas[r+1]
                d_cut = cuts[r] - cuts[r+1]
                # swap accept: exp((beta_r - beta_{r+1})(E_{r+1}-E_r));
                # for Max-Cut we maximise cut, so use d_beta * (cut_r - cut_{r+1}) sign
                arg = d_beta * (cuts[r+1] - cuts[r])
                if arg >= 0 or rng.random() < np.exp(arg):
                    xs[r], xs[r+1] = xs[r+1], xs[r]
                    gcs[r], gcs[r+1] = gcs[r+1], gcs[r]
                    cuts[r], cuts[r+1] = cuts[r+1], cuts[r]

        metrics.record_cut(best_cut)

    return x_best, metrics


if __name__ == "__main__":
    from src.graph import load_gset
    G = load_gset("data/gset/G11.txt")
    print("PT smoke test on G11 (30s, 5 trials):")
    cuts = []
    for t in range(5):
        x, m = parallel_tempering(G, budget_seconds=30, best_known=564, seed=t)
        # verify cut is valid
        assert all(x[v] in (0,1) for v in G.nodes())
        c = compute_cut_value(G, x)
        assert abs(c - m.best_cut) < 1e-6, f"cut mismatch {c} vs {m.best_cut}"
        cuts.append(c)
        print(f"  trial {t+1}: cut={c}")
    print(f"PT median={stats.median(cuts):.1f} max={max(cuts):.1f} (BKS=564)")

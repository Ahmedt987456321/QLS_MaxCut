"""Apply warm-start time cap fix to adaptive_qls.py.
Stops warm-start after 20% of budget to prevent consuming entire budget
on large dense graphs (e.g. SK complete graphs at n=400)."""
import re

with open("src/adaptive_qls.py", "r") as f:
    src = f.read()

old = """    # ?? pool warm-start: 5 diversified cuts ??????????????????????
    pool = []
    for i in range(5):
        x_init = random_cut(G, rng)
        gc_temp = GainCache()
        x_opt, gc_temp, _, _ = one_flip_ls(G, x_init, gc_temp)
        pool.append(dict(x_opt))"""

new = """    # ?? pool warm-start: 5 diversified cuts ??????????????????????
    # Cap warm-start at 20% of budget to handle large dense graphs
    pool = []
    warmstart_deadline = time.time() + 0.2 * budget_seconds
    for i in range(5):
        if time.time() > warmstart_deadline:
            break   # budget cap: stop early if warm-start is too slow
        x_init = random_cut(G, rng)
        gc_temp = GainCache()
        x_opt, gc_temp, _, _ = one_flip_ls(G, x_init, gc_temp)
        pool.append(dict(x_opt))
    if not pool:   # fallback: at least one random cut if all timed out
        x_init = random_cut(G, rng)
        pool.append(dict(x_init))"""

if old in src:
    src = src.replace(old, new)
    with open("src/adaptive_qls.py", "w") as f:
        f.write(src)
    print("Fix applied successfully.")
else:
    print("ERROR: Could not find the warm-start block to patch.")
    print("Apply manually: add time check inside the warm-start loop.")

from scipy import stats
import numpy as np

# two clearly-different paired samples -> tiny p-value
a = [536.0]*15 + [537.0]*15
b = [549.0]*15 + [550.0]*15
stat, p = stats.wilcoxon(a, b)

# this mirrors the exact logic now in experiment.py
p_str = "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"
print(f"raw p = {p}")
print(f"formatted -> {p_str}")
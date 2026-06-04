import json
from scipy import stats

fname = "results/G11_k400_final_30trials.json"
with open(fname) as f:
    data = json.load(f)

# show what's actually in the file
print("Methods in file:", list(data.keys()))
print()

# pick SA and the AQLS method automatically
sa_key = [k for k in data if k.upper() == "SA"][0]
aqls_key = [k for k in data if "FConn" in k or "FrustratedConn" in k][-1]
print(f"Comparing: {sa_key}  vs  {aqls_key}")

a = data[sa_key]["best_cut"]
b = data[aqls_key]["best_cut"]

w_stat, w_p = stats.wilcoxon(a, b)
u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")

print(f"n_trials: SA={len(a)}, AQLS={len(b)}")
print(f"Wilcoxon (paired):       p = {w_p:.2e}")
print(f"Mann-Whitney (unpaired): p = {u_p:.2e}")
print(f"Both significant (p<0.05)? {(w_p < 0.05) and (u_p < 0.05)}")
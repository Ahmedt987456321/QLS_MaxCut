with open("qaoa_dev.py", encoding="utf-8") as f:
    s = f.read()
s = s.replace("reps=2, maxiter=100", "reps=4, maxiter=300")
with open("qaoa_dev.py", "w", encoding="utf-8") as f:
    f.write(s)
print("bumped to reps=4 maxiter=300")

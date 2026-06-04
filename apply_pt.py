with open("src/baselines.py", encoding="utf-8") as f:
    src = f.read()
pt = open("pt_dev.py", encoding="utf-8").read()
# extract just the function (between the def and the __main__ block)
start = pt.index("def parallel_tempering")
end = pt.index('if __name__')
func = pt[start:end].rstrip() + "\n"
with open("src/baselines.py", "w", encoding="utf-8") as f:
    f.write(src.rstrip() + "\n\n\n# " + "-"*61 + "\n"
            + "# Parallel Tempering (replica exchange) - classical gold standard\n"
            + "# " + "-"*61 + "\n\n" + func)
print("parallel_tempering added to baselines.py")

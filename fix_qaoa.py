with open("qaoa_dev.py", encoding="utf-8") as f:
    s = f.read()
s = s.replace(".data.evs[0]", ".data.evs").replace("return est.run([(qc, H)]).result()[0].data.evs",
                                                     "return float(est.run([(qc, H)]).result()[0].data.evs)")
with open("qaoa_dev.py", "w", encoding="utf-8") as f:
    f.write(s)
print("fixed")

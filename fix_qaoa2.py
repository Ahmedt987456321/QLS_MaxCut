with open("qaoa_dev.py", encoding="utf-8") as f:
    s = f.read()
# replace "best = most frequent" with "best = min QUBO energy over sampled bitstrings"
old = '''    best_bits = max(counts, key=counts.get)
    bits = [int(b) for b in reversed(best_bits)]   # qiskit bit order
    return {S[i]: bits[i] for i in range(n)}'''
new = '''    from src.qubo import qubo_energy as _qe
    best_sol, best_e = None, float("inf")
    for bitstr in counts:
        bits = [int(b) for b in reversed(bitstr)]
        sol = {S[i]: bits[i] for i in range(n)}
        e = _qe(Q, sol, S)
        if e < best_e:
            best_e, best_sol = e, sol
    return best_sol'''
s = s.replace(old, new)
with open("qaoa_dev.py", "w", encoding="utf-8") as f:
    f.write(s)
print("fixed: return best-energy sampled bitstring")

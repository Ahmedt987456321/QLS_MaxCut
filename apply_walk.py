with open("src/adaptive_qls.py", encoding="utf-8") as f:
    s = f.read()

# 1. Add 'walk' handling in the acceptance block (after fixed_temp branch,
#    before the "else: improvement_only" comment)
old_accept = """        elif acceptance == 'fixed_temp':
            # Liu & Goan style ? fixed temperature Metropolis
            gamma = 1.0 / T_initial
            prob = float(np.exp(-gamma * abs(delta)))
            if rng.random() < prob:
                accepted = True
        # else: improvement_only ? accepted stays False"""
new_accept = """        elif acceptance == 'fixed_temp':
            # Liu & Goan style ? fixed temperature Metropolis
            gamma = 1.0 / T_initial
            prob = float(np.exp(-gamma * abs(delta)))
            if rng.random() < prob:
                accepted = True
        elif acceptance == 'walk':
            # BLS-style accept-and-walk: ALWAYS accept the (descended)
            # proposal as the new search origin, even if worse. Best is
            # tracked separately in Phase 1. No restart on non-improvement.
            accepted = True
        # else: improvement_only ? accepted stays False"""
s = s.replace(old_accept, new_accept)

# 2. In walk mode, the proposal must be descended first (use lookahead path)
old_la = "        if acceptance == 'lookahead':"
new_la = "        if acceptance in ('lookahead', 'walk'):"
s = s.replace(old_la, new_la)

# 3. In walk mode, never do the random-restart-on-rejection
old_restart = """        else:
            # pool persistence on restart ? keep top-5
            if ema_esc < 0.05:"""
new_restart = """        else:
            # pool persistence on restart ? keep top-5
            if acceptance != 'walk' and ema_esc < 0.05:"""
s = s.replace(old_restart, new_restart)

with open("src/adaptive_qls.py", "w", encoding="utf-8") as f:
    f.write(s)
print("added acceptance='walk' mode")

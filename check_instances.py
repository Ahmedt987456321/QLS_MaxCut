import os
for n in ["G1","G2","G11","G12","G13","G14","G22","G32","G33","G34"]:
    p = f"data/gset/{n}.txt"
    print(f"{n}: {'present' if os.path.exists(p) else 'MISSING'}")

# Show first 20 lines of each file to understand the format
import os
for name in ["torusg3-8.dat","toruspm3-8-50.dat",
             "torusg3-15.dat","toruspm3-15-50.dat"]:
    path = f"data/dimacs/{name}"
    print(f"\n{'='*50}")
    print(f"FILE: {name} ({os.path.getsize(path)} bytes)")
    print("First 20 lines:")
    with open(path) as f:
        for i,line in enumerate(f):
            if i >= 20: break
            print(f"  {repr(line.rstrip())}")

# Check what was actually downloaded
import os
for name in ["delaunay_n10", "delaunay_n11"]:
    path = f"data/delaunay/{name}.tar.gz"
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(100)
    print(f"{name}: {size} bytes")
    print(f"  header: {header[:50]}")

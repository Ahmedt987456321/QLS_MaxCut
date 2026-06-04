# Show first 20 lines of each mtx file
for name in ["delaunay_n10","delaunay_n11"]:
    path = f"data/delaunay/{name}.mtx"
    print(f"\n{name}:")
    with open(path) as f:
        for i,line in enumerate(f):
            if i>=20: break
            print(f"  {repr(line.rstrip())}")

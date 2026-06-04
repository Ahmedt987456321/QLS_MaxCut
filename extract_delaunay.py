import tarfile, os, gzip, shutil

for name in ["delaunay_n10", "delaunay_n11"]:
    path = f"data/delaunay/{name}.tar.gz"
    out_dir = f"data/delaunay/{name}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(out_dir)
        print(f"{name}: extracted to {out_dir}")
        for f in os.listdir(out_dir):
            print(f"  {f}: {os.path.getsize(os.path.join(out_dir,f))} bytes")
    except Exception as e:
        print(f"{name}: tar failed ({e}), trying direct gzip...")
        try:
            out = f"data/delaunay/{name}.mtx"
            with gzip.open(path,"rb") as fin, open(out,"wb") as fout:
                shutil.copyfileobj(fin,fout)
            print(f"  extracted to {out}: {os.path.getsize(out)} bytes")
            with open(out) as f:
                print(f"  first 3 lines: {[next(f) for _ in range(3)]}")
        except Exception as e2:
            print(f"  also failed: {e2}")

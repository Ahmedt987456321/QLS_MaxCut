import gzip, shutil, os, glob

files = glob.glob("data/dimacs/*.dat.gz")
for f in files:
    out = f.replace(".gz", "")
    print(f"Extracting {f} -> {out}")
    with gzip.open(f, 'rb') as fin, open(out, 'wb') as fout:
        shutil.copyfileobj(fin, fout)
    print(f"  OK: {os.path.getsize(out)} bytes")

print("\nFiles in data/dimacs/:")
for f in sorted(os.listdir("data/dimacs")):
    print(f"  {f}")

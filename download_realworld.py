"""Download small real-world graphs from SNAP for Max-Cut benchmarking.
Using graphs with known structure and manageable size (n=500-3000)."""
import urllib.request, os, gzip, shutil

# Small real-world graphs from SNAP (Stanford Network Analysis Project)
# These are standard in the network science literature
GRAPHS = {
    # Social networks
    "karate":     "https://snap.stanford.edu/data/karate.txt.gz",
    "dolphins":   "https://snap.stanford.edu/data/dolphins.txt.gz",
    # Collaboration networks  
    "ca-GrQc":    "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    # Power grid
    "power":      "https://snap.stanford.edu/data/USpowergrid_n4941.txt.gz",
}

os.makedirs("data/realworld", exist_ok=True)

for name, url in GRAPHS.items():
    dest = f"data/realworld/{name}.txt.gz"
    out  = f"data/realworld/{name}.txt"
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        with gzip.open(dest,"rb") as fin, open(out,"wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"  OK: {os.path.getsize(out)} bytes")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nFiles in data/realworld/:")
for f in sorted(os.listdir("data/realworld")):
    print(f"  {f}")

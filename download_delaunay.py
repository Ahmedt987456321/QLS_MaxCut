"""Download DIMACS10 Delaunay instances from SuiteSparse.
These are planar triangulations that straddle the beta* = 0.05 threshold.
delaunay_n10 (n=1024): predicted beta_e ~ 0.062 -> FConn
delaunay_n11 (n=2048): predicted beta_e ~ 0.042 -> Fiedler
This is the single most important threshold test."""
import urllib.request, os, gzip, shutil

os.makedirs("data/delaunay", exist_ok=True)

urls = [
    ("delaunay_n10", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n10.tar.gz"),
    ("delaunay_n11", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n11.tar.gz"),
]

for name, url in urls:
    dest = f"data/delaunay/{name}.tar.gz"
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {os.path.getsize(dest)} bytes")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nAlternative: download manually from")
print("  https://sparse.tamu.edu/DIMACS10/delaunay_n10")
print("  https://sparse.tamu.edu/DIMACS10/delaunay_n11")
print("  Click 'Matrix Market' format (.mtx.gz)")
print("  Save to data/delaunay/")

import json, glob, os

files = glob.glob("results/phd_final_30trials_*.json")
print("Found files:", [os.path.basename(f) for f in files])
if files:
    with open(files[0]) as f:
        data = json.load(f)
    print("\nTop-level keys:", list(data.keys())[:20])
    # walk one level to see structure
    def peek(d, depth=0, maxd=3):
        if depth > maxd: return
        if isinstance(d, dict):
            for k in list(d.keys())[:6]:
                print("  "*depth + f"{k}: {type(d[k]).__name__}")
                peek(d[k], depth+1, maxd)
        elif isinstance(d, list) and d:
            print("  "*depth + f"[list len {len(d)}] first elem: {type(d[0]).__name__}")
            peek(d[0], depth+1, maxd)
    peek(data)

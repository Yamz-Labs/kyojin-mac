"""Node-type census of the reuse layer (both - full) from layer_cpu.py --json files: usage types_diff.py A.json B.json [C.json ...]"""
import json, sys
def reuse(f):
    t = json.load(open(f))["types"]
    d = {k: t["both"].get(k, 0) - t["full"].get(k, 0) for k in set(t["both"]) | set(t["full"])}
    return {k: v for k, v in d.items() if v}
rs = [reuse(f) for f in sys.argv[1:]]
keys = sorted(set().union(*rs), key=lambda k: -abs(rs[-1].get(k, 0) - rs[0].get(k, 0)))
for k in keys:
    print(f"{k[:70]:70s} " + " ".join(f"{r.get(k, 0):6.1f}" for r in rs))

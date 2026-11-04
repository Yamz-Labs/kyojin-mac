"""Summarise the tensor layout of an EXL3 pack from the safetensors HEADERS only (no tensor data is read).
usage: pack_summary.py PACK_DIR [max_files=all]
Groups tensor names by pattern (digits -> N): count, dtype, example shape; for .trellis tensors the K (bits) histogram."""
import sys, os, json, struct, re, glob, collections
d = sys.argv[1]; mx_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
print("dir", d, "files", len(files), "total GiB", round(sum(os.path.getsize(f) for f in files) / 2**30, 1))
cfg = os.path.join(d, "config.json")
if os.path.exists(cfg):
    c = json.load(open(cfg)); qc = c.get("quantization_config", {})
    print("quantization_config", {k: v for k, v in qc.items() if not isinstance(v, (dict, list)) or len(str(v)) < 200})
    print("config keys", [k for k in c if k != "quantization_config"][:40])
pat = collections.OrderedDict()
tot_bytes = collections.Counter()
for f in files[:mx_files]:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        p = re.sub(r"\d+", "N", k)
        e = pat.setdefault(p, {"n": 0, "dtype": set(), "shape": v["shape"], "ex": k, "K": collections.Counter(), "files": set()})
        e["n"] += 1; e["dtype"].add(v["dtype"]); e["files"].add(os.path.basename(f))
        sz = v["data_offsets"][1] - v["data_offsets"][0]; tot_bytes[p] += sz
        if k.endswith(".trellis"): e["K"][v["shape"][-1] // 16] += 1
for p, e in pat.items():
    print(f"{p:90s} x{e['n']:5d} {','.join(sorted(e['dtype'])):8s} {str(e['shape']):28s} {tot_bytes[p]/2**20:9.1f} MiB" + (f"  K={dict(sorted(e['K'].items()))}" if e["K"] else ""))
print("example names:", [e["ex"] for e in list(pat.values())[:3]])

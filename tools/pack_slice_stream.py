#!/usr/bin/env python3
"""Stream a few tensors of a pack (box) as ONE safetensors file to stdout, nothing is written on the box disk.
usage: python tools/pack_slice_stream.py PACK_DIR 'layers.1.engram.*' 'layers.14.engram.*' | ssh mac 'cat > ~/kyojin-mlx/results_mac/x/slice.safetensors'
Names are fnmatch patterns on the pack index (model.safetensors.index.json, or the single model.safetensors). Tensor bytes are copied unchanged."""
import fnmatch
import json
import os
import struct
import sys

pack = sys.argv[1]
pats = sys.argv[2:]
ixp = os.path.join(pack, "model.safetensors.index.json")
wmap = json.load(open(ixp))["weight_map"] if os.path.exists(ixp) else None
files = sorted(set(wmap.values())) if wmap else ["model.safetensors"]
ents = []                                                                  # (name, file, abs offset, nbytes, header entry)
for f in files:
    with open(os.path.join(pack, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    for name, e in hdr.items():
        if (wmap is None or wmap.get(name) == f) and any(fnmatch.fnmatch(name, p) for p in pats):
            lo, hi = e["data_offsets"]
            ents.append((name, f, 8 + n + lo, hi - lo, e))
if not ents:
    sys.exit("pack_slice_stream: no tensor matches " + " ".join(pats))
ents.sort(key=lambda t: t[0])
off, hdr_out = 0, {}
for name, f, a, nb, e in ents:
    hdr_out[name] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + nb]}
    off += nb
hb = json.dumps(hdr_out, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
out = sys.stdout.buffer
out.write(struct.pack("<Q", len(hb))); out.write(hb)
for name, f, a, nb, e in ents:
    with open(os.path.join(pack, f), "rb") as fh:
        fh.seek(a)
        left = nb
        while left:
            d = fh.read(min(left, 1 << 24))
            out.write(d); left -= len(d)
out.flush()
print(f"pack_slice_stream: {len(ents)} tensors, {off / 2 ** 20:.1f} MiB", file=sys.stderr)

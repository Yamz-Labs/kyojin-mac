"""Slice real tensors out of an EXL3 mul1 pack into data/real_slices.npz (run once, on the Linux box).
usage: dump_pack_slices.py PACK_DIR OUT.npz   picks the smallest linear for each K in 2..5 and keeps a
512x512 (or smaller) tile-aligned block with the matching suh / svh slices (Hadamard blocks are independent)."""
import json, re, struct, sys, glob
import numpy as np
from safetensors import safe_open

pack, out = sys.argv[1], sys.argv[2]
best = {}
for f in sorted(glob.glob(pack + "/*.safetensors")):
    n = struct.unpack("<Q", open(f, "rb").read(8))[0]
    hdr = json.loads(open(f, "rb").read(8 + n)[8:])
    for k, v in hdr.items():
        if k.endswith(".trellis"):
            ti, tj, w = v["shape"]
            K = w // 16
            if K in (2, 3, 4, 5) and K not in best and ti >= 32 and tj >= 32:
                best[K] = (f, k[:-8])
    if len(best) == 4:
        break
res = {}
for K, (f, key) in best.items():
    with safe_open(f, "np") as s:
        assert key + ".mul1" in s.keys(), f"{key} is not a mul1 tensor (mcg/3inst pack?)"
for K, (f, key) in best.items():
    with safe_open(f, "np") as s:
        t, su, sv = s.get_tensor(key + ".trellis"), s.get_tensor(key + ".suh"), s.get_tensor(key + ".svh")
    res[f"K{K}_trellis"] = t[:32, :32]
    res[f"K{K}_suh"] = su[:512].view(np.uint16)
    res[f"K{K}_svh"] = sv[:512].view(np.uint16)
    print(K, key.split("layers.")[-1], t.shape, "->", res[f"K{K}_trellis"].shape)
np.savez_compressed(out, **res)

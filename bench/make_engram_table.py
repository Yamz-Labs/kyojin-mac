#!/usr/bin/env python3
"""Create ONE real-size, fully allocated Engram shard for gather benchmarks (sparse holes read back as zeros with no I/O, which would flatter any
gather number): same safetensors header as the official model-00047 shard (layer 1 table: F8_E4M3 [384006168, 256] + F8_E8M0 [384006168, 8],
about 94.4 GiB), table region filled with non-zero pseudo-random bytes, the small q / k / wkv region left as a hole. Random data is fine for I/O
timing; values are meaningless.
usage: python bench/make_engram_table.py OUT_DIR SLICE.npz [--shard model-00047-of-00048.safetensors]
Writes OUT_DIR/<shard> and OUT_DIR/model.safetensors.index.json; checks st_blocks against the size. Delete OUT_DIR when done."""
import argparse
import json
import os
import struct
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("slice")
ap.add_argument("--shard", default="model-00047-of-00048.safetensors")
a = ap.parse_args()
z = np.load(a.slice, allow_pickle=False)
hb = z["hdr_" + a.shard].tobytes()
n = struct.unpack("<Q", hb[:8])[0]
hdr = json.loads(hb[8:8 + n]); hdr.pop("__metadata__", None)
embed = {k: v for k, v in hdr.items() if ".engram.embed." in k}
fill_end = max(v["data_offsets"][1] for v in embed.values())
total = 8 + n + max(v["data_offsets"][1] for v in hdr.values())
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, a.shard)
rng = np.random.default_rng(5)
blk = rng.integers(1, 255, size=1 << 28, dtype=np.uint8)            # 256 MiB of non-zero bytes, perturbed per block
t0 = time.time()
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, hb)
off = 0
i = 0
while off < fill_end:
    m = min(len(blk), fill_end - off)
    blk[::4096] = (i * 37 + 1) % 255 + 1
    os.pwrite(fd, memoryview(blk)[:m], 8 + n + off)
    off += m; i += 1
    if i % 40 == 0:
        print(f"{off / 2 ** 30:.0f}/{fill_end / 2 ** 30:.0f} GiB {time.time() - t0:.0f}s", flush=True)
os.ftruncate(fd, total)
os.fsync(fd)
os.close(fd)
st = os.stat(path)
print(f"wrote {path}: size {st.st_size / 2 ** 30:.1f} GiB, allocated {st.st_blocks * 512 / 2 ** 30:.1f} GiB, {time.time() - t0:.0f}s")
assert st.st_blocks * 512 >= fill_end * 0.99, "file is not fully allocated"
w = {k: a.shard for k in embed}
tot = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in embed.values())
json.dump({"metadata": {"total_size": tot}, "weight_map": w}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)

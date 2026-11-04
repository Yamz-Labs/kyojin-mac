#!/usr/bin/env python3
"""Tile config sweep of the fused EXL3 + NAX kernel on real DS4.1 layer-3 experts: ms per 8-expert launch for rows/expert M x cfg (bmu, nmb, bns, bks).
Interleaved reps (the Air is shared). Used to fit the cost model of metal_nax.pick_cfg.
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/sweep_nax_cfg.py [--proj w1|w2] [--rows 1,8,64,256] [--cfgs ...]"""
import argparse, os, time
import numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX
from nax_common import mini_stack, kperm_cols
import glob
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--proj", default="w1"); ap.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512"); ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--cfgs", default="16,1,32,32;16,1,32,64;32,1,32,32;32,1,32,64;32,2,32,32;32,2,32,64;64,1,32,32;64,1,32,64")
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
f = sorted(glob.glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
with safe_open(f, "np") as s:
    k = [x for x in s.keys() if x.endswith(f".experts.{a.proj}.k")][0]; kv = s.get_tensor(k)
exps = [i for i in range(len(kv)) if kv[i] == kv[0]][:a.experts]
st = mini_stack(a.dir, a.proj, exps)
G = len(exps); KD, N = st.kin, st.nout
eids = mx.array(np.arange(G, dtype=np.int32))
def parse(c):
    """'bmu,nmb,bns,bks[,ilp][/mode]' -> (bmu, nmb, bns, bks[, ilp], mode)"""
    body, _, mode = c.partition("/")
    return tuple(int(v) for v in body.split(",")) + (mode or "full",)
cfgs = [parse(c) for c in a.cfgs.split(";")]
def run(A, rowoff, G, M, c):
    return NX.expert_gemm((st,), eids, rowoff, A, G * M, M, c[:-1], True, c[-1])
print(f"proj {a.proj} {KD}x{N} K{st.ks} experts {G}; ms per launch (TFLOPS)")
print("   M  " + " ".join(f"{str(c[:-1]) + ('/' + c[-1] if c[-1] != 'full' else ''):>24s}" for c in cfgs))
rng = np.random.default_rng(0)
for M in [int(v) for v in a.rows.split(",")]:
    A = mx.array(kperm_cols(rng.standard_normal((G * M, KD)).astype(np.float16)))
    rowoff = mx.array((np.arange(G + 1) * M).astype(np.uint32))
    fl = 2.0 * G * M * KD * N
    best = {c: 1e9 for c in cfgs}
    ok = {c: True for c in cfgs}
    for c in cfgs:
        try:
            mx.eval(run(A, rowoff, G, M, c))
        except Exception as e:
            ok[c] = False; print("fail", c, str(e)[:300])
    ref = None
    for c in cfgs:
        if ok[c]:
            y = np.array(run(A, rowoff, G, M, c)).astype(np.float32)
            ref = y if ref is None else ref
            if c[-1] == "full" and not np.array_equal(y, ref):
                print("  differs", c, "rel", float(np.linalg.norm(y - ref) / np.linalg.norm(ref)))
    for _ in range(a.reps):
        for c in cfgs:
            if not ok[c]:
                continue
            t0 = time.perf_counter(); mx.eval(run(A, rowoff, G, M, c)); best[c] = min(best[c], time.perf_counter() - t0)
    print(f"{M:5d}  " + " ".join((f"{best[c]*1e3:12.3f} ({fl/best[c]/1e12:5.2f})" if ok[c] else f"{'fail':>22s}") for c in cfgs), flush=True)
print(f"peak {mx.get_peak_memory()/1e9:.2f} GB")

#!/usr/bin/env python3
"""S3 crossover on this GPU: one DS4.1-sized MoE call of T rows through the chunked decode path (patch._moe_rows below pf_min_rows: MoeGather calls of
`step` rows) vs the S3 prefill path (MoeGather.prefill). 384 virtual experts (pool of 48 synth experts, K 2 / 3 mixed like the pack), gate/up 5120x2304,
down 2304x5120, top-6 uniform routing, sustained clocks (burn-in, interleaved reps, median).

usage: flock /tmp/ultra_heavy.lock python tools/pfcross.py [--ts 32,64,96,128,160,192,256] [--step 42] [--dims 6144,2048,256,8 (GLM-5.3)]
"""
import argparse, os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import moe as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default="32,64,96,128,160,192,256")
ap.add_argument("--step", type=int, default=0, help="rows per decode call (0 = MAX_PAIRS // top-k, rows.enable / MTP serving: 42 on DS4.1)")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--dims", default="5120,2304,384,6", help="hidden, expert inner, experts, top-k (default DS4.1)")
a = ap.parse_args()
mx.set_cache_limit(16 << 30)

H, I, E, TOPK = (int(v) for v in a.dims.split(","))
POOL = 48
a.step = a.step or M.MAX_PAIRS // TOPK
KS = [2 + (i % 2) for i in range(POOL)]
g0, u0, d0 = synth_stack(H, I, KS, "mul1", 1), synth_stack(H, I, KS, "mul1", 2), synth_stack(I, H, KS, "mul1", 3)


def virtual(st, V):
    phys = np.arange(V) % st.experts
    off = np.array(st.offset)[:st.experts][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]), mx.array(np.array(st.suh)[phys]),
                    mx.array(np.array(st.svh)[phys]), st.kt, st.nt, st.cb)


moe = M.MoeGather(virtual(g0, E), virtual(u0, E), virtual(d0, E), TOPK, limit=10.0, wpre=True)
rng = np.random.default_rng(0)


def chunked(x, ids, w):
    s = a.step
    return mx.concatenate([moe(x[i:i + s], ids[i:i + s], w[i:i + s]) for i in range(0, x.shape[0], s)], axis=0)


def s3(x, ids, w):
    return moe.prefill(x, ids, w)


for T in (int(v) for v in a.ts.split(",")):
    x = mx.array(rng.standard_normal((T, H)).astype(np.float16) * 0.1)
    ids = mx.array(np.stack([rng.choice(E, TOPK, replace=False) for _ in range(T)]).astype(np.int32))
    w = mx.array(rng.random((T, TOPK)).astype(np.float32) / TOPK)
    mx.eval(x, ids, w)
    ya, yb = np.array(chunked(x, ids, w)).astype(np.float32), np.array(s3(x, ids, w)).astype(np.float32)
    rel = np.linalg.norm(ya - yb) / np.linalg.norm(ya)
    for _ in range(3):
        mx.eval(chunked(x, ids, w)); mx.eval(s3(x, ids, w))
    ts = {"chunk": [], "s3": []}
    for _ in range(a.reps):
        for name, f in (("chunk", chunked), ("s3", s3)):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(f(x, ids, w)); ts[name].append(time.perf_counter() - t0)
    mc, ms = (float(np.median(ts[k])) * 1e3 for k in ("chunk", "s3"))
    print(f"T={T:4d} rows/expert {T * TOPK / E:5.2f}: chunk(step {a.step}) {mc:7.2f} ms  s3 {ms:7.2f} ms  x{mc / ms:4.2f}  rel {rel:.1e}", flush=True)
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)

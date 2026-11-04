#!/usr/bin/env python3
"""Microbench the attention core (kv_ring + attend), rope, grouped wo_a and hc mix on real DS4.1 shapes (bf16/fp16).
Slope of chains, median of reps, under flock. usage: ds41_opbench2.py [--reps 7] [--json OUT]
"""
import os
import argparse, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--chain", type=int, default=32)
ap.add_argument("--json", default="")
a = ap.parse_args()
rng = np.random.default_rng(1)

import importlib
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
from kyojin_mlx import fuse_attn as FA


def batch(thunk, k=200, reps=5):
    """ms/call for tiny ops: one eval of k distinct calls, min over reps."""
    best = None
    for _ in range(reps + 1):
        outs = [thunk() for _ in range(k)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return 1e3 * best / k


def slope(thunk):
    def run(m):
        outs = [thunk() for _ in range(m)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0
    run(a.chain)
    vals = []
    for _ in range(a.reps + 1):
        t1, t4 = run(a.chain), run(4 * a.chain)
        vals.append((t4 - t1) / (3 * a.chain))
    return 1e3 * min(vals[1:])


res = []

# rope tables for position p (shared across layers, built once per position)
def rope_pair():
    from mlx_vlm.models.deepseek_v41.language import DeepseekV4RoPE
    rope = DeepseekV4RoPE(64, 160000.0, None, 1048576)
    rt, it = FA.rope_tables(rope, 512)
    return rt, it, rope

# kv_ring + attend at real shapes: H=64, D=512, W=128, K=4 pooled
def core_thunk():
    q = mx.array((rng.standard_normal((1, 1, 64 * 512)) * 0.1).astype(np.float16)).astype(mx.float16)
    ring = mx.array((rng.standard_normal((1, 128, 512)) * 0.1).astype(np.float16)).astype(mx.float16)
    pool = mx.array((rng.standard_normal((1, 4, 512)) * 0.1).astype(np.float16)).astype(mx.float16)
    idx = mx.array(np.array([0, 1, 2, 3], dtype=np.int32))
    sink = mx.array(rng.standard_normal(64).astype(np.float32))
    rt = mx.array(rng.standard_normal(64).astype(np.float32))
    it = mx.array(rng.standard_normal(64).astype(np.float32))
    pos = mx.array([512], dtype=mx.int32)
    kv = mx.array((rng.standard_normal((1, 1, 512)) * 0.1).astype(np.float16)).astype(mx.float16)
    mx.eval(q, ring, pool, idx, sink, rt, it, pos, kv)
    def thunk():
        nr = FA.kv_ring(kv, ring, rt, pos, 64)
        return FA.attend(q, nr, pool, idx, sink, rt, it, pos, 64, 64, 64 ** -0.5)
    return thunk
t = slope(core_thunk())
print(f"attn_core kv_ring+attend: {t:.4f} ms/call x38/step", flush=True)
res.append(dict(op="attn_core", ms=round(t, 4), per_step=38))

# rope 1-row q (64 heads x 512, tail 64)
def rope_thunk():
    qq = mx.array((rng.standard_normal((1, 64, 1, 512)) * 0.3).astype(np.float16)).astype(mx.float16)
    from mlx_vlm.models.deepseek_v41.language import DeepseekV4RoPE
    rope = DeepseekV4RoPE(64, 160000.0, None, 1048576)
    def thunk():
        return rope(qq, 512)
    return thunk
t = batch(rope_thunk())
print(f"rope_q_64x512: {t:.4f} ms/call x40/step", flush=True)
res.append(dict(op="rope_q", ms=round(t, 4), per_step=40))

# wo_a grouped: 8 groups; bench one group GEMV 128x512 EXL3 and report x8 weights (grouped kernel does all in 3 launches)
from kyojin_mlx.exl3_linear import Exl3Linear
def woa_thunk():
    kin, nout, K = 512, 128, 3
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K)).astype(np.int16)
    su = rng.uniform(0.5, 1.5, kin).astype(np.float16)
    sv = rng.uniform(0.5, 1.5, nout).astype(np.float16)
    lin = Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    mx.eval(lin.parameters())
    nbytes = int(mx.array(tr).nbytes + mx.array(su).nbytes + mx.array(sv).nbytes)
    def thunk():
        return lin(mx.array((rng.standard_normal((1, kin)) * 0.3).astype(np.float16)).astype(mx.float16))
    return thunk, nbytes
th, nb = woa_thunk()
t = batch(th)
print(f"wo_a_per_group_128x512: {t:.4f} ms/call bytes {nb/1e6:.3f}MB (grouped kernel: 3 launches for all 8) x40/step", flush=True)
res.append(dict(op="wo_a_group", ms=round(t, 4), per_step=40))

# hc mix coeffs (stock compiled)
def mix_thunk():
    x = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16)).astype(mx.float16)
    fn = mx.array((rng.standard_normal((24, 20480)) * 0.05).astype(np.float32))
    sc = mx.array(np.ones(3, dtype=np.float32))
    bs = mx.array((rng.standard_normal(24) * 0.05).astype(np.float32))
    def thunk():
        return L.hc_mix_coeffs(x, fn, sc, bs, 4, 20, 1e-6, 1e-6)
    return thunk
t = slope(mix_thunk())
print(f"hc_mix_coeffs: {t:.4f} ms/call x80/step", flush=True)
res.append(dict(op="hc_mix", ms=round(t, 4), per_step=80))

if a.json:
    import json
    json.dump(res, open(a.json, "w"), indent=1)

#!/usr/bin/env python3
"""Per-op-class us/step microbench (slope of chains, fresh inputs, median of reps) on representative DS4.1 decode shapes.

Dense GEMVs use synthetic EXL3 trellis (same Mul1Gemv path as the real pack; bytes read from trellis nbytes, roof at --bw).
Elementwise/norm/rope/custom kernels are pack-independent (bf16).
usage: ds41_opbench.py [--reps 7] [--bw 1140] [--json OUT]
"""
import os
import argparse, statistics, sys, time

sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--bw", type=float, default=1140.0)
ap.add_argument("--chain", type=int, default=8)
ap.add_argument("--json", default="")
a = ap.parse_args()
rng = np.random.default_rng(0)


def slope(thunk, n_inputs=8):
    """ms/call = slope of eval(chain) vs 4x chain; min over reps. Thunk builds one call with fresh inputs."""
    def run(m):
        outs = [thunk() for _ in range(m)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0
    run(n_inputs)  # warm (kernel builds)
    vals = []
    for _ in range(a.reps + 1):
        t1, t4 = run(a.chain), run(4 * a.chain)
        vals.append((t4 - t1) / (3 * a.chain))
    vals = vals[1:]
    return 1e3 * min(vals)


def exl3_gemv(kin, nout, K=2, dtype=mx.float16, pool=8):
    from kyojin_mlx.exl3_linear import Exl3Linear
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K)).astype(np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    lins = [Exl3Linear(mx.array(tr) + 0 * c, mx.array(suh), mx.array(svh), cb="mul1", out_dtype=dtype) for c in range(pool)]
    lin = lins[0]
    nbytes = int(mx.array(tr).nbytes + mx.array(suh).nbytes + mx.array(svh).nbytes)
    mx.eval([l.parameters() for l in lins])
    return lins, nbytes


def bench_exl3(name, kin, nout, K=2, rows=1, count=1, dtype=mx.float16):
    lins, nbytes = exl3_gemv(kin, nout, K, dtype)
    it = [0]
    def call():
        it[0] += 1
        lin = lins[it[0] % len(lins)]
        return lin(mx.array((rng.standard_normal((rows, kin)) * 0.3).astype(np.float16)).astype(dtype))
    ms = slope(call)
    roof = nbytes / (a.bw * 1e9) * 1e3
    print(f"{name:28s} {ms:8.4f} ms/call  bytes {nbytes/1e6:7.2f} MB roof {roof:7.4f} ms x{count}/step", flush=True)
    return dict(op=name, ms=round(ms, 4), mb=round(nbytes / 1e6, 2), roof_ms=round(roof, 4), per_step=count)


def bench_dense(name, kin, nout, count=1, dtype=mx.bfloat16):
    import mlx.nn as nn
    lin = nn.Linear(kin, nout, bias=False)
    ws = []
    for _ in range(8):
        w = (rng.standard_normal((nout, kin)) * (kin ** -0.5)).astype(np.float16 if dtype != mx.float32 else np.float32)
        ws.append(mx.array(w).astype(dtype))
    mx.eval(*ws)
    it = [0]
    lin._pool, lin._it = ws, it
    nbytes = nout * kin * (4 if dtype == mx.float32 else 2)
    def call():
        lin._it[0] += 1
        lin.weight = lin._pool[lin._it[0] % len(lin._pool)]
        return lin(mx.array((rng.standard_normal((1, kin)) * 0.3).astype(np.float16)).astype(dtype))
    ms = slope(call)
    roof = nbytes / (a.bw * 1e9) * 1e3
    print(f"{name:28s} {ms:8.4f} ms/call  bytes {nbytes/1e6:7.2f} MB roof {roof:7.4f} ms x{count}/step", flush=True)
    return dict(op=name, ms=round(ms, 4), mb=round(nbytes / 1e6, 2), roof_ms=round(roof, 4), per_step=count)


res = []
# ---- dense decode GEMVs (real DS4.1 shapes; EXL3 path; K=3 proxy, bytes measured) ----
res.append(bench_exl3("gemv_wq_a_1280x5120", 5120, 1280, count=38))
res.append(bench_exl3("gemv_wkv_512x5120", 5120, 512, count=38))
res.append(bench_exl3("gemv_wq_b_32768x1280", 1280, 32768, count=38))
res.append(bench_exl3("gemv_wo_b_5120x8192", 8192, 5120, count=40))
res.append(bench_exl3("gemv_idx_wq_b_4096x1280", 1280, 4096, count=8))
res.append(bench_dense("gemv_idx_wproj_32x5120", 5120, 32, count=8))
res.append(bench_exl3("gemv_comp_wkv/wgate_512x5120", 5120, 512, count=36))
res.append(bench_exl3("gemv_shared_gate/up_2304x5120", 5120, 2304, count=80))
res.append(bench_exl3("gemv_shared_down_5120x2304", 2304, 5120, count=40))
res.append(bench_exl3("gemv_gate_fp32_384x5120", 5120, 384, count=40, dtype=mx.float32))
def bench_fn(name, thunk, count=1, note=""):
    ms = slope(thunk)
    print(f"{name:28s} {ms:8.4f} ms/call x{count}/step {note}", flush=True)
    return dict(op=name, ms=round(ms, 4), per_step=count)


res.append(bench_dense("gemv_hc_fn_24x20480", 20480, 24, count=80, dtype=mx.float32))
res.append(bench_exl3("gemv_head_129280x5120", 5120, 129280, count=1, dtype=mx.float32))

# ---- norms / rope / elementwise (bf16, pack-independent) ----
def rms_thunk():
    import mlx.nn as nn
    w = mx.array((rng.standard_normal(5120) * 0.1 + 1).astype(np.float16)).astype(mx.bfloat16)
    n = nn.RMSNorm(5120)
    n.weight = w
    mx.eval(n.weight)
    def thunk():
        x = mx.array((rng.standard_normal((1, 5120)) * 2).astype(np.float16)).astype(mx.bfloat16)
        return n(x)
    return thunk
res.append(bench_fn("rmsnorm_5120", rms_thunk(), count=160, note="attn/ffn norms+q/kv norms"))

# mhc expand / pre
from kyojin_mlx import fuse as F
def mhc_e_thunk():
    def thunk():
        x = mx.array((rng.standard_normal((1, 1, 5120)) * 0.02).astype(np.float16)).astype(mx.bfloat16)
        r = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16)).astype(mx.bfloat16)
        p = mx.array(rng.standard_normal((1, 1, 4)).astype(np.float32))
        c = mx.array((rng.standard_normal((1, 1, 4, 4)) * 0.1).astype(np.float32))
        return F.mhc_expand(x, r, p, c)[0]
    return thunk
res.append(bench_fn("mhc_expand", mhc_e_thunk(), count=80))
def mhc_p_thunk():
    def thunk():
        h = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16)).astype(mx.bfloat16)
        p = mx.array(rng.standard_normal((1, 1, 4)).astype(np.float32))
        return F.mhc_pre(h, p)
    return thunk
res.append(bench_fn("mhc_pre", mhc_p_thunk(), count=120))

if a.json:
    import json
    json.dump(res, open(a.json, "w"), indent=1)

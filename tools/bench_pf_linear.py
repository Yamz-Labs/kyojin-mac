#!/usr/bin/env python3
"""Prefill linear microbench: Exl3Linear (default path, and fused NAX forced) vs MLX 4-bit qmm vs fp16 matmul roofline at the same shapes.
   PYTHONPATH=$PWD python tools/bench_pf_linear.py [--rows 256,512,4096] [--reps 9] [--classes ...]"""
import argparse, json, statistics, time, os
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="256,512,4096"); ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--pack", default=os.path.expanduser("~/rivals/models/Qwen3.5-4B-exl3-4.0"))
ap.add_argument("--classes", default="")
ap.add_argument("--arms", default="ours,fused,qmm,qmmbf,fp16")
a = ap.parse_args()
mx.set_cache_limit(512 << 20)
L = "model.language_model.layers."
CL = {"qkv": L + "9.linear_attn.in_proj_qkv", "z": L + "9.linear_attn.in_proj_z", "o": L + "8.linear_attn.out_proj", "v": L + "7.self_attn.v_proj",
      "gate": L + "9.mlp.gate_proj", "up": L + "9.mlp.up_proj", "down": L + "9.mlp.down_proj", "head": "lm_head"}
sel = a.classes.split(",") if a.classes else list(CL)
cb = json.load(open(f"{a.pack}/config.json"))["quantization_config"]["codebook"]
w = mx.load(f"{a.pack}/model.safetensors")
arms = a.arms.split(",")
INN = 8     # calls per eval (distinct inputs not needed: weights re-read from memory/SLC either way)

def tm(fn):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter(); mx.eval([fn() for _ in range(INN)]); return (time.perf_counter() - t0) / INN * 1e3

print(f"cb={cb}  ms per call (TFLOPS); min over {a.reps} interleaved reps")
for c in sel:
    p = CL[c]
    if p + ".trellis" not in w:
        print(c, "missing", p); continue
    lin = EL.Exl3Linear(w[p + ".trellis"], w[p + ".suh"], w[p + ".svh"], cb=cb)
    kin, nout = lin.in_features, lin.out_features
    mw = mx.random.normal((nout, kin)).astype(mx.bfloat16)
    qw, sc, bi = mx.quantize(mw, group_size=64, bits=4)
    wfp = mw.astype(mx.float16).T
    mx.eval(qw, sc, bi, wfp)
    for M in map(int, a.rows.split(",")):
        x = (mx.random.normal((M, kin)) * 0.7).astype(mx.float16); xb = x.astype(mx.bfloat16); mx.eval(x, xb)
        fns = {"ours": lambda: lin(x), "qmm": lambda: mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=64, bits=4),
               "fp16": lambda: x @ wfp, "qmmbf": lambda: mx.quantized_matmul(xb, qw, sc, bi, transpose=True, group_size=64, bits=4)}
        def fused():
            return lin(x)
        res = {k: [] for k in arms}
        for r in range(a.reps + 1):
            for k in arms:
                if k == "fused":
                    old = (lin.fused_max, EL.DENSE_FUSED_MAX); lin.fused_max = 1 << 20
                    v = tm(fused); lin.fused_max = old[0]
                else:
                    v = tm(fns[k])
                if r: res[k].append(v)
        fl = 2 * M * kin * nout / 1e9
        s = f"{c:5s} {kin}x{nout} K{lin.K} M={M:5d}: " + "  ".join(f"{k} {min(v):7.3f} ({fl/min(v):5.1f})" for k, v in res.items())
        for q in ("qmm", "qmmbf"):
            if q in res and "ours" in res: s += f"  ours/{q} {min(res['ours'])/min(res[q]):.2f}"
        print(s, flush=True)

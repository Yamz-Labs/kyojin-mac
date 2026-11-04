"""Dense prefill linear, sync vs pipelined: fused NAX decode vs decode + MLX matmul vs MLX qmm vs fp16 (weights of the real pack, ms per call, min over reps).
Sync = mx.eval after every call (dependent-chain latency, what the model graph sees); pipe = 8 calls in one eval (microbench style).
usage: PYTHONPATH=$PWD python tools/bench_pf_lin2.py PACK [--rows 512,2048] [--reps 5] [--classes qkv,z,o,gate,down]"""
import argparse, json, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="512"); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--classes", default="qkv,z,o,gate,down"); a = ap.parse_args()
mx.set_cache_limit(512 << 20)
L = "model.language_model.layers."
CL = {"qkv": L + "9.linear_attn.in_proj_qkv", "z": L + "9.linear_attn.in_proj_z", "o": L + "8.linear_attn.out_proj", "v": L + "7.self_attn.v_proj",
      "gate": L + "9.mlp.gate_proj", "up": L + "9.mlp.up_proj", "down": L + "9.mlp.down_proj"}
cb = json.load(open(f"{a.pack}/config.json"))["quantization_config"]["codebook"]
w = mx.load(f"{a.pack}/model.safetensors")
def sync(fn, n=8):
    mx.eval(fn()); t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1e3
def pipe(fn, n=8):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter(); mx.eval([fn() for _ in range(n)]); return (time.perf_counter() - t0) / n * 1e3
for c in a.classes.split(","):
    p = CL[c]; lin = EL.Exl3Linear(w[p + ".trellis"], w[p + ".suh"], w[p + ".svh"], cb=cb); kin, nout = lin.in_features, lin.out_features
    mw = mx.random.normal((nout, kin)).astype(mx.bfloat16); qw, sc, bi = mx.quantize(mw, group_size=64, bits=4)
    for M in map(int, a.rows.split(",")):
        x = (mx.random.normal((M, kin)) * 0.7).astype(mx.float16); mx.eval(x, qw, sc, bi)
        def mode(v):
            def f():
                lin.fused_max = v; EL.DENSE_FUSED_MAX = v; return lin(x)
            return f
        fns = {"fused": mode(1 << 20), "dec+mm": mode(0), "qmm": lambda: mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=64, bits=4)}
        out = []
        for name, t in (("sync", sync), ("pipe", pipe)):
            res = {k: [] for k in fns}
            for _ in range(a.reps):
                for k in fns: res[k].append(t(fns[k]))
            out.append(f"{name}: " + " ".join(f"{k} {min(v):.2f}" for k, v in res.items()))
        print(f"{c:5s} {kin}x{nout} K{lin.K} M={M}: " + " | ".join(out), flush=True)

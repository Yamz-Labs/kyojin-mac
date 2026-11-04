#!/usr/bin/env python3
"""Qwen3.5-2B (or any dense EXL3 pack) prefill end to end, fused EXL3 + NAX dense GEMM vs decode + matmul, interleaved A/B in one process
(the Air is shared: only same-minute pairs are comparable). Prompts above KYOJIN_DENSE_FUSED_MAX rows run the same code in both arms (noise control).
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/bench_nax_dense_e2e.py MODEL_DIR [--prefill 16,64,128,512] [--reps 7]"""
import argparse, statistics, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx import exl3_linear as EL, metal_nax as NX

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--prefill", default="16,64,128,512"); ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--cap-gb", type=float, default=2.8, help="abort when the peak memory goes over (the Air is shared)")
a = ap.parse_args()
mx.set_cache_limit(512 << 20)
_gemm, OLD_TILE = NX.expert_gemm, (64, 1, 32, 32)
NX.expert_gemm = lambda *args, **kw: _gemm(*(args[:6] + (OLD_TILE,) + args[7:]), **kw) if ARM == "fused-old-tile" and len(args) > 6 else _gemm(*args, **kw)
ARM = ""
model, _ = EL.load_exl3(a.path)
ids_all = AutoTokenizer.from_pretrained(a.path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400)
lm = model.language_model

def prefill(P):
    x = mx.array(ids_all[:P])[None]
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    h = lm.model(x, cache)[:, -1:]
    out = lm.lm_head(h)[:, -1]
    mx.eval(out)
    return time.perf_counter() - t0, out

ARMS = ("decode+matmul", "fused-old-tile", "fused")


def set_arm(arm):
    global ARM
    ARM = arm
    EL.DENSE_FUSED_MAX = 0 if arm == "decode+matmul" else 128


for P in map(int, a.prefill.split(",")):
    res = {k: [] for k in ARMS}
    outs = {}
    for arm in ARMS:                                       # warm-up (kernel compile)
        set_arm(arm)
        outs[arm] = prefill(P)[1]
    for _ in range(a.reps):
        for arm in ARMS:
            set_arm(arm)
            res[arm].append(P / prefill(P)[0])
    d = mx.abs(outs["fused"].astype(mx.float32) - outs["decode+matmul"].astype(mx.float32)).max().item()
    same = bool(mx.argmax(outs["fused"], -1).item() == mx.argmax(outs["decode+matmul"], -1).item())
    dn = mx.abs(outs["fused"].astype(mx.float32) - outs["fused-old-tile"].astype(mx.float32)).max().item()
    m = {k: statistics.median(v) for k, v in res.items()}; b = {k: max(v) for k, v in res.items()}
    print(f"P={P:5d}  tok/s median (best): decode+matmul {m['decode+matmul']:7.1f} ({b['decode+matmul']:7.1f}) | fused old tile {m['fused-old-tile']:7.1f} ({b['fused-old-tile']:7.1f})"
          f" | fused {m['fused']:7.1f} ({b['fused']:7.1f})  x{m['fused']/m['decode+matmul']:.2f} vs decode+matmul, x{m['fused']/m['fused-old-tile']:.2f} vs old tile"
          f"  max |dlogit| {d:.3g}, new vs old tile {dn:.3g}  same argmax {same}", flush=True)
    pk = mx.get_peak_memory() / 1e9
    if pk > a.cap_gb:
        print(f"peak {pk:.2f} GB over the cap, stop"); break
print(f"peak {mx.get_peak_memory()/1e9:.2f} GB")

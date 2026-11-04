"""Burst-state stage times of one DS4.1 layer (Air, Low Power Mode: sustained GPU load runs at ~45 % of the clock a stage gets right after an idle gap).
Each stage: `reps` rounds, every round starts after `idle` s of idle GPU, then `chain` back-to-back calls are timed one by one; the minimum over all calls is the burst time.
Stages use the real modules of the ds2 slice (full DS4.1 dims) with random inputs. usage: bench_burst.py PACK T [reps]"""
import sys, time, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, attn_prefill as AP, mhc_prefill as MPF, fuse as F, exl3_linear as EL
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
L0 = model.language_model.layers[0]; LM = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
rng = np.random.default_rng(0); H = 5120
IDLE = 1.5
def burst(fn, chain=3):
    mx.eval(fn()); best = 1e9
    for _ in range(reps):
        time.sleep(IDLE)
        for _ in range(chain):
            t0 = time.perf_counter(); mx.eval(fn()); best = min(best, time.perf_counter() - t0)
    return best * 1e3
def rnd(*shape, dt=mx.float16, s=1.0): return mx.array((rng.standard_normal(shape) * s).astype(np.float32)).astype(dt)
out = {}
a = L0.attn
x = rnd(1, T, H)
for n in ("wq_a", "wkv"): out["lin." + n] = burst(lambda m=getattr(a, n): m(x))
out["lin.wq_b"] = burst(lambda: a.wq_b(rnd(1, T, 1280)) if False else a.wq_b(xq), ) if False else None
xq = rnd(1, T, 1280); out["lin.wq_b"] = burst(lambda: a.wq_b(xq))
xo = rnd(1, T, 8192); out["lin.wo_b"] = burst(lambda: a.wo_b(xo))
xg = rnd(1, 8, T, 4096); out["lin.wo_a"] = burst(lambda: a.wo_a(xg))
sh = L0.ffn.shared_experts; out["shared_experts"] = burst(lambda: sh(x))
# attention core: ratio-2 sparse (window 128 + top-512 of T/2) and window-only
Hh, D, W, topk = 64, 512, 128, 512
q = rnd(1, Hh, T, D, s=0.5); win = rnd(1, T, D, s=0.5); Np = T // 2; pool = rnd(1, Np, D, s=0.5); sink = mx.array(rng.standard_normal(Hh).astype(np.float32))
idx = np.full((1, T, topk), -1, np.int32)
for t in range(T):
    n = (t + 1) // 2; k = min(topk, n)
    if k: idx[0, t, :k] = rng.choice(n, k, replace=False)
idxs = mx.array(idx); sc = D ** -0.5
out["attn.sparse_core"] = burst(lambda: AP.sparse_attention(q, win, pool, idxs, sc, sink, W))
pool0 = win[:, :1]; idx0 = mx.zeros((1, T, 0), dtype=mx.int32)
out["attn.window_only_core"] = burst(lambda: AP.sparse_attention(q, win, pool0, idx0, sc, sink, W))
# mHC glue on [1, T, 4, D]
h4 = rnd(1, T, 4, H); post = mx.array(rng.uniform(0, 2, (1, T, 4)).astype(np.float32)); comb = mx.array(rng.uniform(0, 1, (1, T, 4, 4)).astype(np.float32)); pre = mx.array(rng.uniform(0, 1, (1, T, 4)).astype(np.float32))
xs = rnd(1, T, H)
exp = F._impl_mhc_expand(LM.hc_expand); out["glue.hc_expand"] = burst(lambda: exp(xs, h4, post, comb))
pf = F._impl_mhc_pre(LM.DeepseekV41Block.__dict__["hc_pre"].__func__ if isinstance(LM.DeepseekV41Block.__dict__["hc_pre"], staticmethod) else LM.DeepseekV41Block.hc_pre)
out["glue.hc_pre"] = burst(lambda: pf(h4, pre))
fn = L0.hc_attn_fn; sc3 = L0.hc_attn_scale; bs = L0.hc_attn_base
out["glue.hc_mixes"] = burst(lambda: L0.hc_mixes(h4, fn, sc3, bs))
out["glue.norm"] = burst(lambda: L0.attn_norm(xs))
fqf = LM.fake_quant_fp8_ue8m0; out["glue.fake_quant_fp8 (x1 of 3 per layer)"] = burst(lambda: fqf(xs))
for k, v in out.items(): print(f"T={T} {k:42s} {v:8.2f} ms", flush=True)

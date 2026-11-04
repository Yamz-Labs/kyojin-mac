"""Accuracy of GDN prefill cores vs an fp32 step reference (same bf16 q,k,v values). usage: gdn_acc84.py [T]
variants: step (stock kernel, bf16 inputs), fast_bf (mx.fast chunked, bf16 q/k/v, fp32 g/beta passed), fast_f16, fast_f32 (all fp32 inputs)."""
import sys, mlx.core as mx
from mlx_lm.models import gated_delta as G
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
mx.random.seed(1)
q = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); k = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
a = mx.random.normal((1, T, 32)).astype(mx.bfloat16); b = mx.random.normal((1, T, 32)).astype(mx.bfloat16)
A_log = mx.log(mx.random.uniform(low=0.5, high=16, shape=(32,))); dtb = mx.random.normal((32,)).astype(mx.bfloat16) * 0.5
q, k = G.normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)
q = q.astype(mx.bfloat16); k = k.astype(mx.bfloat16)
st = mx.zeros((1, 32, 128, 128), dtype=mx.float32)
g = G.compute_g(A_log, a, dtb); beta = mx.sigmoid(b.astype(mx.float32))
print("g range", float(g.min()), float(g.max()), "mean", float(g.mean()))
mx.eval(q, k, v, g, beta)
f32 = lambda x: x.astype(mx.float32)
ref_y, ref_s = G.gated_delta_kernel(f32(q), f32(k), f32(v), g, beta, st, None); ref_y = f32(ref_y)
def rep(n, y, s):
    y = f32(y); d = mx.abs(y - ref_y); ds = mx.abs(s - ref_s)
    print(f"{n:9s} y max|d| {float(d.max()):.3g} rms {float(mx.sqrt((d*d).mean())):.3g} (|y|rms {float(mx.sqrt((ref_y*ref_y).mean())):.3g})  state max|d| {float(ds.max()):.3g} (|s|max {float(mx.abs(ref_s).max()):.3g})", flush=True)
y, s = G.gated_delta_kernel(q, k, v, g, beta, st, None); rep("step_bf", y, s)
y, s = mx.fast.gated_delta_update(q, k, v, g, beta, st); rep("fast_bf", y, s)
h = mx.float16
y, s = mx.fast.gated_delta_update(q.astype(h), k.astype(h), v.astype(h), g, beta.astype(h), st); rep("fast_f16", y, s)
y, s = mx.fast.gated_delta_update(f32(q), f32(k), f32(v), g, beta, st); rep("fast_f32", y, s)

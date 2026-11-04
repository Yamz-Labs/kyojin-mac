"""Forked NAX GDN kernel vs fp32 step reference, and its time vs stock/mx.fast. usage: gdn_nax_test84.py [T] [realistic=1]"""
import sys, time, mlx.core as mx
from mlx_lm.models import gated_delta as G
sys.path.insert(0, __import__("os").path.dirname(__file__)); import gdn_nax84 as N
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
corr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0     # key correlation between rows (real keys are strongly correlated)
mx.random.seed(1)
q = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); k = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16); v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
a = mx.random.normal((1, T, 32)).astype(mx.bfloat16) - 3; b = (mx.random.normal((1, T, 32)) + (3 if corr else 0)).astype(mx.bfloat16)
A_log = mx.log(mx.random.uniform(low=0.5, high=16, shape=(32,))); dtb = (mx.zeros((32,))).astype(mx.bfloat16)
if corr: k = (mx.random.normal((1, 1, 16, 128)) * 3 + k * (1 - corr)).astype(mx.bfloat16)
q, k = G.normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6); q = q.astype(mx.bfloat16); k = k.astype(mx.bfloat16)
st = mx.random.normal((1, 32, 128, 128)) * 0.05
g = G.compute_g(A_log, a, dtb); beta = mx.sigmoid(b.astype(mx.float32)); mx.eval(q, k, v, g, beta, st)
print("g range", float(g.min()), float(g.max()), "mean", float(g.mean()))
f32 = lambda x: x.astype(mx.float32)
ref_y, ref_s = G.gated_delta_kernel(f32(q), f32(k), f32(v), g, beta, st, None); ref_y = f32(ref_y)
def rep(n, y, s):
    y = f32(y); d = mx.abs(y - ref_y); ds = mx.abs(s - ref_s)
    print(f"{n:22s} y max|d| {float(d.max()):.3g} rms {float(mx.sqrt((d*d).mean())):.3g} (|y|rms {float(mx.sqrt((ref_y*ref_y).mean())):.3g})  state max|d| {float(ds.max()):.3g} rms {float(mx.sqrt((ds*ds).mean())):.3g} (|s|max {float(mx.abs(ref_s).max()):.3g})", flush=True)
def bench(fn, it=6, blocks=5):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best * 1e3
cands = {"step_bf": lambda: G.gated_delta_kernel(q, k, v, g, beta, st, None),
         "fast_bf(g bf16)": lambda: mx.fast.gated_delta_update(q, k, v, g, beta, st),
         "fast_f32": lambda: mx.fast.gated_delta_update(f32(q), f32(k), f32(v), g, beta, st)}
cands["nax"] = lambda: N.gated_delta_nax(q, k, v, g, beta, st)
for n, f in cands.items():
    y, s = f(); rep(n, y, s)
for n, f in cands.items():
    print(f"{n:22s} {bench(f):7.3f} ms", flush=True)

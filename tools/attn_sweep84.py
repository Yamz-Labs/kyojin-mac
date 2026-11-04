"""Sweep tile shapes of the forked d-split hd=256 kernel, interleaved with the fork baseline. usage: attn_sweep84.py [o]"""
import sys, mlx.core as mx
import attn_ds84 as A
o = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
L = 2048; N = o + L
q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal"); mx.eval(ref)
fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
cfgs = [(64, 32, 4, 2), (64, 64, 4, 2), (64, 16, 4, 2), (64, 32, 4, 4), (64, 64, 4, 4), (32, 32, 2, 2), (32, 32, 2, 4), (128, 32, 8, 2), (128, 32, 8, 4), (128, 64, 8, 2)]
for c in cfgs:
    try:
        y = A.attention(q, k, v, 0.0625, *c); mx.eval(y)
        d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        t = A.bench(lambda: A.attention(q, k, v, 0.0625, *c)); t0 = A.bench(lambda: A.attention(q, k, v, 0.0625))
        print(f"BQ{c[0]} BK{c[1]} WM{c[2]} WN{c[3]}: {t*1e3:6.2f} ms {fl/t/1e12:5.2f} TF  (base fork {fl/t0/1e12:5.2f})  max|d| {d:.2g}", flush=True)
    except Exception as e:
        print(c, "ERR", str(e).split("error:")[-1][:120].strip(), flush=True)

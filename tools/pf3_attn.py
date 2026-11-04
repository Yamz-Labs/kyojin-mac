import os
import sys, mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/kmlx-macpf3/tools"))
import attn_ds84 as A
o = int(sys.argv[1]); L = 2048; N = o + L
q = mx.random.normal((1, 16, L, 256)).astype(mx.float16); k = mx.random.normal((1, 4, N, 256)).astype(mx.float16); v = mx.random.normal((1, 4, N, 256)).astype(mx.float16); mx.eval(q, k, v)
fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
ref = st(); mx.eval(ref)
t = A.bench(st); print(f"stock sdpa o={o} fp16: {t*1e3:.2f} ms {fl/t/1e12:.2f} TF", flush=True)
for c in [(128, 32, 8, 2), (64, 32, 4, 2), (128, 64, 8, 2)]:
    try:
        y = A.attention(q, k, v, 0.0625, *c); mx.eval(y)
        d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        t = A.bench(lambda: A.attention(q, k, v, 0.0625, *c)); print(f"fork {c}: {t*1e3:.2f} ms {fl/t/1e12:.2f} TF max|d| {d:.2g}", flush=True)
    except Exception as e: print(c, "ERR", str(e)[-150:])

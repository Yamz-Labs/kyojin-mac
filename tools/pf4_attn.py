"""pf4: tile sweep of the forked hd256 NAX dsplit attention vs stock sdpa, interleaved. usage: pf4_attn.py O [hd]"""
import sys, os, mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
import attn_ds84 as A
o = int(sys.argv[1]); L = 2048; N = o + L; hd = 256
q = mx.random.normal((1, 16, L, hd)).astype(mx.float16); k = mx.random.normal((1, 4, N, hd)).astype(mx.float16); v = mx.random.normal((1, 4, N, hd)).astype(mx.float16); mx.eval(q, k, v)
fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
ref = st(); mx.eval(ref)
cfgs = [(64, 32, 4, 2), (64, 64, 4, 2), (64, 128, 4, 2), (128, 64, 8, 2), (64, 64, 4, 4), (32, 64, 2, 2), (128, 32, 8, 2)]
fs = {"stock": st}
for c in cfgs:
    try:
        y = A.attention(q, k, v, 0.0625, *c); mx.eval(y)
        d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()); fs[str(c)] = (lambda c=c: A.attention(q, k, v, 0.0625, *c)); print(c, "ok d", d, flush=True)
    except Exception as e: print(c, "ERR", str(e)[-120:], flush=True)
best = {n: 9 for n in fs}; ts = {n: [] for n in fs}
for r in range(4):
    for n, f in fs.items():
        t = A.bench(f, 2, 2); ts[n].append(t)
for n in fs:
    s = sorted(ts[n]); print(f"o={o} {n}: best {fl/s[0]/1e12:.2f} TF med {fl/s[len(s)//2]/1e12:.2f} TF ({s[0]*1e3:.2f} ms)", flush=True)

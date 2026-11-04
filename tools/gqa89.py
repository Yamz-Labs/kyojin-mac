"""gqa89.py N T: fp16 K/V of N keys (Hk=4, D=256, 8 layers), T query rows per head (Hq=16): stock sdpa (q_len T, causal tail) vs the GQA-folded call (the G*T rows of a kv head as one 'head' of q_len G*T, explicit mask)."""
import sys, time, json, statistics
import mlx.core as mx
N, T = int(sys.argv[1]), int(sys.argv[2]); Hk, G, D, NL = 4, 4, 256, 8; Hq = Hk * G
mx.random.seed(1)
ks = [mx.random.normal((1, Hk, N, D)).astype(mx.float16) for _ in range(NL)]; vs = [mx.random.normal((1, Hk, N, D)).astype(mx.float16) for _ in range(NL)]
q = mx.random.normal((1, Hq, T, D)).astype(mx.float16); mx.eval(ks, vs, q); sc = D ** -0.5
def stock():
    return [mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal") for k, v in zip(ks, vs)]
# row (g, t) of kv head h is query head h*G+g at position t; it sees keys <= N - T + t
pos = mx.arange(N)[None, :]; lim = (N - T + mx.arange(T))[:, None]
m1 = mx.where(pos <= lim, 0.0, -mx.inf).astype(mx.float16)                  # [T, N]
mask = mx.tile(m1, (G, 1))[None, None]                                       # [1, 1, G*T, N], row = g*T + t
def folded():
    qf = q.reshape(1, Hk, G, T, D).reshape(1, Hk, G * T, D)
    return [mx.fast.scaled_dot_product_attention(qf, k, v, scale=sc, mask=mask).reshape(1, Hk, G, T, D).reshape(1, Hq, T, D) for k, v in zip(ks, vs)]
a = mx.concatenate(stock()); b = mx.concatenate(folded()); mx.eval(a, b)
print("maxdiff", float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max()))
res = {"stock": [], "folded": []}
for f in (stock, folded): mx.eval(f())
for _ in range(10):
    for n, f in (("stock", stock), ("folded", folded)):
        t0 = time.perf_counter(); mx.eval(f()); res[n].append((time.perf_counter() - t0) * 1e3)
print("GQA", N, T, json.dumps({k: [round(min(v), 2), round(statistics.median(v), 2)] for k, v in res.items()}))

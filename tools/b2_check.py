import numpy as np, mlx.core as mx
from kyojin_mlx import tq as TQ, kvwin as KW
KW.install()
r = np.random.default_rng(3); d = 128
f = lambda h, n: mx.array(r.standard_normal((2, h, n, d)).astype(np.float16))
q, k, v = f(8, 200), f(2, 200), f(2, 200)
def run(c, q, k, v, chunks):
    outs, s = [], 0
    for n in chunks:
        kv = c.update_and_fetch(k[:, :, s:s + n], v[:, :, s:s + n])
        outs.append(KW._base.scaled_dot_product_attention(q[:, :, s:s + n], kv[0], kv[1], cache=c, scale=d ** -0.5, mask=None)); s += n
    return mx.concatenate(outs, axis=2)
ch = [97, 40, 1, 1, 33, 1, 27]
both = run(TQ.TQKVCache(6, 6, 4, 32), q, k, v, ch)
one = [run(TQ.TQKVCache(6, 6, 4, 32), q[i:i+1], k[i:i+1], v[i:i+1], ch) for i in range(2)]
print("B=2 vs B=1 max abs diff", max(float(mx.abs(both[i] - one[i][0]).max()) for i in range(2)))

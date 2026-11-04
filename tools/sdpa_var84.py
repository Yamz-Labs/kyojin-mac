"""Variants of the prefill sdpa call (same math, different launch shapes): base | per kv-group calls | q split in row blocks | L sweep. usage: sdpa_var84.py [o,o,..]"""
import sys, time, mlx.core as mx
offs = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16384, 30720]
def bench(fn, it=4, blocks=5):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
L = 2048; sc = 256 ** -0.5
for o in offs:
    N = o + L; dt = mx.bfloat16
    q = mx.random.normal((1, 16, L, 256)).astype(dt); k = mx.random.normal((1, 4, N, 256)).astype(dt); v = mx.random.normal((1, 4, N, 256)).astype(dt); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
    S = mx.fast.scaled_dot_product_attention
    def base(): return S(q, k, v, scale=sc, mask="causal")
    def grp(): return mx.concatenate([S(q[:, 4 * g:4 * g + 4], k[:, g:g + 1], v[:, g:g + 1], scale=sc, mask="causal") for g in range(4)], axis=1)
    def rows(nb):                                                              # q row blocks, each sees the keys up to its last row (causal mask offset by the key count)
        outs = []
        for b in range(nb):
            r0 = b * L // nb; r1 = (b + 1) * L // nb; n = o + r1
            outs.append(S(q[:, :, r0:r1], k[:, :, :n], v[:, :, :n], scale=sc, mask="causal"))
        return mx.concatenate(outs, axis=2)
    def kvx():                                                                 # K/V repeated to 16 heads
        return S(q, mx.repeat(k, 4, axis=1), mx.repeat(v, 4, axis=1), scale=sc, mask="causal")
    ref = base(); mx.eval(ref)
    for name, fn in (("base", base), ("per-kv-group", grp), ("rows x2", lambda: rows(2)), ("rows x4", lambda: rows(4)), ("kv expanded", kvx)):
        t = bench(fn); y = fn(); mx.eval(y)
        print(f"o={o:6d} {name:14s}: {t*1e3:7.2f} ms {fl/t/1e12:5.2f} TF  max|d| vs base {float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)))):.3g}", flush=True)

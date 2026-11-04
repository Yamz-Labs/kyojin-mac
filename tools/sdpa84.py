"""Prefill sdpa microbench at Qwen3.5 attention shape (16 q heads, 4 kv heads, hd 256), L query rows at offset o. usage: sdpa84.py [L=2048] [o,o,..]
TF = visible-key FLOPs (causal) / time; also mask=None (no skipping) to see whether the causal mask skips blocks."""
import sys, time, mlx.core as mx
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
offs = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 4096, 16384, 30720]
def bench(fn, it=6, blocks=4):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
for dt in (mx.bfloat16, mx.float16):
    for o in offs:
        N = o + L
        q = mx.random.normal((1, 16, L, 256)).astype(dt); k = mx.random.normal((1, 4, N, 256)).astype(dt); v = mx.random.normal((1, 4, N, 256)).astype(dt); mx.eval(q, k, v)
        fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
        for name, mask in (("causal", "causal"), ("none", None)):
            t = bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask=mask))
            print(f"{str(dt)[9:]:9s} o={o:6d} mask={name:6s}: {t*1e3:8.2f} ms  {fl/t/1e12:5.2f} TF (visible) {(4*L*N*16*256)/t/1e12:5.2f} TF (full)", flush=True)

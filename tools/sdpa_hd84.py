"""MLX fast sdpa TF vs head dim (GQA 4:1, causal prefill chunk L=2048 at offset o): is hd=256 below what the NAX path reaches at hd=64/128? interleaved min of blocks."""
import sys, time, mlx.core as mx
S = mx.fast.scaled_dot_product_attention
def bench(fn, it=3, blocks=6):
    mx.eval(fn()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [fn() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best
L = 2048
for o in (8192, 24576):
    N = o + L; cs = {}
    for hd, nh in ((64, 32), (128, 32), (256, 16)):                       # same q-head count x hd = 8192 per row for hd 128/256; hd 64 doubles heads
        q = mx.random.normal((1, nh, L, hd)).astype(mx.bfloat16); k = mx.random.normal((1, nh // 4, N, hd)).astype(mx.bfloat16); v = mx.random.normal((1, nh // 4, N, hd)).astype(mx.bfloat16); mx.eval(q, k, v)
        cs[hd] = (lambda q=q, k=k, v=v, hd=hd: S(q, k, v, scale=hd ** -0.5, mask="causal"), 4 * L * (o + (L + 1) / 2) * nh * hd)
    for r in range(2):
        for hd, (f, fl) in cs.items():
            t = bench(f); print(f"o={o} hd={hd:3d}: {t*1e3:7.2f} ms {fl/t/1e12:5.2f} TF", flush=True)

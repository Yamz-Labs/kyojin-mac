"""GPU micro: gdn_fuse.fused_step batched (one launch, B rows) vs per-row loop. Qwen4Exp-like dims: Hk 8? read from args. Run: PYTHONPATH=. python tests/micro_gdn_batch.py"""
import sys, time
import mlx.core as mx
from kyojin_mlx import gdn_fuse as G

Hk, Hv = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (16, 32)
C = (2 * Hk + Hv) * 128
dt = mx.bfloat16
mx.random.seed(1)
rn = lambda *s: (mx.random.normal(s) * 1.5).astype(dt)
w = rn(C, 4, 1) * 0.3; Al = mx.log(mx.random.uniform(low=0.5, high=8, shape=(Hv,))); dtb = (mx.random.normal((Hv,)) * .5).astype(mx.float32)
nw = (1 + 0.1 * mx.random.normal((128,))).astype(dt)

def loop(B, a):
    qkv, z, aa, b, cs, st = a
    r = [G.fused_step(qkv[i:i+1], z[i:i+1], aa[i:i+1], b[i:i+1], cs[i:i+1], st[i:i+1], w, Al, dtb, nw, 1e-6, Hk, Hv, qw4=True)[:3] for i in range(B)]
    return [mx.concatenate([x[j] for x in r], 0) for j in range(3)]
def bat(B, a):
    qkv, z, aa, b, cs, st = a
    return list(G.fused_step(qkv, z, aa, b, cs, st, w, Al, dtb, nw, 1e-6, Hk, Hv, qw4=True)[:3])
def tm(f, n=200):
    for _ in range(10): mx.eval(f())
    t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1e6
for B in (2, 4):
    for S in (1, 2, 3, 4):
        a = (rn(B, S, C), rn(B, S, Hv, 128), rn(B, S, Hv), rn(B, S, Hv), rn(B, 3, C), mx.random.normal((B, Hv, 128, 128)))
        mx.eval(*a)
        x, y = loop(B, a), bat(B, a)
        d = max(mx.max(mx.abs(p.astype(mx.float32) - q.astype(mx.float32))).item() for p, q in zip(x, y))
        print(f"B={B} S={S} loop {tm(lambda: loop(B, a)):.0f}us batched {tm(lambda: bat(B, a)):.0f}us maxdiff {d}")

"""q4nax launch-shape sweep at ROWS rows for the dense shapes of the 4B pack, hot, vs MLX qmm and the fp16 matmul peak. usage: mu4_nax.py ROWS"""
import sys, time, itertools, os
import mlx.core as mx
from kyojin_mlx import q4mma as Q
R = int(sys.argv[1])
z = mx.random.normal((2048, 2048)).astype(mx.float16)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 6: mx.eval([z @ z for _ in range(8)])
def tm(f, n=30):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval([f() for _ in range(8)]); ts.append((time.perf_counter() - t) / 8)
    ts.sort(); return ts[n // 2] * 1e3
t = tm(lambda: z @ z); print("fp16 matmul 2048^3: %.3f ms = %.2f TFLOPs" % (t, 2 * 2048 ** 3 / t / 1e9))
SH = [tuple(int(v) for v in a.split('x')) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else ((9216, 2560), (2560, 9216), (8192, 2560), (4096, 2560), (2560, 4096), (18432, 2560))
for N, K in SH:
    w = mx.random.normal((N, K)).astype(mx.float16); wq, sc, bi = mx.quantize(w, group_size=64, bits=4); x = mx.random.normal((R, K)).astype(mx.float16); mx.eval(wq, sc, bi, x)
    byts = wq.nbytes + sc.nbytes + bi.nbytes
    ref = tm(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4))
    cur = Q.nax_cfg(N, K, R)
    res = []
    for u, nb, ks in itertools.product((1, 2, 4), (1, 2, 4, 8), (1, 2, 4)):
        if K % (128 * u) or nb * ks > 16: continue
        try:
            xp = Q._xn(x, u)
            res.append((tm(lambda: Q.q4nax(xp, wq, sc, bi, R, u, nb, ks)), u, nb, ks))
        except Exception as e:
            pass
    res.sort()
    ms_cur = next((r[0] for r in res if cur and r[1:] == tuple(cur)), None)
    print("N=%d K=%d R=%d  qmm %.3f  cur %s %s  best %s  | weight-roof %.3f ms, flop-roof(peak) %.3f ms" % (N, K, R, ref, cur, "%.3f" % ms_cur if ms_cur else "-", ["%.3f u%d nb%d ks%d" % r for r in res[:3]], byts / 139.3e9 * 1e3, 2 * R * N * K / (t and 2 * 2048 ** 3 / (t / 1e3)) * 1e3))

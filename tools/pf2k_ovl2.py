import sys, time, mlx.core as mx, numpy as np
from kyojin_mlx import exl3_linear as EL, kernels as KN
d, K, R = 2560, 4, 2048
def lin(seed, kin, nout):
    kt, nt = kin // 16, nout // 16
    tr = mx.array(np.random.default_rng(seed).integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    return EL.Exl3Linear(tr, mx.ones((kin,), dtype=mx.float16), mx.ones((nout,), dtype=mx.float16), cb="mul1")
l1, l2 = lin(1, 2560, 9216), lin(2, 2560, 9216)
x = (mx.random.normal((R, d)) * 0.1).astype(mx.float16)
w1 = KN.decode_wr(l1._words, l1.K, l1._kt, l1._nt, l1.cb); mx.eval(x, w1)
dec2 = lambda: KN.decode_wr(l2._words, l2.K, l2._kt, l2._nt, l2.cb)
mm = lambda: x @ w1.T
def T(f):
    mx.eval(f()); ts = []
    for _ in range(9):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3
print("mm", T(lambda: [mm()]), "dec", T(lambda: [dec2()]), "both indep", T(lambda: [mm(), dec2()]), "both rev", T(lambda: [dec2(), mm()]))

"""ovcheck89.py: TQ decode attend bit-identity across the V-loop knobs (OV4 / OVU / VU / VH) for several lengths."""
import mlx.core as mx
from kyojin_mlx import tq
Hk, Hq, D = 4, 16, 256
mx.random.seed(5)
ok = True
for N in (40, 300, 520, 1100, 4096, 9000):
    c = tq.TQKVCache(6, 6, 4, 256)
    c.update_and_fetch(mx.random.normal((1, Hk, N, D)).astype(mx.float16), mx.random.normal((1, Hk, N, D)).astype(mx.float16))
    q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16)
    ref = None
    for ov4, ovu, vu, vh in ((0, 4, 2, 1), (1, 4, 2, 1), (1, 2, 2, 1), (1, 8, 2, 1), (1, 4, 4, 2), (1, 4, 1, 1)):
        tq.OV4, tq.OVU, tq.VU, tq.VH = ov4, ovu, vu, vh
        o = c.attend(q, D ** -0.5); mx.eval(o)
        if ref is None: ref = o
        eq = bool(mx.array_equal(ref, o)); ok &= eq
        if not eq: print("DIFF", N, ov4, ovu, vu, vh)
print("all bit-identical" if ok else "MISMATCH")

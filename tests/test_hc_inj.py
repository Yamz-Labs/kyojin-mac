"""KYOJIN_HC_INJ micro: fused inject+gate kernel vs stock linear + gate (tolerance, order differs), fire counter. GPU micro."""
import mlx.core as mx, mlx.nn as nn
from kyojin_mlx import hc_fuse as F

class M(nn.Module):
    def __init__(s, H=2560, HC=4, dt=mx.float32):
        super().__init__(); s.hc_count = HC; s.hidden_size = H
        s.block_inject_weight = nn.Linear(HC * H, HC, bias=False); s.block_inject_weight.weight = (mx.random.normal((HC, HC * H)) * 0.02).astype(dt)

def ref(m, n, HC):
    inj = m.block_inject_weight(n); dt = inj.dtype
    p = (inj / HC).astype(dt)
    return (2 * mx.sigmoid(p)).astype(dt)

for dt in (mx.float32, mx.float16):
    for rows in (1, 4):
        m = M(dt=dt); n = mx.random.normal((rows, 4 * 2560)).astype(mx.bfloat16)
        k = F._k("hcf_inj", F._INJ, ["x", "w"], ["out"])
        W = m.block_inject_weight.weight
        o = k(inputs=[n, W], template=[("T", n.dtype), ("TW", W.dtype), ("K", 10240), ("HC", 4)], grid=(4 * 256, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, 4)], output_dtypes=[W.dtype])[0]
        r = ref(m, n, 4); mx.eval(o, r)
        print(dt, rows, "maxabs", float(mx.abs(o.astype(mx.float32) - r.astype(mx.float32)).max()), "equal", bool(mx.array_equal(o, r)))
import time
m = M(); n = mx.random.normal((1, 10240)).astype(mx.bfloat16); W = m.block_inject_weight.weight
k = F._k("hcf_inj", F._INJ, ["x", "w"], ["out"])
def a(): return k(inputs=[n, W], template=[("T", n.dtype), ("TW", W.dtype), ("K", 10240), ("HC", 4)], grid=(1024, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, 4)], output_dtypes=[W.dtype])[0]
def b(): return ref(m, n, 4)
for nm, f in (("stock", b), ("inj", a)):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(20):
        mx.eval([f() for _ in range(70)])
    print(nm, "us/call", (time.perf_counter() - t) / 1400 * 1e6)

"""Does matmul2d accept a strided device tensor as A? K1 variant reading q as [H,T,D] directly. Bit-compare vs shipped K1."""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC, T = 64, 512, 640, 512, 2048
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
q = rnd(H, T, D); tab = rnd(6144, D)
ids = mx.array(np.stack([rng.choice(6144, NK, replace=False) for _ in range(T)]).astype(np.int32))
src = AP._K1.replace("tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(Q + ulong(t) * HH * DD), dextents<int32_t, 2>(DD, HH));",
   "tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(Q + ulong(t0 + t) * DD), dextents<int32_t, 2>(DD, HH), array<int32_t, 2>{1, TT * DD});")
src = src.replace("const uint t = threadgroup_position_in_grid.y;", "const uint t = threadgroup_position_in_grid.y, t0 = params[0];")
sub = lambda s: s.replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", "0.04f").replace("TT", str(T))
k_new = mx.fast.metal_kernel(name="exp_stride", header=NX.HEADER_NAX, input_names=["Q", "tab", "ids", "params"], output_names=["S"], source=sub(src))
k_old = mx.fast.metal_kernel(name="exp_stride_old", header=NX.HEADER_NAX, input_names=["Q", "tab", "ids"], output_names=["S"], source=sub(AP._K1))
a = 512
qt = mx.contiguous(q.transpose(1, 0, 2))
fo = lambda: k_old(inputs=[qt[a:a + TC], tab, ids[a:a + TC]], grid=(5 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]
fn = lambda: k_new(inputs=[q, tab, ids[a:a + TC], mx.array([a], dtype=mx.uint32)], grid=(5 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]
o = fo(); n = fn(); mx.eval(o, n)
print("maxdiff", float(mx.max(mx.abs(o - n)).item()))
best = {"old": 1e9, "new": 1e9}
for r in range(6):
    time.sleep(1.0)
    for k, f in (("old", fo), ("new", fn)) if r % 2 == 0 else (("new", fn), ("old", fo)):
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(f()); best[k] = min(best[k], time.perf_counter() - t0)
print({k: round(v * 1e3, 2) for k, v in best.items()})

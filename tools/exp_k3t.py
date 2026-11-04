"""K3 TG-staged variants: key step KS (32/64), A tile. Bit-compare vs shipped."""
import time, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC = 64, 512, 640, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
tab = rnd(6144, D)
ids_np = np.stack([rng.choice(6144, NK, replace=False) for _ in range(TC)]).astype(np.int32); ids_np[:, 5::37] = -1
ids = mx.array(ids_np)
P = mx.softmax(mx.array(rng.standard_normal((TC * H, NK)).astype(np.float32) * 3), axis=-1).astype(mx.float16)
P = mx.where(mx.repeat(ids >= 0, H, axis=0), P, 0).astype(mx.float16)
par = mx.array([TC], dtype=mx.uint32)
def variant(ks):
    s = AP._K3.replace("threadgroup half stage[32 * 128];", f"threadgroup half stage[{ks} * 128];")
    s = s.replace("matmul2d_descriptor(HH, 32, 32,", f"matmul2d_descriptor(HH, 32, {ks},").replace("dextents<int32_t, 2>(128, 32)", f"dextents<int32_t, 2>(128, {ks})")
    s = s.replace("kt += 32", f"kt += {ks}").replace("for (uint i = 0; i < 4; ++i) {\n            const uint c = tid + 128 * i, r = c >> 4", f"for (uint i = 0; i < {ks // 8}; ++i) {{\n            const uint c = tid + 128 * i, r = c >> 4")
    s = s.replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK))
    return mx.fast.metal_kernel(name=f"exp_k3t_{ks}", header=NX.HEADER_NAX, input_names=["P", "tab", "ids", "params"], output_names=["O"], source=s)
mk = lambda k: (lambda: k(inputs=[P, tab, ids, par], grid=(4 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(H, TC, D)], output_dtypes=[mx.float16])[0])
arms = {"old": mk(AP._kernels(H, D, NK, 0.1)[2]), "ks16": mk(variant(16)), "ks64": mk(variant(64))}
ref = arms["old"](); mx.eval(ref)
for k, f in arms.items(): o = f(); mx.eval(o); print(k, "equal", bool(mx.array_equal(o, ref).item()))
best = {k: 1e9 for k in arms}
for r in range(8):
    time.sleep(1.0)
    ks = list(arms); ks = ks[r % 3:] + ks[:r % 3]
    for k in ks:
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
fl = 2 * H * NK * D * TC
for k, v in best.items(): print(f"{k:5s} {v*1e3:.2f} ms {fl/v/1e12:.1f} TF")

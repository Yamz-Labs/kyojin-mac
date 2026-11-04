"""Stage split of the decode+matmul dense path at T rows: rot, decode_wr, matmul, fin; chained (one eval) vs separate evals. usage: exp_dense_gap.py PACK T"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, exl3_linear as EL, kernels as KN
pack, T = sys.argv[1], int(sys.argv[2])
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
L = model.language_model.layers[0]
EL.DENSE_FUSED_MAX = 0
mods = {"wq_a": L.attn.wq_a, "wq_b": L.attn.wq_b, "wkv": L.attn.wkv, "wo_b": L.attn.wo_b, "wo_a.g0": L.attn.wo_a.groups[0]}
rng = np.random.default_rng(0)
def tm(fn, reps=6):
    mx.eval(fn()); best = 1e9
    for _ in range(reps):
        time.sleep(1.0)
        for _ in range(3):
            t0 = time.perf_counter(); mx.eval(fn()); best = min(best, time.perf_counter() - t0)
    return best * 1e3
for name, m in mods.items():
    x = mx.array(rng.standard_normal((T, m.in_features)).astype(np.float16)); mx.eval(x)
    fl = 2 * T * m.in_features * m.out_features
    xh = KN.rot(x, m._suh, mx.float16); mx.eval(xh)
    wr = KN.decode_wr(m._words, m.K, m._kt, m._nt, m.cb); mx.eval(wr)
    inner = xh @ wr.T; mx.eval(inner)
    res = {}
    res["full"] = tm(lambda: m._dense(x))
    res["rot"] = tm(lambda: KN.rot(x, m._suh, mx.float16))
    res["decode"] = tm(lambda: KN.decode_wr(m._words, m.K, m._kt, m._nt, m.cb))
    res["matmul"] = tm(lambda: xh @ wr.T)
    fp = mx.array([1, m.out_features, T], dtype=mx.uint32); grid, tg = KN.S.had_grid(m.out_features // 128, T)
    res["fin"] = tm(lambda: KN.fin_kernel(m.out_dtype, mx.float16)(inputs=[inner.reshape(1, T, -1), m._svh, fp], template=[("OT", m.out_dtype)], grid=grid, threadgroup=tg, output_shapes=[(T, m.out_features)], output_dtypes=[m.out_dtype])[0])
    s = sum(res[k] for k in ("rot", "decode", "matmul", "fin"))
    print(f"{name:8s} T={T} {fl/1e9:.0f} GF  full {res['full']:.1f} ms ({fl/res['full']/1e9:.1f} TF) | " + " ".join(f"{k} {res[k]:.1f}" for k in ("rot","decode","matmul","fin")) + f" | sum {s:.1f} | matmul alone {fl/res['matmul']/1e9:.1f} TF", flush=True)

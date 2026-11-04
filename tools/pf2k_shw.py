"""Fused dense linear with the finalize in the epilogue (metal_nax_shw) vs SH + fin kernel. usage: pf2k_shw.py kin,nout,K R [sg_rows,nq,nsg;...] [reps]"""
import sys, time, mlx.core as mx, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX, metal_nax_sh as SH, metal_nax_shw as SHW
kin, nout, K = map(int, sys.argv[1].split(",")); R = int(sys.argv[2])
cfgs = [tuple(int(v) if v.lstrip("-").isdigit() else v for v in c.split(",")) for c in (sys.argv[3] if len(sys.argv) > 3 else "2,4").split(";") if c]
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 9
rng = np.random.default_rng(1)
kt, nt = kin // 16, nout // 16
tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
lin = EL.Exl3Linear(tr, f16(kin), f16(nout), cb="mul1")
x = (mx.random.normal((R, kin)) * 0.5).astype(mx.float16)
lin._nax = (NX.SingleStack(lin._words, lin.K, lin._kt, lin._nt, lin.cb), mx.array([0], dtype=mx.int32)); st = lin._nax[0]
grid, tg = KN.S.had_grid(nout // 128, R); fp = mx.array([1, nout, R], dtype=mx.uint32)
def ref():
    xh = KN.rot(x, lin._suh, mx.float16, perm=True)
    inner = SH.dense_sh(st, xh, R, (64, 1), True, "", 2, False, 4).reshape(1, R, nout)
    return KN.fin_kernel(lin.out_dtype, mx.float16)(inputs=[inner, lin._svh, fp], template=[("OT", lin.out_dtype)], grid=grid, threadgroup=tg, output_shapes=[(R, nout)], output_dtypes=[lin.out_dtype])[0]
def real():
    wr = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)
    inner = ((KN.rot(x, lin._suh, mx.float16)) @ wr.T).reshape(1, R, nout)
    return KN.fin_kernel(lin.out_dtype, mx.float16)(inputs=[inner, lin._svh, fp], template=[("OT", lin.out_dtype)], grid=grid, threadgroup=tg, output_shapes=[(R, nout)], output_dtypes=[lin.out_dtype])[0]
wrc = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb); mx.eval(wrc)
def realc():
    inner = ((KN.rot(x, lin._suh, mx.float16)) @ wrc.T).reshape(1, R, nout)
    return KN.fin_kernel(lin.out_dtype, mx.float16)(inputs=[inner, lin._svh, fp], template=[("OT", lin.out_dtype)], grid=grid, threadgroup=tg, output_shapes=[(R, nout)], output_dtypes=[lin.out_dtype])[0]
arms = {"SH+fin": ref, "dec+mm+fin": real, "mm+fin(cached)": realc}
for c in cfgs:
    arms[f"SHW{c}"] = (lambda c=c: SHW.dense_shw(st, KN.rot(x, lin._suh, mx.float16, perm=True), R, lin._svh, lin.out_dtype, *c))
r0 = ref(); mx.eval(r0)
res = {}
for n, f in arms.items():
    try:
        o = f(); mx.eval(o); res[n] = []
        if n.startswith("SHW"):
            print(n, "maxdiff", mx.abs(o.astype(mx.float32) - r0.astype(mx.float32)).max().item(), "bitexact", bool(mx.array_equal(o, r0).item()), flush=True)
    except Exception as e:
        print(n, "FAIL", str(e)[:600], flush=True)
for _ in range(NR):
    for n in list(res):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(arms[n]()); res[n].append((time.perf_counter() - t0) * 1e3)
for n, v in res.items(): print(f"{kin}x{nout} R={R} {n:16s} min {min(v):.3f} med {sorted(v)[len(v)//2]:.3f} ms", flush=True)

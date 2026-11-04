"""Fused NAX dense linear at R rows: coop (per-simdgroup decode) cfg sweep and the decode-once shared kernel vs decode+matmul and matmul alone.
usage: pf512_cfg.py kin,nout,K R [cfg;cfg...]  cfg = bmu,nmb,bns,bks   (SH arms: bmu,nmb pairs of metal_nax_sh)"""
import sys, time, mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import kernels as KN, metal_nax as NX, metal_nax_sh as SH
def synth_linear(kin, nout, K, cb="mul1", seed=1):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)
_aux = {}
def nax_call(lin, x2, cfg, out_half=True):
    rows = x2.shape[0]
    if lin._nax is None:
        lin._nax = (NX.SingleStack(lin._words, lin.K, lin._kt, lin._nt, lin.cb), mx.array([0], dtype=mx.int32))
    st, eids = lin._nax
    ent = _aux.get((id(lin), rows))
    if ent is None:
        ent = _aux[(id(lin), rows)] = (mx.array([0, rows], dtype=mx.uint32), mx.array([1, lin.out_features, rows], dtype=mx.uint32), KN.S.had_grid(lin.out_features // 128, rows))
    rowoff, fp, (grid, tg) = ent
    xh = KN.rot(x2, lin._suh, mx.float16, perm=True)
    inner = NX.expert_gemm((st,), eids, rowoff, xh, rows, rows, cfg, out_half).reshape(1, rows, lin.out_features)
    return KN.fin_kernel(lin.out_dtype, mx.float16 if out_half else mx.float32)(
        inputs=[inner, lin._svh, fp], template=[("OT", lin.out_dtype)], grid=grid, threadgroup=tg,
        output_shapes=[(rows, lin.out_features)], output_dtypes=[lin.out_dtype])[0]
kin, nout, K = map(int, sys.argv[1].split(",")); R = int(sys.argv[2])
SHC = [tuple(c.split(",")) for c in (sys.argv[3] if len(sys.argv) > 3 else "32,2").split(";") if c]
SHC = [(int(c[0]), int(c[1]), int(c[2]) if len(c) > 2 else 1, c[3] if len(c) > 3 else "", len(c) > 4 and c[4] == "p", int(c[5]) if len(c) > 5 else 4, len(c) > 6 and c[6] == "c") for c in SHC]
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 15
lin = synth_linear(kin, nout, K)
x = (mx.random.normal((R, kin)) * 0.5).astype(mx.float16); mx.eval(x)
def T(f, n=8):
    mx.eval(f()); mx.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3
fl = 2 * R * kin * nout
tf = lambda ms: fl / ms / 1e9
wr = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb); mx.eval(wr)
st = lin._nax[0] if lin._nax else None
nax_call(lin, x, (32, 2, 32, 64)); st = lin._nax[0]
xh = KN.rot(x, lin._suh, mx.float16, perm=True)
r2 = NX.expert_gemm((st,), lin._nax[1], mx.array([0, R], dtype=mx.uint32), xh, R, R, (32, 2, 32, 64)); mx.eval(r2)
arms = {"mm": lambda: x @ wr.T, "dec+mm": lambda: x @ KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb).T,
        "coop": lambda: NX.expert_gemm((st,), lin._nax[1], mx.array([0, R], dtype=mx.uint32), xh, R, R, (32, 2, 32, 64))}
for c in SHC:
    arms[f"SH{c}"] = (lambda c=c: SH.dense_sh(st, xh, R, c[:2], True, c[3], c[2], c[4], c[5]))
res = {}
for n, f in arms.items():
    try:
        mx.eval(f()); res[n] = []
    except Exception as e:
        print(n, "FAIL", str(e)[:300], flush=True)
for _ in range(NR):
    for n in list(res):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(arms[n]()); res[n].append((time.perf_counter() - t0) * 1e3)
for n in res:
    v = sorted(res[n]); d = ""
    if n.startswith("SH"):
        d = f"  maxdiff vs coop {mx.abs(arms[n]().astype(mx.float32) - r2.astype(mx.float32)).max().item():.3g}"
    print(f"{kin}x{nout} K{K} R={R} {n:14s} min {v[0]:.3f} med {v[len(v)//2]:.3f} ms  {tf(v[0]):.1f} TF{d}", flush=True)

"""Gate: the fp8 simdgroup-matrix kernel (foreign_metal.mma8_source, float / half A, NTN 1 / 2, 32-row slices) against fp16 matmul and dequant kernel + matmul, Gw/s on 8 x 8192^2. Args: row counts. Run under /tmp/lane_ab.lock."""
import os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mlx.core as mx
from kyojin_mlx import foreign_linear as FL, foreign_metal as FM
KIN, N, COUNT, REPS = 8192, 8192, 8, 7
ROWS = tuple(int(a) for a in sys.argv[1:]) or (16, 24, 32, 64)
TAB = np.array(FM.E4M3_TABLE, np.float32)
rng = np.random.default_rng(1)
order = np.argsort(TAB[:127])
idx = np.clip(np.searchsorted(np.sort(TAB[:127]), np.abs(rng.normal(0, 0.05, (N, KIN))).astype(np.float32)), 0, 126)
codes = (order[idx] | (rng.integers(0, 2, (N, KIN)) << 7)).astype(np.uint8)
W8s, W16s = [], []
for m in range(COUNT):
    c = np.ascontiguousarray(np.roll(codes, m * 37, axis=0))
    W8s.append(mx.array(c.view(np.uint32).reshape(N, KIN // 4))); W16s.append(mx.array(TAB[c].astype(np.float16)))
mx.eval(W8s, W16s)
gs = mx.ones((N,), mx.float32)
PERM = mx.array(FM.xperm8_index(KIN))
def deq(q):
    return FL._deq8_kern("fp8")(inputs=[q, gs], template=[("KIN", KIN), ("N", N)], grid=(N * KIN // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, KIN)], output_dtypes=[mx.float16])[0]
def mma(x, q, sl, ntn, ha):
    k = KY(ntn, ha); outs = []
    xp = x[:, PERM]
    for i in range(0, x.shape[0], sl):
        xs = xp[i:i + sl]; R = xs.shape[0]
        outs.append(k(inputs=[xs, q, gs], template=[("R", R), ("KIN", KIN), ("N", N), ("NTN", ntn)], grid=(-(-N // (32 * ntn)) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(R, N)], output_dtypes=[mx.float16])[0])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, 0)
_k = {}
def KY(ntn, ha):
    if (ntn, ha) not in _k:
        _k[(ntn, ha)] = mx.fast.metal_kernel(name=f"mma8t_{ntn}_{int(ha)}", input_names=["xp", "qs", "gs"], output_names=["y"], header=FM.header8("fp8"), source=FM.mma8_source("fp8", ntn, ha))
    return _k[(ntn, ha)]
for R in ROWS:
    x = mx.array(rng.normal(0, 1, (R, KIN)).astype(np.float16))
    ref = x.astype(mx.float32) @ mx.array(TAB[codes]).T; mx.eval(x, ref)
    q0 = mx.array(np.ascontiguousarray(codes).view(np.uint32).reshape(N, KIN // 4))
    arms = {"fp16": lambda m: (x @ W16s[m].T).astype(mx.float16), "deq_mm": lambda m: (x @ deq(W8s[m]).T).astype(mx.float16)}
    for sl in sorted({min(R, 32)}):
        for ntn in (1, 2):
            for ha in (False, True):
                nm = f"mma8_s{sl}_t{ntn}_{'h' if ha else 'f'}"
                e = mma(x, q0, sl, ntn, ha); mx.eval(e)
                print(f"R{R} {nm} rel err {float(mx.max(mx.abs(e.astype(mx.float32) - ref))) / float(mx.max(mx.abs(ref))):.5f}", flush=True)
                arms[nm] = (lambda sl, ntn, ha: lambda m: mma(x, W8s[m], sl, ntn, ha))(sl, ntn, ha)
    res = {a: [] for a in arms}
    for rep in range(REPS):
        for a, f in arms.items():
            t = time.perf_counter(); mx.eval([f(m) for m in range(COUNT)]); res[a].append(time.perf_counter() - t)
    w = COUNT * N * KIN * R
    print(f"R{R} " + " | ".join(f"{a} {w / st.median(v[1:]) / 1e9:.0f}" for a, v in res.items()), flush=True)

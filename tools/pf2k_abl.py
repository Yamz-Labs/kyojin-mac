"""Whole-model prefill ablation on the decode + MLX matmul path (rows > DENSE_FUSED_MAX): normal | nofin | norot | none | nodec (weights cached across reps).
usage: pf2k_abl.py MODEL_DIR rows"""
import sys, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, wcache as WC
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
FL = {"fin": True, "rot": True}
def dense(self, x2):
    rows = x2.shape[0]
    xh = KN.rot(x2, self._suh, mx.float16) if FL["rot"] else x2.astype(mx.float16)
    wr = WC.decode_wr(self)
    inner = (xh @ wr.T).reshape(1, rows, self.out_features)
    if not FL["fin"]:
        return inner
    fp = mx.array([1, self.out_features, rows], dtype=mx.uint32)
    grid, tg = KN.S.had_grid(self.out_features // 128, rows)
    return KN.fin_kernel(self.out_dtype, mx.float16)(inputs=[inner, self._svh, fp], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
                                                     output_shapes=[(rows, self.out_features)], output_dtypes=[self.out_dtype])[0]
EL.Exl3Linear._dense = dense
def go(scope):
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
    with WC.prefill_scope(be.model, R, 2 if scope else 1):
        be.prefill_chunk(c, ids)
    return time.perf_counter() - t0
arms = {"normal": (True, True, False), "nofin": (False, True, False), "norot": (True, False, False), "none": (False, False, False)}
res = {n: [] for n in arms}
go(False)
for _ in range(5):
    for n, (f, r, s) in arms.items():
        FL["fin"], FL["rot"] = f, r; res[n].append(go(False))
# decode share: time the decode of every linear alone
lins = WC._linears(be.model)
def dec():
    for m in lins: mx.eval(KN.decode_wr(m._words, m._kt and m.K, m._kt, m._nt, m.cb))
dec(); ts = []
for _ in range(3):
    mx.synchronize(); t0 = time.perf_counter(); dec(); ts.append(time.perf_counter() - t0)
for n, v in res.items(): print(f"{n:7s} {min(v)*1e3:7.1f} ms  {R/min(v):7.0f} tok/s", flush=True)
print(f"decode of {len(lins)} linears: {min(ts)*1e3:.1f} ms")

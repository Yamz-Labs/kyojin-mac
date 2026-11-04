"""Whole-model prefill ablation of the Hadamard passes on the SH path (served call prefill_chunk): normal | nofin | norot | none. usage: abl.py MODEL_DIR rows"""
import sys, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX, metal_nax_sh as NSH
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
orig = EL.Exl3Linear._dense
FLAGS = {"fin": True, "rot": True}
def dense(self, x2):
    rows = x2.shape[0]
    if not (self._fused_ok(rows) and self._kt % 8 == 0 and self._ks_for(rows) == 1):
        return orig(self, x2)
    if self._nax is None:
        self._nax = (NX.SingleStack(self._words, self.K, self._kt, self._nt, self.cb), mx.array([0], dtype=mx.int32))
    ent = self._nax_rows.get(rows)
    if ent is None:
        self._inner_sh(x2); ent = self._nax_rows[rows]
    xh = KN.rot(x2, self._suh, mx.float16, perm=True) if FLAGS["rot"] else x2.astype(mx.float16)
    inner = NSH.dense_sh(self._nax[0], xh, rows, (64, 1), True, "", 2, False, 4)
    if not FLAGS["fin"]:
        return inner
    fp, (grid, tg) = ent[2:4]
    return KN.fin_kernel(self.out_dtype, mx.float16)(inputs=[inner.reshape(1, rows, self.out_features), self._svh, fp], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
                                                     output_shapes=[(rows, self.out_features)], output_dtypes=[self.out_dtype])[0]
EL.Exl3Linear._dense = dense
def go():
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter(); be.prefill_chunk(c, ids); return time.perf_counter() - t0
res = {}
for name, (f, r) in {"normal": (True, True), "nofin": (False, True), "norot": (True, False), "none": (False, False)}.items():
    FLAGS["fin"], FLAGS["rot"] = f, r
    try:
        go(); go()
    except Exception as e:
        print(name, "FAIL", str(e)[:200]); continue
    res[name] = []
for _ in range(5):
    for name, (f, r) in {"normal": (True, True), "nofin": (False, True), "norot": (True, False), "none": (False, False)}.items():
        if name in res:
            FLAGS["fin"], FLAGS["rot"] = f, r; res[name].append(go())
for n, v in res.items(): print(f"{n:7s} {min(v)*1e3:7.1f} ms  {R/min(v):7.0f} tok/s", flush=True)

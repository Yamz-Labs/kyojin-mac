"""Whole-model 2048-row prefill: normal vs decode_wr replaced by one cached fp16 weight per (kin, nout, dtype) shape (timing only) vs also no fin / no rot. usage: pf2k_abl2.py MODEL_DIR rows"""
import sys, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, wcache as WC
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
FL = {"fin": True, "rot": True, "dec": True}
shape_w = {}
def dense(self, x2):
    rows = x2.shape[0]
    xh = KN.rot(x2, self._suh, mx.float16) if FL["rot"] else x2.astype(mx.float16)
    if FL["dec"]:
        wr = WC.decode_wr(self)
    else:
        key = (self.in_features, self.out_features)
        wr = shape_w.get(key)
        if wr is None:
            wr = shape_w[key] = KN.decode_wr(self._words, self.K, self._kt, self._nt, self.cb); mx.eval(wr)
    inner = (xh @ wr.T).reshape(1, rows, self.out_features)
    if not FL["fin"]:
        return inner
    fp = mx.array([1, self.out_features, rows], dtype=mx.uint32)
    grid, tg = KN.S.had_grid(self.out_features // 128, rows)
    return KN.fin_kernel(self.out_dtype, mx.float16)(inputs=[inner, self._svh, fp], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
                                                     output_shapes=[(rows, self.out_features)], output_dtypes=[self.out_dtype])[0]
EL.Exl3Linear._dense = dense
def go():
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter(); be.prefill_chunk(c, ids); return time.perf_counter() - t0
arms = {"normal": (True, True, True), "nodec": (True, True, False), "nodec+nofin": (False, True, False), "nodec+nofin+norot": (False, False, False)}
res = {n: [] for n in arms}
for n, (f, r, d) in arms.items():
    FL.update(fin=f, rot=r, dec=d); go()
for _ in range(5):
    for n, (f, r, d) in arms.items():
        FL.update(fin=f, rot=r, dec=d); res[n].append(go())
for n, v in res.items(): print(f"{n:18s} {min(v)*1e3:7.1f} ms  {R/min(v):7.0f} tok/s", flush=True)

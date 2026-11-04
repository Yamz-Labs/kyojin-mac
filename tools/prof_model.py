"""Whole-model prefill ablation (Mac). usage: prof_model.py MODEL_DIR [rows=512]
modes: normal | dense (decoded fp16 weights cached once: matmul floor) | nodec (skip trellis decode, reuse one cached Wr)
       | norot (skip rot+finalize Hadamards)"""
import sys, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import kernels as KN, exl3_linear as EL
from transformers import AutoTokenizer
d = sys.argv[1]; P = int(sys.argv[2]) if len(sys.argv) > 2 else 512
model, _ = EL.load_exl3(d)
ids = AutoTokenizer.from_pretrained(d).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400)[:P]
x = mx.array(ids)[None]
def run():
    cache = make_prompt_cache(model)
    t0 = time.perf_counter(); lg = model(x, cache=cache)[:, -1]; mx.eval(lg); return time.perf_counter() - t0
def best(n=3):
    run(); return min(run() for _ in range(n))
orig = EL.Exl3Linear._dense
t = best(); print(f"normal      {t*1e3:7.1f} ms  {P/t:7.1f} tok/s", flush=True)
# dense: cache decoded Wr per module
def dense_cached(self, x2):
    if not hasattr(self, "_wrc"):
        self._wrc = KN.decode_wr(self._words, self.K, self._kt, self._nt, self.cb); mx.eval(self._wrc)
    rows = x2.shape[0]
    xh = KN.rot(x2, self._suh, mx.float16)
    inner = (xh @ self._wrc.T).reshape(1, rows, self.out_features)
    fp = mx.array([1, self.out_features, rows], dtype=mx.uint32)
    grid, tg = KN.S.had_grid(self.out_features // 128, rows)
    return KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, self._svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg,
                 output_shapes=[(rows, self.out_features)], output_dtypes=[mx.float16])[0]
EL.Exl3Linear._dense = dense_cached
t = best(); print(f"dense-cache {t*1e3:7.1f} ms  {P/t:7.1f} tok/s", flush=True)
def norot(self, x2):
    rows = x2.shape[0]
    return (x2.astype(mx.float16) @ self._wrc.T)
EL.Exl3Linear._dense = norot
t = best(); print(f"matmul only {t*1e3:7.1f} ms  {P/t:7.1f} tok/s   (no Hadamards: lower bound for any dequant-GEMM design)", flush=True)

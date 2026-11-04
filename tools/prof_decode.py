"""Decode (1 row) ablation on the whole model. usage: prof_decode.py MODEL_DIR [steps=64]
normal | norot (skip input Hadamard kernel) | nofin (skip finalize kernel: sum of partials) | nolin (EXL3 linears return zeros)"""
import sys, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import kernels as KN, exl3_linear as EL
from transformers import AutoTokenizer
d = sys.argv[1]; D = int(sys.argv[2]) if len(sys.argv) > 2 else 64
model, _ = EL.load_exl3(d)
ids = AutoTokenizer.from_pretrained(d).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 40)[:128]
x = mx.array(ids)[None]
def run():
    cache = make_prompt_cache(model); lg = model(x, cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t)
    t0 = time.perf_counter()
    for _ in range(D):
        lg = model(t[:, None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t)
    return (time.perf_counter() - t0) / D * 1e3
def best(): run(); return min(run() for _ in range(3))
orig_call = EL.Exl3Linear.__call__
def report(name, t): print(f"{name:8s} {t:6.2f} ms/token  {1e3/t:6.1f} tok/s", flush=True)
report("normal", best())
def mk(skip_rot=False, skip_fin=False, zeros=False):
    def call(self, x):
        shape = x.shape; x2 = x.reshape(-1, shape[-1])
        if x2.dtype not in (mx.float16, mx.float32): x2 = x2.astype(mx.float16)
        if x2.shape[0] > EL.GEMV_MAX_ROWS: return orig_call(self, x)
        if zeros: return mx.zeros((*shape[:-1], self.out_features), dtype=mx.float16) + x2[..., :1].reshape(*shape[:-1], 1)
        g = self._gemv(x2.shape[0])
        xh = x2.astype(mx.float32) if skip_rot else KN.rot(x2, g.suh)
        part = g.partial(xh)
        y = part[0].astype(mx.float16) if skip_fin else g.finalize(part).astype(mx.float16)
        return y.reshape(*shape[:-1], self.out_features)
    return call
for name, kw in (("norot", dict(skip_rot=True)), ("nofin", dict(skip_fin=True)), ("nolin", dict(zeros=True))):
    EL.Exl3Linear.__call__ = mk(**kw); report(name, best())
EL.Exl3Linear.__call__ = mk(skip_rot=True, skip_fin=True); report("partial", best())

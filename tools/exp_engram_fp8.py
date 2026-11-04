"""Is the loaded Engram wkv (fp16 in the model) exactly the FP8 e4m3 x E8M0 block-scale dequant of the pack? usage: exp_engram_fp8.py PACK"""
import sys, json, struct
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R
pack = sys.argv[1]
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
en = [l.engram for l in lm.layers if l.engram is not None][0]
w = en.wkv
print(type(w).__name__, [ (k, type(v).__name__, getattr(v, 'dtype', None), getattr(v, 'shape', None)) for k, v in vars(w).items() if isinstance(v, mx.array)][:6])
W = w.weight if hasattr(w, "weight") else w.inner.weight
print("loaded", W.dtype, W.shape)
t = P.load_pack(pack)
f8 = np.array(t["layers.1.engram.wkv.weight"].view(mx.uint8)).reshape(25600, 6144) if t["layers.1.engram.wkv.weight"].dtype != mx.uint8 else np.array(t["layers.1.engram.wkv.weight"])
print("pack tensor", t["layers.1.engram.wkv.weight"].dtype, t["layers.1.engram.wkv.weight"].shape, t["layers.1.engram.wkv.scale"].dtype)
sc = np.array(t["layers.1.engram.wkv.scale"].view(mx.uint8)).reshape(800, 192)
def e4m3(b):
    s = (b >> 7) & 1; e = (b >> 3) & 15; m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(s == 1, -v, v)
vals = e4m3(f8.astype(np.int64))
scale = 2.0 ** (sc.astype(np.float64) - 127)
full = vals * np.repeat(np.repeat(scale, 32, axis=0), 32, axis=1)
Wf = np.array(W.astype(mx.float32))
d = np.abs(full - Wf)
print("max abs diff vs numpy dequant", d.max(), "equal fraction", (full == Wf).mean(), "nan", np.isnan(full).sum())
h16 = full.astype(np.float16).astype(np.float64)
print("fp16-exact (dequant representable in fp16):", (h16 == full).mean(), " range", np.nanmin(full), np.nanmax(full))

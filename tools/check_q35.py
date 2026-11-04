"""Exactness check on the Mac against the box CPU reference (ref/fixtures.npz, ref/layers.npz, ref/ref_tokens.json).
usage: check_q35.py MODEL_DIR REF_DIR"""
import sys, json
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx.exl3_linear import Exl3Linear, load_exl3
model_dir, ref = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{model_dir}/config.json")); cb = cfg["quantization_config"]["codebook"]
w = mx.load(f"{model_dir}/model.safetensors")
fx = np.load(f"{ref}/fixtures.npz")
print("== per linear: rel err vs float64 reference (rows 1/4 = Metal GEMV, rows 8 = decode+matmul)")
for k in sorted({n.split("|")[0] for n in fx.files}):
    x, y = fx[k + "|x"], fx[k + "|y"]
    lin = Exl3Linear(w[k + ".trellis"], w[k + ".suh"], w[k + ".svh"], cb=cb)
    errs = []
    for r in (1, 4, 8):
        o = np.array(lin(mx.array(x[:r])).astype(mx.float32))
        errs.append(float(np.linalg.norm(o - y[:r]) / np.linalg.norm(y[:r])))
    print(f"{k:60s} K={lin.K} " + " ".join(f"{e:.2e}" for e in errs))
print("== per decoder layer (prompt, fp16 MLX vs fp32 torch CPU on dequantized weights)")
model, _ = load_exl3(model_dir, cb=cb)
lm = model.language_model
rec = {}
class Rec(nn.Module):
    def __init__(self, l, i):
        super().__init__(); self.l, self.i, self.is_linear = l, i, l.is_linear
    def __call__(self, x, mask=None, cache=None):
        o = self.l(x, mask=mask, cache=cache); rec[self.i] = o; return o
for i, l in enumerate(lm.model.layers): lm.model.layers[i] = Rec(l, i)
lay = np.load(f"{ref}/layers.npz")
ids = json.load(open(f"{ref}/ref_tokens.json"))[0]["prompt_ids"]
cache = make_prompt_cache(model)
lg = model(mx.array(ids)[None], cache=cache)[0, -1]
mx.eval(lg, *rec.values())
for i in range(len(rec)):
    a, b = np.array(rec[i][0].astype(mx.float32)), lay[f"h{i}"]
    print(f"layer {i:2d} rel {np.linalg.norm(a-b)/np.linalg.norm(b):.2e} maxabs {np.abs(a-b).max():.2e}")
a, b = np.array(lg.astype(mx.float32)), lay["logits"]
print("logits rel", np.linalg.norm(a - b) / np.linalg.norm(b), "argmax", a.argmax(), b.argmax(), "top5 overlap", len(set(a.argsort()[-5:]) & set(b.argsort()[-5:])))

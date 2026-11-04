"""Per-layer-type prefill timing with an eval after each layer. usage: prof_layers.py exl3|mlx MODEL_DIR [rows=512]"""
import sys, time, collections
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
kind, d = sys.argv[1], sys.argv[2]; P = int(sys.argv[3]) if len(sys.argv) > 3 else 512
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(d)
else:
    from mlx_lm import load
    model, _ = load(d)
ids = AutoTokenizer.from_pretrained(d).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400)[:P]
x = mx.array(ids)[None]
lm = model.language_model
acc = collections.defaultdict(float)
class T(nn.Module):
    def __init__(self, l, name):
        super().__init__(); self.l, self.name, self.is_linear = l, name, l.is_linear
    def __call__(self, h, mask=None, cache=None):
        mx.eval(h); t0 = time.perf_counter()
        o = self.l(h, mask=mask, cache=cache); mx.eval(o); acc[self.name] += time.perf_counter() - t0; return o
for i, l in enumerate(lm.model.layers): lm.model.layers[i] = T(l, "gdn-layer" if l.is_linear else "attn-layer")
def run():
    cache = make_prompt_cache(model); acc.clear()
    t0 = time.perf_counter(); h = lm.model(x, cache=cache); mx.eval(h); t1 = time.perf_counter()
    lg = lm.lm_head(h) if hasattr(lm, "lm_head") else None; mx.eval(lg); t2 = time.perf_counter()
    return t1 - t0, t2 - t1
run(); r = [run() for _ in range(3)]; (tb, tl) = min(r)
print(kind, "body %.1f ms  lm_head %.1f ms" % (tb * 1e3, tl * 1e3), {k: round(v * 1e3 / 3, 1) for k, v in acc.items()}, "(per-type ms; last 3 runs mean-ish)")

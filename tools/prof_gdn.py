"""Sub-op timing inside the first GDN layers during prefill. usage: prof_gdn.py exl3|mlx MODEL_DIR [rows=512]"""
import sys, time, collections
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import qwen3_5 as Q
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
    def __init__(self, m, name):
        super().__init__(); self.m, self.name = m, name
    def __call__(self, *a, **k):
        mx.eval(a[0]); t0 = time.perf_counter(); o = self.m(*a, **k); mx.eval(o); acc[self.name] += time.perf_counter() - t0; return o
for l in lm.model.layers:
    if l.is_linear:
        g = l.linear_attn
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "norm", "out_proj"):
            setattr(g, n, T(getattr(g, n), n))
orig = Q.gated_delta_update
def gdu(*a, **k):
    mx.eval(*[t for t in a if isinstance(t, mx.array)]); t0 = time.perf_counter(); o = orig(*a, **k); mx.eval(o); acc["gated_delta_update"] += time.perf_counter() - t0; return o
Q.gated_delta_update = gdu
for l in lm.model.layers:
    if not l.is_linear: continue
    mlp = l.mlp
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, n, T(getattr(mlp, n), "mlp." + n))
def run():
    cache = make_prompt_cache(model); acc.clear()
    h = lm.model(x, cache=cache); mx.eval(h)
run(); run()
print(kind, {k: round(v * 1e3, 1) for k, v in acc.items()}, "ms over 18 GDN layers")
a0 = lm.model.layers[0].linear_attn
for n in ("A_log", "dt_bias"): print(n, getattr(a0, n).dtype, end="  ")
print("in_proj_b", type(a0.in_proj_b.m).__name__)

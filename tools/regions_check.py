"""Bit check of kyojin_mlx.step_regions: N greedy decode steps with the stock forward vs the compiled regions, same prompt; logits compared bit for bit
at every step, md5 of the token stream printed. usage: regions_check.py exl3|mlx MODEL_DIR [--steps 200] [--ctx 512]"""
import argparse, hashlib, os, sys
os.environ["KYOJIN_STEP_REGIONS"] = "0"        # load stock; the second run enables the regions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx import step_regions as SR
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("path"); ap.add_argument("--steps", type=int, default=200); ap.add_argument("--ctx", type=int, default=512)
a = ap.parse_args()
if a.kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(a.path)
else:
    from mlx_lm import load
    model, _ = load(a.path)
ids = AutoTokenizer.from_pretrained(a.path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 100)[:a.ctx]
def run(on):
    tm = model.language_model.model
    if on: SR.enable(model)
    cache = make_prompt_cache(model)
    lg = model(mx.array(ids)[None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t)
    toks, lgs = [], []
    for _ in range(a.steps):
        lg = model(t[:, None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t, lg)
        toks.append(int(t.item())); lgs.append(np.array(lg.astype(mx.float32)))
    return toks, lgs
SR.ENABLED = True
ref_t, ref_l = run(False)
on_t, on_l = run(True)
bad = sum(not np.array_equal(x, y) for x, y in zip(ref_l, on_l))
print("tokens equal", ref_t == on_t, "md5", hashlib.md5(str(ref_t).encode()).hexdigest()[:12], hashlib.md5(str(on_t).encode()).hexdigest()[:12], "logit-steps differing", bad, "/", a.steps, SR.STATS)

"""Greedy generation with an EXL3 pack on MLX. Usage: gen_exl3.py MODEL_DIR [--n 64] [--prompt TEXT] [--dump OUT.json]"""
import argparse, json, sys, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx.exl3_linear import load_exl3

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--prompt", default="The history of the Roman Empire begins with")
ap.add_argument("--dump"); a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok.encode(a.prompt)
t0 = time.time(); model, cfg = load_exl3(a.model); print("load s", round(time.time() - t0, 1), flush=True)
cache = make_prompt_cache(model)
x = mx.array(ids)[None]
logits = model(x, cache=cache)[:, -1]
out = []; last = []
for i in range(a.n):
    t = mx.argmax(logits, axis=-1); mx.eval(t); out.append(int(t.item()))
    logits = model(t[:, None], cache=cache)[:, -1]
print("prompt_ids", ids); print("tokens", out); print(repr(tok.decode(out)))
if a.dump: json.dump({"prompt_ids": ids, "tokens": out}, open(a.dump, "w"))

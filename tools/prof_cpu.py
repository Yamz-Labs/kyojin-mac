"""CPU-side cost of one decode step: time to build the lazy graph (no eval) vs full step. usage: prof_cpu.py exl3|mlx MODEL_DIR"""
import sys, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
kind, d = sys.argv[1], sys.argv[2]
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(d)
else:
    from mlx_lm import load
    model, _ = load(d)
ids = AutoTokenizer.from_pretrained(d).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 40)[:128]
cache = make_prompt_cache(model); lg = model(mx.array(ids)[None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t)
for _ in range(5): t = mx.argmax(model(t[:, None], cache=cache)[:, -1], axis=-1); mx.eval(t)
bs, fs = [], []
for _ in range(20):
    t0 = time.perf_counter(); nt = mx.argmax(model(t[:, None], cache=cache)[:, -1], axis=-1); t1 = time.perf_counter()
    mx.eval(nt); t2 = time.perf_counter(); t = nt; bs.append(t1 - t0); fs.append(t2 - t1)
bs.sort(); fs.sort()
print(f"{kind}: graph build (python+C++ op creation) {bs[10]*1e3:.2f} ms/token, eval (encode+GPU) {fs[10]*1e3:.2f} ms/token")

"""Forward time of the trunk of an mlx-lm Qwen3.5 pack at S rows (synced), per component: whole model, GDN layers only, attention layers only, head. usage: mlxrows_prof.py PACK"""
import sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, _ = load(sys.argv[1]); lm = model.language_model; tm = lm.model
def timeit(f, n=6):
    f(); f()
    t0 = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t0) / n * 1e3
cache = make_prompt_cache(model)
mx.eval(tm(mx.arange(512)[None] % 1000, cache))
for S in (1, 2, 4, 6):
    x = (mx.arange(S)[None] + 5)
    def fwd():
        for c in cache:
            if hasattr(c, "trim") and c.offset > 512: c.trim(c.offset - 512)
        mx.eval(tm(x, cache))
    # restore: only attention caches trimmed; gdn states drift but cost is same
    t = timeit(fwd)
    h = mx.random.normal((1, S, tm.args.hidden_size)).astype(mx.float16) if hasattr(tm, "args") else mx.random.normal((1, S, 2560)).astype(mx.float16)
    th = timeit(lambda: mx.eval(tm.embed_tokens.as_linear(h)))
    print(f"S={S} trunk {t:6.1f} ms  head {th:5.1f} ms", flush=True)

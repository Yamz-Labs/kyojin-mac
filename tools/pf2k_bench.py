"""Whole-model prefill tok/s, exl3 or mlx-lm, chunked. usage: pf2k_bench.py exl3|mlx MODEL_DIR T CHUNK [reps=5]  (min and median tok/s; exl3 inside wcache.prefill_scope like the served path)"""
import sys, time, mlx.core as mx
from transformers import AutoTokenizer
kind, path, T, C = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); R = int(sys.argv[5]) if len(sys.argv) > 5 else 5
mx.set_cache_limit(256 * 2 ** 20)
ids = AutoTokenizer.from_pretrained(path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:T]
if kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    from kyojin_mlx import wcache as WC
    be, _ = load_backend(path)
    def run():
        c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
        with WC.prefill_scope(be.model, C, -(-T // C)):
            for s in range(0, T, C):
                mx.eval(be.prefill_chunk(c, ids[s:s + C]))
        return T / (time.perf_counter() - t0)
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(path); lm = model.language_model
    def run():
        c = make_prompt_cache(model); mx.synchronize(); t0 = time.perf_counter()
        for s in range(0, T, C):
            h = lm.model(mx.array(ids[s:s + C])[None], c)[:, -1:]
            lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
            mx.eval(lg, *[st for ci in c for st in ci.state if st is not None])
        return T / (time.perf_counter() - t0)
run(); v = sorted(run() for _ in range(R))
print(f"{kind} T={T} chunk={C}: max {v[-1]:.0f} med {v[len(v)//2]:.0f} tok/s", flush=True)

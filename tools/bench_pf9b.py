"""4K prefill of Qwen3.5-9B in chunks. usage: bench_pf9b.py exl3|mlx MODEL_DIR [--T 4096] [--chunk 256] [--reps 4]
exl3 = the served path (Q35Backend.prefill_chunk); mlx = mlx-lm model on the same chunk grid (logits of the last row only)."""
import argparse, time, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("path")
ap.add_argument("--T", type=int, default=4096); ap.add_argument("--chunk", default="256"); ap.add_argument("--reps", type=int, default=4); ap.add_argument("--norot", action="store_true", help="TIMING CEILING ONLY: rot becomes a cast (wrong numerics)")
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
if a.norot:
    from kyojin_mlx import kernels as _KN
    _KN.rot = lambda x, suh, out_dtype=mx.float32, fq8=False, perm=False: x.astype(out_dtype)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(a.path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 600)[:a.T]
if a.kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(a.path)
    mk = be.make_cache; run = lambda c, x: be.prefill_chunk(c, x)
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    m, _ = load(a.path); lm = m.language_model
    mk = lambda: make_prompt_cache(m)
    def run(c, x):
        h = lm.model(mx.array(x)[None], c)[:, -1:]
        lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
        mx.eval(lg, *[s for ci in c for s in ci.state if s is not None]); return lg
for ch in map(int, a.chunk.split(",")):
    def go(T=a.T):
        c = mk(); t0 = time.perf_counter()
        for i in range(0, T, ch): run(c, ids[i:i + ch])
        return T / (time.perf_counter() - t0)
    go(512); r = sorted((go() for _ in range(a.reps)), reverse=True)
    print(f"{a.kind} T={a.T} chunk={ch}: best {r[0]:.0f} p25 {r[len(r)//4]:.0f} tok/s peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)

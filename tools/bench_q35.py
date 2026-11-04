"""Prefill / decode speed on the Mac, N reps each (after a warm-up).
usage: bench_q35.py exl3|mlx|kmlx MODEL_DIR [--prefill 512,2048] [--decode 128] [--reps 3] [--last-only]
Prefill = one forward over the prompt. Default computes the logits of every prompt position (what model(x) returns);
--last-only runs the lm_head on the last position only (what an engine does, mlx-lm skips the dead rows by lazy eval).
Decode runs after a 512-token prompt, greedy: 'sync' = eval every token, 'async' = mlx-lm style (the next token's graph is
built and queued while the GPU still runs the current one). Prints per-rep tok/s and the median."""
import argparse, statistics, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
ap = argparse.ArgumentParser()
ap.add_argument("kind"); ap.add_argument("path")
ap.add_argument("--prefill", default="512,2048"); ap.add_argument("--decode", type=int, default=128)
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--last-only", action="store_true"); a = ap.parse_args()
if a.kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(a.path)
elif a.kind == "kmlx":
    from kyojin_mlx.mlxpack import load
    model, _ = load(a.path)
else:
    from mlx_lm import load
    model, _ = load(a.path)
from transformers import AutoTokenizer
text = "The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400
ids_all = AutoTokenizer.from_pretrained(a.path).encode(text)
lm = model.language_model

def forward(x, cache, last_only=False):
    if not last_only:
        return model(x, cache=cache)[:, -1]
    h = lm.model(x, cache)[:, -1:]
    out = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    return out[:, -1]

def prefill(P, last_only):
    x = mx.array(ids_all[:P])[None]
    cache = make_prompt_cache(model)
    t0 = time.perf_counter(); lg = forward(x, cache, last_only); mx.eval(lg); dt = time.perf_counter() - t0
    return dt, cache, lg

def decode_sync(n, P=512):
    _, cache, lg = prefill(P, True)
    t = mx.argmax(lg, axis=-1); mx.eval(t)
    t0 = time.perf_counter()
    for _ in range(n):
        t = mx.argmax(model(t[:, None], cache=cache)[:, -1], axis=-1); mx.eval(t)
    return time.perf_counter() - t0

def decode_async(n, P=512):
    _, cache, lg = prefill(P, True)
    t = mx.argmax(lg, axis=-1); mx.eval(t)
    step = lambda y: mx.argmax(model(y[:, None], cache=cache)[:, -1], axis=-1)
    t0 = time.perf_counter()
    nxt = step(t); mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n:
            nxt = step(cur); mx.async_eval(nxt)
        cur.item()
    return time.perf_counter() - t0

decode_sync(8); decode_async(8); prefill(512, a.last_only); prefill(64, a.last_only)
tag = "last-only" if a.last_only else "all-rows "
for P in map(int, a.prefill.split(",")):
    r = [P / prefill(P, a.last_only)[0] for _ in range(a.reps)]
    print(f"{a.kind:4s} prefill {P:5d} ({tag}): " + " ".join(f"{v:7.1f}" for v in r) + f"   median {statistics.median(r):7.1f} tok/s", flush=True)
for name, fn in (("sync ", decode_sync), ("async", decode_async)):
    r = [a.decode / fn(a.decode) for _ in range(a.reps)]
    print(f"{a.kind:4s} decode  {a.decode:5d} ({name})    : " + " ".join(f"{v:7.1f}" for v in r) + f"   median {statistics.median(r):7.1f} tok/s", flush=True)

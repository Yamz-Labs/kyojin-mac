"""Served-path prefill vs chunk size, interleaved, one process, NO fp16 fold (Air memory). exl3 arms go through backend.prefill_chunk (the served call); mlx arms = stock mlx-lm.
usage: bench88.py EXL3_DIR MLX_DIR T[,T] chunks(comma) [reps=3]   e.g. ... 512,2048 256,512,1024,2048"""
import sys, time, statistics, mlx.core as mx
exl3, mlxp, Ts, chunks = sys.argv[1], sys.argv[2], [int(t) for t in sys.argv[3].split(",")], [int(c) for c in sys.argv[4].split(",")]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
mx.set_cache_limit(256 * 2 ** 20)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(exl3).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 1200)
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC
be, _ = load_backend(exl3)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
m, _ = load(mlxp); lm = m.language_model
def run_mlx(c, x):
    h = lm.model(mx.array(x)[None], c)[:, -1:]
    lg = lm.lm_head(h) if hasattr(lm, "lm_head") else lm.model.embed_tokens.as_linear(h)
    mx.eval(lg, *[s for ci in c for s in ci.state if s is not None])
def go(k, T, ch):
    t0 = time.perf_counter()
    if k == "mlx":
        c = make_prompt_cache(m); mx.synchronize(); t0 = time.perf_counter()
        for i in range(0, T, ch): run_mlx(c, ids[i:i + ch][:min(ch, T - i)])
    else:
        c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
        with WC.prefill_scope(be.model, ch, -(-T // ch)):
            for i in range(0, T, ch): be.prefill_chunk(c, ids[i:i + ch][:min(ch, T - i)])
    return T / (time.perf_counter() - t0)
for T in Ts:
    arms = [(k, ch) for ch in chunks if ch <= T for k in ("exl3", "mlx")]
    for k, ch in arms: go(k, T, ch)
    r = {a: [] for a in arms}
    for _ in range(reps):
        for a in arms: r[a].append(go(*a, T) if False else go(a[0], T, a[1]))
    print(f"T={T}: " + "  ".join(f"{k}@{ch} {max(v):.0f}/{statistics.median(v):.0f}" for (k, ch), v in r.items()), flush=True)

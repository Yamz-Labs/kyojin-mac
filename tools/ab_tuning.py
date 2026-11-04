"""One arm of the autotune A/B (cold constants vs the pack's tuning table), run in its own process; the driver (ab_tuning_drive.py) alternates the arms.
  ab_tuning.py cold|tuned PACK [--prefill 128,512,2048] [--decode 128] [--reps 5]
cold = KYOJIN_TUNING=0 (set here before the import). Prints one JSON line: min / p25 / median of the prefill tok/s (one forward over the prompt, lm_head on the
last row, burst: 0.3 s idle before each) and the decode tok/s (async, mlx-lm style, after a 512-token prompt), the greedy ids of the first 24 decode tokens, the
logit checksum of the prefill, and the tuning status."""
import argparse
import json
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["cold", "tuned"]); ap.add_argument("pack")
ap.add_argument("--prefill", default="128,512,2048"); ap.add_argument("--decode", type=int, default=128); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
os.environ["KYOJIN_TUNING"] = "0" if a.arm == "cold" else "1"
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx import tuning as TN
from kyojin_mlx.exl3_linear import load_exl3

model, _ = load_exl3(a.pack)
lm = model.language_model
ids_all = AutoTokenizer.from_pretrained(a.pack).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400)


def forward(x, cache):
    h = lm.model(x, cache)[:, -1:]
    out = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    return out[:, -1]


def prefill(P):
    cache = make_prompt_cache(model)
    time.sleep(0.3)
    t0 = time.perf_counter(); lg = forward(mx.array(ids_all[:P])[None], cache); mx.eval(lg)
    return time.perf_counter() - t0, cache, lg


def decode(n, P=512):
    _, cache, lg = prefill(P)
    t = mx.argmax(lg, axis=-1); mx.eval(t)
    step = lambda y: mx.argmax(model(y[:, None], cache=cache)[:, -1], axis=-1)
    toks = []
    t0 = time.perf_counter()
    nxt = step(t); mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n:
            nxt = step(cur); mx.async_eval(nxt)
        toks.append(int(cur.item()))
    return n / (time.perf_counter() - t0), toks


def stats(v):
    v = sorted(v)
    return dict(min=v[0], p25=v[len(v) // 4], median=statistics.median(v), max=v[-1])


res = dict(arm=a.arm, tuning=dict(TN.INFO), shapes_tuned=len(TN.TABLE))
for P in [int(p) for p in a.prefill.split(",")]:
    prefill(P)
    res[f"prefill_{P}"] = stats([P / prefill(P)[0] for _ in range(a.reps)])
_, _, lg = prefill(512)
res["logit_sum"] = float(mx.sum(lg.astype(mx.float32)).item())
decode(8)
runs = [decode(a.decode) for _ in range(a.reps)]
res["decode"] = stats([r[0] for r in runs])
res["tokens"] = runs[0][1][:24]
res["peak_gib"] = round(mx.get_peak_memory() / 2 ** 30, 2)
print("RESULT " + json.dumps(res))

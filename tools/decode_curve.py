"""Solo decode tok/s against the context length (the flat curve of DS4.1: the KV is a sliding ring + compressed pools, so the decode should barely slow down with the context).
usage: decode_curve.py PACK [--points 4096,32768,65536,131072,262144] [--chunk 4096] [--n 64] [--reps 3] [--ref ref.json] [--seed 0] [--json F] [--set key=json ...]

One cache, one prefill pass: the context grows from point to point (the decode steps of the previous point stay in the context, <= n x reps tokens, negligible against the gaps).
At every point: `reps` runs of `n` plain solo decode steps (greedy, one sync per token, the rows-enabled model like tools/spec_gen.py depth 0); tok/s min / median / max, plus the
prefill tok/s of the segment that led to the point and the MLX peak memory. Prompt ids: the --ref prompts tiled (real routing), else seeded random ids."""
import argparse, json, statistics, sys, time

import mlx.core as mx
import numpy as np

from kyojin_mlx import headlast as HL, patch as P, rows as R, wcache as WC

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--points", default="4096,32768,65536,131072,262144"); ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--n", type=int, default=64); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--ref"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json"); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
try:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg)
    if es is not None:
        tables = ER.EngramTables(a.pack, es)
        bad = tables.validate()
        assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
        EM.install_parallel_gather(model, tables)
except ImportError:
    pass
print("trunk:", R.enable(model, spec), flush=True)
vocab = {**cfg["text_config"], **over}["vocab_size"]
pts = [int(x) for x in a.points.split(",")]
total = max(pts)
if a.ref:
    base = [t for r in json.load(open(a.ref))["prompts"] for t in r["prompt_ids"]]
    ids = (base * (total // len(base) + 1))[:total]
else:
    ids = np.random.default_rng(a.seed).integers(16, vocab - 96, total).tolist()
lm = model.language_model
cache = lm.make_cache()
done, rows, tok = 0, [], None


def prefill_to(n):
    """prefill ids[done:n] in chunks (dense weights decoded once per segment); returns the logits of the last row"""
    global done
    lg = None
    steps = -(-(n - done) // a.chunk)
    with WC.prefill_scope(model, a.chunk, steps):
        for s in range(done, n, a.chunk):
            with HL.last_row():
                lg = lm(mx.array(ids[s:min(s + a.chunk, n)])[None], cache=cache).logits[:, -1].astype(mx.float32)
            mx.eval(lg)
    done = n
    return lg


for p in pts:
    t0 = time.perf_counter()
    lg = prefill_to(p)
    t_pf = time.perf_counter() - t0
    tok = int(mx.argmax(lg[0]).item())
    for _ in range(4):                                                                       # warm up this context size
        tok = int(mx.argmax(R.step(model, cache, mx.array([tok]))[0]).item())
    rates = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for _ in range(a.n):
            tok = int(mx.argmax(R.step(model, cache, mx.array([tok]))[0]).item())
        rates.append(a.n / (time.perf_counter() - t0))
    r = dict(context=p, tok_s_min=min(rates), tok_s_median=statistics.median(rates), tok_s_max=max(rates), prefill_tok_s=(p - (rows[-1]["context"] if rows else 0)) / t_pf,
             peak_gib=mx.get_peak_memory() / 2 ** 30)
    rows.append(r)
    print(f"context {p:7d}: decode {r['tok_s_median']:.1f} tok/s (min {r['tok_s_min']:.1f}, max {r['tok_s_max']:.1f}); prefill of the segment {r['prefill_tok_s']:.0f} tok/s; peak {r['peak_gib']:.2f} GiB", flush=True)
first, last = rows[0]["tok_s_median"], rows[-1]["tok_s_median"]
print(f"flatness: {last / first:.3f} (decode at {rows[-1]['context']} / at {rows[0]['context']})", flush=True)
if a.json:
    json.dump(dict(pack=a.pack, chunk=a.chunk, n=a.n, reps=a.reps, points=rows), open(a.json, "w"), indent=1)

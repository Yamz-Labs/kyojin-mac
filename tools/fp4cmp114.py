"""fp4cmp114.py PACK [--tokens 4096] [--chunk 512] [--decode 24]: the stock DS4.1 cache vs the fp4-packed cache (fp4pool) on the same prompt: prefill + greedy decode logits compared bitwise
at every step, the cache arrays compared bitwise, bytes held by each (pool + index keys), decode ms per token."""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tools")); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx
import ced_gate as G
from kyojin_mlx import fp4pool
from kyojin_mlx.serve.backends import arrays_of

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--tokens", type=int, default=4096); ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--decode", type=int, default=24); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
model, spec, cfg, tables = G.load(a.pack, None)
lm = model.language_model
vocab = cfg.get("text_config", cfg)["vocab_size"]
ids = [int(x) for x in np.random.default_rng(a.seed).integers(3, vocab, a.tokens)]


def run(packed):
    cache = lm.make_cache()
    if packed:
        cache = fp4pool.wrap_cache(cache)
    logs = []
    for s in range(0, a.tokens, a.chunk):
        out = lm(mx.array([ids[s:s + a.chunk]]), cache=cache)
        mx.eval(out.logits, *arrays_of(cache))
    lg = out.logits[:, -1].astype(mx.float32)
    logs.append(lg)
    t = int(mx.argmax(lg, -1).item())
    t0 = None
    for i in range(a.decode):
        if i == 4: t0 = time.time()
        lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg, *arrays_of(cache))
        logs.append(lg); t = int(mx.argmax(lg, -1).item())
    ms = (time.time() - t0) / (a.decode - 4) * 1000
    e = cache[0]
    pool = sum(x.nbytes for x in arrays_of(e.compress)) + sum(x.nbytes for x in arrays_of(e.keys))
    return cache, logs, ms, pool


cd, ld, msd, bd = run(False)
cp, lp, msp, bp = run(True)
same_logits = all(bool(mx.array_equal(x, y)) for x, y in zip(ld, lp))
sd, sp = cd[0].state, cp[0].state
same_state = len(sd) == len(sp) and all(bool(mx.array_equal(x, y)) for x, y in zip(sd, sp))
print("FP4CMP114", json.dumps(dict(tokens=a.tokens, logits_bitwise=same_logits, cache_arrays_bitwise=same_state, pool_keys_bytes_dense=bd, pool_keys_bytes_packed=bp,
                                   ratio=round(bp / max(bd, 1), 3), ms_tok_dense=round(msd, 1), ms_tok_packed=round(msp, 1))), flush=True)

"""glm115.py PACK [--prompt 32768] [--chunk 2048] [--new 16] [--dup 3]: GLM latent TQ decode with the gathered-row attention (latent_tq.install_attend) vs the stock dense-latent attention,
same prompt, two caches: logits bitwise at every step, then ABAB decode ms/token on one cache."""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx
from kyojin_mlx import patch as P, latent_tq
from mlx_vlm.models.glm5_next import language as L

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--prompt", type=int, default=32768); ap.add_argument("--new", type=int, default=16)
ap.add_argument("--dup", type=int, default=3); ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--nobit", action="store_true"); a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
lm = model.language_model
inner = lm.model if hasattr(lm, "model") else lm
last = inner.layers[-1]
while sum(1 for l in inner.layers if l is last) < a.dup: inner.layers.append(last)
V = cfg.get("text_config", cfg)["vocab_size"]
ids = [int(x) for x in np.random.default_rng(0).integers(3, V, a.prompt)]
stock = L.Glm5NextAttention._attend
latent_tq.install_attend()
gath = L.Glm5NextAttention._attend


def mode(g):
    L.Glm5NextAttention._attend = gath if g else stock


def prefill():
    cache = lm.make_cache()
    latent_tq.wrap_latent_layers(cache, dict(kbits=6, sink=4, window=256))
    for s in range(0, a.prompt - 1, a.chunk):
        e = min(s + a.chunk, a.prompt - 1)
        lm(mx.array([ids[s:e]]), cache=cache); mx.eval(mx.array(0))
    return cache


def step(cache, t):
    lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg); return lg


forced = [int(x) for x in np.random.default_rng(1).integers(3, V, a.new)]
same = None
if a.nobit:
    mode(True); c1 = prefill()
else:
    mode(False); c0 = prefill(); mode(True); c1 = prefill()
    same = True
    t = ids[-1]
    for i in range(a.new):
        mode(False); l0 = step(c0, t); mode(True); l1 = step(c1, t)
        same &= bool(mx.array_equal(l0, l1)); t = forced[i]
    print("bitwise", same, flush=True)
    del c0
ms = {"stock": [], "gather": []}
for r in range(4):
    for g in ((False, True) if r % 2 == 0 else (True, False)):
        mode(g)
        for i in range(12):
            if i == 2: t0 = time.time()
            step(c1, forced[i % a.new])
        ms["gather" if g else "stock"].append(round((time.time() - t0) / 10 * 1000, 2))
print("GLM115", json.dumps(dict(prompt=a.prompt, bitwise=same, **ms)), flush=True)

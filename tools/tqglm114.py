"""tqglm114.py PACK [--prompt 2048] [--new 24] [--dup 3] [--chunk 512]: the REAL LatentTQCache (serve path: latent_tq.wrap_latent_layers on lm.make_cache()) against the fp16 cache
on a whole-model slice (last layer repeated, shared weights): KL / top-1 per teacher-forced step, decode ms per token, cache bytes. Prefill goes in chunks."""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx
from kyojin_mlx import patch as P, latent_tq, tq

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--prompt", type=int, default=2048); ap.add_argument("--new", type=int, default=24)
ap.add_argument("--dup", type=int, default=3); ap.add_argument("--chunk", type=int, default=512); ap.add_argument("--window", type=int, default=256); ap.add_argument("--sink", type=int, default=4); ap.add_argument("--seeds", type=int, default=6); a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
lm = model.language_model
inner = lm.model if hasattr(lm, "model") else lm
last = inner.layers[-1]
while sum(1 for l in inner.layers if l is last) < a.dup: inner.layers.append(last)
print("latent_ok", latent_tq.latent_ok(lm), "layers", len(inner.layers), flush=True)
ids = None


def fakek6v6(x, bits=6, group=64):
    w, sc, bi = mx.quantize(x, group_size=group, bits=bits)
    return mx.dequantize(w, sc, bi, group_size=group, bits=bits).astype(x.dtype)


def run(tqmode, force=None):
    cache = lm.make_cache()
    n = 0
    if tqmode == "k6v6":                                    # the engine k6v6 cache on the latent rows (affine g64, no sink / window), chunked prefill like the TQ run
        for c in cache:
            if hasattr(c, "caches"):
                sub = c.caches[0]; orig = sub.update_and_fetch
                sub.update_and_fetch = (lambda k, v, orig=orig: orig(fakek6v6(k), v)); n += 1
    elif tqmode:
        n = latent_tq.wrap_latent_layers(cache, dict(kbits=6, sink=a.sink, window=a.window))
    for s in range(0, a.prompt - a.chunk, a.chunk):
        lm(mx.array([ids[s:s + a.chunk]]), cache=cache); mx.eval([c.state for c in cache if hasattr(c, "state")][:0] or [mx.array(0)])
    lg = lm(mx.array([ids[(a.prompt - 1) // a.chunk * a.chunk:]]), cache=cache).logits[:, -1].astype(mx.float32)
    toks, lps = [], []
    t0 = None
    for i in range(a.new):
        lp = lg - mx.logsumexp(lg, -1, keepdims=True); mx.eval(lp); lps.append(np.array(lp[0]))
        if i == 4: t0 = time.time()
        t = force[i] if force is not None else int(mx.argmax(lp, -1).item()); toks.append(t)
        lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
    mx.eval(lg)
    ms = (time.time() - t0) / (a.new - 5) * 1000
    lat = [c.caches[0] for c in cache if hasattr(c, "caches")]
    nb = sum(getattr(c, "nbytes", 0) if not hasattr(c, "keys") or c.keys is None else (c.keys[..., :c.offset, :].nbytes) for c in lat)
    print("run", "tq" if tqmode else "fp16", "wrapped", n, "ms/token %.1f" % ms, "latent cache bytes", nb, "types", sorted({type(c).__name__ for c in lat}), flush=True)
    return toks, np.stack(lps)


res = {"tq6w": [], "k6v6": []}
for seed in range(a.seeds):
    ids = np.random.default_rng(seed).integers(100, 4000, a.prompt).tolist()
    toks, ref = run(False)
    for name, m in (("tq6w", "tq"), ("k6v6", "k6v6")):
        _, lp = run(m, toks)
        kl = (np.exp(ref) * (ref - lp)).sum(-1)
        res[name].append(float(kl.mean()))
        print("TQGLM114 seed", seed, name, "kl_mean %.3e max %.3e top1 %.2f" % (kl.mean(), kl.max(), (ref.argmax(-1) == lp.argmax(-1)).mean()), flush=True)
print("TQGLM114 SUMMARY", json.dumps({k: dict(mean=float(np.mean(v)), median=float(np.median(v)), min=min(v), max=max(v)) for k, v in res.items()}), flush=True)

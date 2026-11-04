"""tqkl111.py PACK [--new 24] [--prompt 1536] [--modes lat6,lat6pe6,lat5]: KL / top-1 of a TurboQuant MLA latent cache (GLM sparse-attention layers) against the fp16 cache,
on a whole-model slice. The kv latent (kv_lora_rank values, quantized in 256-wide halves, rotation + Lloyd-Max codebook, one fp16 norm per half) and optionally the k_pe
rows are stored fake-quantized (rotate -> tq_write -> tq_dequant -> unrotate) at write time, with NO fp16 sink / window overlay (the worst case of the engine's tq6w). The
fp16 run is greedy; every TQ run is teacher-forced on its tokens, so the per-step KL(fp16 || tq) is meaningful. Indexer keys stay fp16."""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx
from kyojin_mlx import patch as P, tq

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--new", type=int, default=24); ap.add_argument("--prompt", type=int, default=1536)
ap.add_argument("--modes", default="lat6,k6v6,lat5"); ap.add_argument("--dup", type=int, default=3, help="in-process slice: the last decoder layer is repeated until the model has this many cached latent layers (shared weights, one cache each)"); a = ap.parse_args()


def fakek6v6(x, bits=6, group=64):
    """The engine k6v6 cache (WinKVCache, mx.quantize affine, group 64) applied to the latent rows, no sink / window."""
    w, sc, bi = mx.quantize(x, group_size=group, bits=bits)
    return mx.dequantize(w, sc, bi, group_size=group, bits=bits).astype(x.dtype)


def fakeq(x, bits, width=256):
    sh, dt = x.shape, x.dtype
    d = sh[-1]
    parts = []
    for o in range(0, d, width):
        w = min(width, d - o)
        y = tq.rotate(x[..., o:o + w].reshape(-1, w), w)
        words, sc = tq.tq_write(y, bits)
        parts.append(tq.unrotate(tq.tq_dequant(words, sc, bits, w, mx.float32), w).reshape(*sh[:-1], w))
    return mx.concatenate(parts, -1).astype(dt)


model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
lm = model.language_model
inner = lm.model if hasattr(lm, "model") else lm
last = inner.layers[-1]
while sum(1 for l in inner.layers if l is last) < a.dup: inner.layers.append(last)
print("slice: %d layers (last repeated, shared weights)" % len(inner.layers), flush=True)
vocab = int(cfg["text_config"].get("vocab_size", 4096)) if isinstance(cfg, dict) else 4096
ids = np.random.default_rng(0).integers(100, min(vocab, 4000), a.prompt).tolist()


def make(mode):
    cache = lm.make_cache()
    n = 0
    if mode != "fp16":
        bits = 5 if mode.startswith("lat5") else 6
        kq = mode == "k6v6"
        pe = "pe" in mode
        for c in cache:
            kv = c[0] if isinstance(c, (list, tuple)) or type(c).__name__ == "CacheList" else None
            if kv is None or not hasattr(kv, "update_and_fetch") or type(c).__name__ != "CacheList":
                continue
            orig = kv.update_and_fetch
            def f(k, v, orig=orig, bits=bits, pe=pe):
                if not getattr(f, 'seen', 0): print('hook shapes', k.shape, v.shape, flush=True); f.seen = 1
                return orig(fakek6v6(k) if kq else fakeq(k, bits), fakeq(v, 6 if bits == 5 else bits, 64) if pe else v)
            kv.update_and_fetch = f; n += 1
    return cache, n


def run(mode, force=None):
    cache, n = make(mode)
    t0 = time.time()
    lg = lm(mx.array([ids]), cache=cache).logits[:, -1].astype(mx.float32)
    toks, lps = [], []
    for i in range(a.new):
        lp = lg - mx.logsumexp(lg, -1, keepdims=True); mx.eval(lp); lps.append(np.array(lp[0]))
        t = force[i] if force is not None else int(mx.argmax(lp, -1).item()); toks.append(t)
        lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
    print("run", mode, "hooked layers", n, "cache types", sorted({type(c).__name__ for c in cache}), "s %.1f" % (time.time() - t0), flush=True)
    return toks, np.stack(lps)


toks, ref = run("fp16")
res = {}
for m in a.modes.split(","):
    _, lp = run(m, toks)
    kl = (np.exp(ref) * (ref - lp)).sum(-1)
    top1 = float((ref.argmax(-1) == lp.argmax(-1)).mean())
    res[m] = dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), top1=top1)
    print("TQKL111", m, json.dumps(res[m]), flush=True)

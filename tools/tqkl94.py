"""tqkl94.py MODEL CTX [PFS] [EVAL]: KL / top-1 of the TurboQuant KV prefill (tq6w, KYOJIN_TQ_PF modes, comma list, default 0,1) against the fp16
KV cache, on the last EVAL (default 512) positions of a CTX-token corpus prompt prefilled in 2048-row chunks (the server's chunk), plus the
prefill wall time of each config. mlx-lm model + the kvwin quantized-SDPA hook (the engine's attention route for TQ caches)."""
import os, sys, json, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
from kyojin_mlx import kvwin, tq

MODEL, CTX = sys.argv[1], int(sys.argv[2])
PFS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 512
CH = 2048
model, tok = load(MODEL)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json")))
ids = (ids["ids"] if isinstance(ids, dict) else ids)[300000:300000 + CTX]
kvwin.install()


def run(pf):
    cache = make_prompt_cache(model)
    if pf is not None:
        tq.PF = pf
        cache = [tq.TQKVCache(6, 6, 4, 256) if type(c) is KVCache else c for c in cache]
    out = []
    t0 = time.perf_counter()
    for a in range(0, CTX, CH):
        x = mx.array(ids[a:a + CH])[None]
        lg = model(x, cache=cache)
        keep = max(0, (a + x.shape[1]) - (CTX - EVAL))
        if keep:
            out.append((lambda z: z - mx.logsumexp(z, axis=-1, keepdims=True))(lg[0, -keep:].astype(mx.float32)))
            mx.eval(out[-1])
        mx.eval([c.state for c in cache])
    dt = time.perf_counter() - t0
    return mx.concatenate(out, axis=0), dt


ref, tr = run(None)
print(f"{os.path.basename(MODEL)} ctx {CTX} fp16 KV prefill {tr:.1f} s", flush=True)
for pf in PFS:
    lp, t = run(pf)
    kl = mx.sum(mx.exp(ref) * (ref - lp), axis=-1)
    top = mx.mean((mx.argmax(ref, -1) == mx.argmax(lp, -1)).astype(mx.float32)).item()
    print(f"  tq6w PF={pf}: prefill {t:.1f} s  KL mean {mx.mean(kl).item():.2e} max {mx.max(kl).item():.2e} top1 {top:.4f}", flush=True)
    del lp, kl

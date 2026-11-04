"""vmodel87.py MODEL [N=4096] [T=4]: logits of a T-token chunk (fused verify kernel) vs the same T tokens one by one (fused decode) vs the chunk on the generic path; max |dlogp| and KL."""
import sys, os, numpy as np
import mlx.core as mx
import kvwin_kl as K
from mlx_lm import load
mdir = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
model, tok = load(mdir); lm = model.language_model; hd = K.head(lm); ids = K.tokens(tok, N + 3 * T, 0)
def logp(h):
    lg = hd(h)[0].astype(mx.float32); return np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True))
def run(mode):
    cache = K.cache_for(model, [6, 6, 0, 4, 256])
    for s in range(0, N, 512):
        mx.eval(lm.model(mx.array(ids[s:s + 512])[None], cache))
    out = []
    for r in range(3):
        toks = ids[N + r * T:N + (r + 1) * T]
        if mode == "seq":
            for t in toks: out.append(logp(lm.model(mx.array([[t]]), cache))[0])
        else:
            out.extend(list(logp(lm.model(mx.array(toks)[None], cache))))
        mx.eval(cache[0].state) if False else None
    return np.stack(out)
res = {}
for mode, dec in (("seq", "1"), ("chunk_fused", "1"), ("chunk_generic", "0")):
    for c in range(1):
        pass
    os.environ["KYOJIN_TQ_DECODE"] = dec
    res[mode] = run("seq" if mode == "seq" else "chunk")
def kl(p, q): return float((np.exp(p.astype(np.float64)) * (p - q)).sum(-1).mean())
print("max|dlogp| fused chunk vs seq", float(np.abs(res["chunk_fused"] - res["seq"]).max()), " generic chunk vs seq", float(np.abs(res["chunk_generic"] - res["seq"]).max()))
print("KL(seq||fused chunk)", kl(res["seq"], res["chunk_fused"]), " KL(seq||generic chunk)", kl(res["seq"], res["chunk_generic"]), " top1 agree fused/generic", float((res["seq"].argmax(-1) == res["chunk_fused"].argmax(-1)).mean()), float((res["seq"].argmax(-1) == res["chunk_generic"].argmax(-1)).mean()))

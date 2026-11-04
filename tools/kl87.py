"""kl87.py MODEL OUT_NPZ [N=4096] [STEPS=64]: N-token real prefill (512-row chunks) + STEPS teacher-forced decode steps, TQ6 cache and fp16 cache; saves log-softmax of the last 16 prefill rows
and every decode step (fp16 npz) so two source trees (int86 / lc87) can be compared offline (kl87_cmp.py)."""
import sys, os, numpy as np
import mlx.core as mx
import kvwin_kl as K
import kyojin_mlx
from mlx_lm import load
mdir, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
model, tok = load(mdir)
lm = model.language_model; hd = K.head(lm); ids = K.tokens(tok, N + STEPS, 0)
res = {}
for name, cfg in (("tq6", [6, 6, 0, 4, 256]), ("fp16", [0, 0, 0, 0, 0])):
    cache = K.cache_for(model, cfg)
    for s in range(0, N, 512):
        h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h)
    pre = hd(h[:, -16:])[0].astype(mx.float32)
    outs = [pre - mx.logsumexp(pre, axis=-1, keepdims=True)]
    for i in range(STEPS):
        lg = hd(lm.model(mx.array([[ids[N + i]]]), cache))[0].astype(mx.float32)
        outs.append(lg - mx.logsumexp(lg, axis=-1, keepdims=True)); mx.eval(outs[-1])
    res[name] = np.array(mx.concatenate(outs, axis=0))
    del cache; mx.clear_cache()
np.savez(out, **res)
print(kyojin_mlx.__file__.split("/")[-3], mdir.split("/")[-1], {k: v.shape for k, v in res.items()}, flush=True)

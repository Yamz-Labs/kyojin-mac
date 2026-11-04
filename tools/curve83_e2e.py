"""End-to-end check: greedy tokens of the real 4B with the fused TQ decode kernel vs the generic dequant path (same cache contents), 2K context, 48 steps."""
import sys, mlx.core as mx
from mlx_lm import load
import kvwin_kl as K
from kyojin_mlx import tq
model, tok = load(sys.argv[1]); lm = model.language_model; hd = K.head(lm)
ids = K.tokens(tok, 2048, 0)
outs = {}
for fused in (True, False):
    cache = K.cache_for(model, [6, 6, 0, 4, 256])
    for c in cache:
        if hasattr(c, "fused"): c.fused = fused
    for s in range(0, 2048, 512):
        h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h)
    y = mx.argmax(hd(h[:, -1:])[0], axis=-1); seq = []
    for _ in range(48):
        h = lm.model(y[None], cache); y = mx.argmax(hd(h)[0], axis=-1); mx.eval(y); seq.append(y.item())
    outs[fused] = seq
agree = sum(a == b for a, b in zip(outs[True], outs[False]))
print("fused vs generic greedy tokens agree:", agree, "/ 48"); print(tok.decode(outs[True])[:120].replace("\n", " "))

"""gdnpf89.py MODEL_DIR [steps]: fused GDN front end + gated norm at decode (KYOJIN_GDN_DEC) vs stock: logits equality over greedy steps from the same prefill."""
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import gdn_pf
model, tok = load(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
assert gdn_pf.install(), "gdn_pf not installed"
ids = tok.encode("The history of the Roman empire is long and " * 40)[:300]
def run(dec):
    gdn_pf.DEC = dec
    c = make_prompt_cache(model)
    lg = model(mx.array(ids)[None], cache=c)[:, -1]; y = mx.argmax(lg, axis=-1); outs = [lg]
    for _ in range(n):
        lg = model(y[None], cache=c)[:, -1]; y = mx.argmax(lg, axis=-1); outs.append(lg)
    mx.eval(outs); return outs
a = run(False); b = run(True)
d = [float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)))) for x, y in zip(a, b)]
print("steps", len(a), "max |dlogit|", max(d), "equal", all(bool(mx.array_equal(x, y)) for x, y in zip(a, b)), "argmax same", all(bool(mx.array_equal(mx.argmax(x, -1), mx.argmax(y, -1))) for x, y in zip(a, b)))
def kl(x, y):
    lx = x.astype(mx.float32) - mx.logsumexp(x.astype(mx.float32), axis=-1, keepdims=True); ly = y.astype(mx.float32) - mx.logsumexp(y.astype(mx.float32), axis=-1, keepdims=True)
    return float(mx.sum(mx.exp(lx) * (lx - ly)))
k = [kl(x, y) for x, y in zip(a, b)]
print("KL(stock||fused) mean %.2e max %.2e step1 %.2e" % (sum(k) / len(k), max(k), k[1]))

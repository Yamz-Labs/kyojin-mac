import sys, time, os
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx import gdn_fast
d, P, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(d)
else:
    from mlx_lm import load
    model, _ = load(d)
gdn_fast.install()
ids = AutoTokenizer.from_pretrained(d).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 800)[:P]
x = mx.array(ids)[None]
def run():
    lm = model.language_model; c = make_prompt_cache(model); h = lm.model(x, cache=c)[:, -1]
    out = lm.lm_head(h) if getattr(lm, "lm_head", None) is not None else lm.model.embed_tokens.as_linear(h)
    mx.eval(out); return out.astype(mx.float32)
def lp(l): return l - mx.logsumexp(l, -1, keepdims=True)
res = {}
for mode in ("0", "64", "0", "64", "0", "64"):
    os.environ["KYOJIN_GDN_FAST"] = mode
    run(); t = time.perf_counter(); l = run(); dt = time.perf_counter() - t
    res.setdefault(mode, []).append(dt); res[mode + "l"] = l
a, b = lp(res["0l"]), lp(res["64l"])
kl = (mx.exp(a) * (a - b)).sum().item()
print(f"P={P} stock {min(res['0'])*1e3:.0f} ms  fast {min(res['64'])*1e3:.0f} ms  KL {kl:.3g} argmax equal {mx.argmax(a).item()==mx.argmax(b).item()}")

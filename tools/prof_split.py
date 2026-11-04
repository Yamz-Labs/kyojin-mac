"""Sync-barrier time split of one 9B prefill: usage prof_split.py exl3|mlx MODEL_DIR [--T 4096] [--chunk 2048]
Every wrapped module is evaluated before/after (inflates by the barrier cost, so read the SHARES); leaf linears are timed inside their parent."""
import argparse, time, collections, mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("path")
ap.add_argument("--T", type=int, default=4096); ap.add_argument("--chunk", type=int, default=2048); a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(a.path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 600)[:a.T]
if a.kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(a.path); model = be.model; mk = be.make_cache
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(a.path); mk = lambda: make_prompt_cache(model)
lm = model.language_model; T = collections.defaultdict(float); N = collections.Counter()
class Tm(nn.Module):
    def __init__(s, m, name): super().__init__(); s.m = m; s.name = name
    def __call__(s, *x, **k):
        mx.synchronize(); t0 = time.perf_counter(); y = s.m(*x, **k); mx.eval(y); dt = time.perf_counter() - t0
        T[s.name] += dt; N[s.name] += 1; return y
    def __getattr__(s, k):
        try: return super().__getattr__(k)
        except AttributeError:
            if k in ("m", "name"): raise
            return getattr(s.m, k)
def wrap(parent, key, name): setattr(parent, key, Tm(getattr(parent, key), name))
def prep():
    for l in lm.model.layers:
        kind = "gdn" if l.is_linear else "attn"
        for k in ("input_layernorm", "post_attention_layernorm"): wrap(l, k, "norm")
        mix = l.linear_attn if l.is_linear else l.self_attn
        for k in [k for k in mix.keys() if k.endswith("_proj") or k.startswith("in_proj")]:
            wrap(mix, k, f"{kind}.{k}")
        wrap(l, "linear_attn" if l.is_linear else "self_attn", f"{kind}.TOTAL")
        for k in ("gate_proj", "up_proj", "down_proj"): wrap(l.mlp, k, f"mlp.{k}")
        wrap(l, "mlp", "mlp.TOTAL")
def run(c, x):
    h = lm.model(mx.array(x)[None], c)[:, -1:]
    lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h); mx.eval(lg)
def full():
    c = mk(); mx.synchronize(); t0 = time.perf_counter()
    for i in range(0, a.T, a.chunk): run(c, ids[i:i + a.chunk])
    return time.perf_counter() - t0
full(); t_plain = min(full() for _ in range(2))
prep(); full(); T.clear(); N.clear(); t_prof = full()
print(f"{a.kind} T={a.T} chunk={a.chunk}: plain {a.T/t_plain:.0f} tok/s ({t_plain*1e3:.0f} ms), instrumented {t_prof*1e3:.0f} ms")
for k in sorted(T): print(f"  {k:22s} {T[k]*1e3:8.1f} ms  n={N[k]:4d}  {T[k]/t_prof*100:5.1f} %")
tot = lambda p: sum(v for k, v in T.items() if k.startswith(p) and not k.endswith("TOTAL"))
for kind in ("gdn", "attn", "mlp"):
    print(f"  {kind}: glue (TOTAL - linears) = {(T[kind+'.TOTAL']-tot(kind+'.'))*1e3:.1f} ms")
print(f"  rest (embed/final norm/head/outside wrappers) = {(t_prof-T['norm']-T['gdn.TOTAL']-T['attn.TOTAL']-T['mlp.TOTAL'])*1e3:.1f} ms")

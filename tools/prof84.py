"""Per-op prefill table, Qwen3.5 (4B/9B), chunked as served. usage: prof84.py mlx|exl3 MODEL_DIR [--T 32768] [--chunk 2048] [--stride 1] [--reps 2] [--out F.jsonl]
For every chunk offset o = 0, chunk, 2*chunk ... < T: plain chunk time (no barriers, min of reps) and an instrumented chunk (sync barrier around every op, self times by class).
The KV of the full-attention layers is tiled to o rows (timing does not depend on the values; GDN state and the stock KVCache growth copy are as served).
Classes: gemm (every *_proj / in_proj_* linear), gdn_core (gated_delta_update), gdn_conv, sdpa, rope, kv_write (update_and_fetch), norm, glue_* (module self time = the rest)."""
import argparse, time, json, collections, mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("path")
ap.add_argument("--T", type=int, default=32768); ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--reps", type=int, default=2); ap.add_argument("--gdnfast", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
from transformers import AutoTokenizer
from mlx_lm.models import qwen3_5 as Q5, qwen3_next as QN
from mlx_lm.models.cache import KVCache
ids = AutoTokenizer.from_pretrained(a.path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:a.T]
if a.kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(a.path); model = be.model; mk = be.make_cache
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(a.path); mk = lambda: make_prompt_cache(model)
if a.gdnfast:
    import os; os.environ["KYOJIN_GDN_FAST"] = "1"; from kyojin_mlx import gdn_fast; gdn_fast.install()
lm = model.language_model
def run(c, x):
    h = lm.model(mx.array(x)[None], c)[:, -1:]
    lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    mx.eval(lg, *[s for ci in c for s in (ci.state if hasattr(ci, "state") else []) if s is not None]); return lg
# fabricated context: master KV at T rows
cache = mk(); run(cache, ids[:a.chunk])
kvl = [c for c in cache if isinstance(c, KVCache)]
master = [(mx.tile(c.keys[:, :, :a.chunk], (1, 1, a.T // a.chunk, 1)), mx.tile(c.values[:, :, :a.chunk], (1, 1, a.T // a.chunk, 1))) for c in kvl]
mx.eval(master)
def at(o):
    for c, (k, v) in zip(kvl, master):
        c.keys = k[:, :, :o] + 0 if o else None; c.values = v[:, :, :o] + 0 if o else None; c.offset = o
    mx.eval([x for c in kvl for x in (c.keys, c.values) if x is not None]); mx.synchronize()
# instrumentation
S = collections.defaultdict(float); stack = []
def timed(fn, cls):
    def w(*x, **k):
        mx.eval([v for v in list(x) + list(k.values()) if isinstance(v, mx.array)])             # lazy upstream ops are charged to the parent's glue, not to this op
        mx.synchronize(); stack.append(0.0); t0 = time.perf_counter(); y = fn(*x, **k); mx.eval(y)
        dt = time.perf_counter() - t0; ch = stack.pop(); S[cls] += dt - ch
        if stack: stack[-1] += dt
        return y
    return w
class Tm(nn.Module):
    def __init__(s, m, cls): super().__init__(); s.m = m; s.cls = cls; s._f = timed(m, cls)
    def __call__(s, *x, **k): return s._f(*x, **k)
    def __getattr__(s, k):
        try: return super().__getattr__(k)
        except AttributeError:
            if k in ("m", "cls", "_f"): raise
            return getattr(s.m, k)
def wrap(parent, key, cls): setattr(parent, key, Tm(getattr(parent, key), cls))
INST = False
def prep():
    for l in lm.model.layers:
        wrap(l, "input_layernorm", "norm_in"); wrap(l, "post_attention_layernorm", "norm_in")
        mix = l.linear_attn if l.is_linear else l.self_attn; kind = "gdn" if l.is_linear else "attn"
        for k in [k for k in mix.keys() if k.endswith("_proj") or k.startswith("in_proj")]: wrap(mix, k, "gemm")
        if l.is_linear:
            wrap(mix, "conv1d", "gdn_conv"); wrap(mix, "norm", "norm_gdn")
        else:
            for k in ("q_norm", "k_norm"): wrap(mix, k, "norm_qk")
            wrap(mix, "rope", "rope")
        wrap(l, "linear_attn" if l.is_linear else "self_attn", "glue_" + kind)
        for k in ("gate_proj", "up_proj", "down_proj"): wrap(l.mlp, k, "gemm")
        wrap(l, "mlp", "glue_mlp")
    Q5.gated_delta_update = timed(Q5.gated_delta_update, "gdn_core")
    QN.scaled_dot_product_attention = timed(QN.scaled_dot_product_attention, "sdpa")
    KVCache.update_and_fetch = (lambda f: (lambda self, *x, **k: timed(lambda: f(self, *x, **k), "kv_write")()))(KVCache.update_and_fetch)
_g = lm.model.layers[0].mlp.gate_proj; _x = (mx.random.normal((a.chunk, lm.args.hidden_size)) * 0.5).astype(mx.bfloat16); mx.eval(_x)
def ref_tf():                                                     # instantaneous clock regime: the model's own gate_proj on a full chunk, best of 5 single calls
    ts = []
    for _ in range(5):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(_g(_x)); ts.append(time.perf_counter() - t0)
    return round(2 * a.chunk * _g.weight.shape[0] * lm.args.hidden_size / min(ts) / 1e12, 2) if hasattr(_g, "weight") else None
ref_tf()
offs = list(range(0, a.T, a.chunk * a.stride))
plain = {}; refs = {}
for o in offs:                                               # plain pass
    ts = []
    for r in range(a.reps + 1):
        at(o); mx.synchronize(); t0 = time.perf_counter(); run(cache, ids[o:o + a.chunk]); ts.append(time.perf_counter() - t0)
    plain[o] = min(ts[1:]); refs[o] = ref_tf()
prep(); res = []
for o in offs:
    S.clear(); at(o); stack.clear(); mx.synchronize(); t0 = time.perf_counter(); run(cache, ids[o:o + a.chunk]); tt = time.perf_counter() - t0
    S["outside"] = tt - sum(v for k, v in S.items()); d = {k: round(v * 1e3, 2) for k, v in S.items()}
    r = {"o": o, "chunk": a.chunk, "plain_ms": round(plain[o] * 1e3, 1), "inst_ms": round(tt * 1e3, 1), "ref_tf": refs[o], "cls_ms": d}; res.append(r)
    print(json.dumps(r), flush=True)
    if a.out: open(a.out, "a").write(json.dumps(r) + "\n")
print(f"{a.kind} {a.path.split('/')[-1]} peak {mx.get_peak_memory()/2**30:.2f} GiB")

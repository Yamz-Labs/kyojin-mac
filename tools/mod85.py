"""Per-module scaling with N users, serial chain through the real layers (x <- x + mod_l(norm(x)), the module of every layer of its kind), interleaved over N.
Modules: mlp (32), gdn (24, batched states), attn (8, per-user TQ caches via RowsKV), head. ms per full chain.
usage: mod85.py KIND MODEL CTX USERS [PASSES] [MODS]"""
import sys, time, statistics
import mlx.core as mx
from kyojin_mlx import tq, multiuser as MU
import kvwin_kl as K
from mlx_lm.models.cache import KVCache
kind, mdir, CTX = sys.argv[1], sys.argv[2], int(sys.argv[3]); Ns = [int(v) for v in sys.argv[4].split(",")]
PASS = int(sys.argv[5]) if len(sys.argv) > 5 else 15
MODS = (sys.argv[6] if len(sys.argv) > 6 else "mlp,gdn,attn,head").split(",")
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    from transformers import AutoTokenizer
    model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
else:
    from mlx_lm import load
    model, tok = load(mdir)
MU.install(model)
lm = model.language_model; Ls = lm.model.layers
P0 = 1024
ids = K.tokens(tok, P0, 0)
base = K.cache_for(model, [6, 6, 0, 4, 256])
for s0 in range(0, P0, 512):
    h = lm.model(mx.array(ids[s0:s0 + 512])[None], base); mx.eval(h); mx.clear_cache()
reps = -(-CTX // P0)
def tiled(cache):
    c2 = MU.fork_cache(cache)
    for c in c2:
        if isinstance(c, tq.TQKVCache):
            st = c.state
            c.state = [mx.contiguous(mx.tile(x[:, :, :P0], (1, 1, reps) + (1,) * (x.ndim - 3))[:, :, :CTX]) for x in st[:4]] + list(st[4:])
    mx.eval([c.state for c in c2])
    return c2
bcs = {N: MU.batch_caches([tiled(base) for _ in range(N)]) for N in Ns}
H = lm.args.hidden_size
xs = {N: (mx.random.normal((N, 1, H)) * 0.1).astype(mx.bfloat16) for N in Ns}
def chain(N, mod):
    x, bc = xs[N], bcs[N]
    for i, l in enumerate(Ls):
        if mod == "mlp":
            x = x + l.mlp(l.post_attention_layernorm(x)) * 0.1
        elif mod == "gdn" and l.is_linear:
            x = x + l.linear_attn(l.input_layernorm(x), mask=None, cache=bc[i]) * 0.1
        elif mod == "attn" and not l.is_linear:
            x = x + l.self_attn(l.input_layernorm(x), mask=None, cache=bc[i]) * 0.1
    if mod == "head":
        hh = lm.model.norm(x)
        lg = lm.model.embed_tokens.as_linear(hh) if lm.args.tie_word_embeddings else lm.lm_head(hh)
        return mx.argmax(lg[:, -1], axis=-1)
    return x
def t(N, mod, reps_=3):
    r = []
    for _ in range(reps_):
        t0 = time.perf_counter(); mx.eval(chain(N, mod)); r.append((time.perf_counter() - t0) * 1e3)
    return min(r)
import os, re, collections, tempfile
if os.environ.get("CENSUS"):
    for N in Ns:
        for m in MODS:
            path = os.path.join(tempfile.gettempdir(), f"mod85_{os.getpid()}.dot"); mx.export_to_dot(path, chain(N, m)); txt = open(path).read(); os.unlink(path)
            c = collections.Counter(re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt))
            VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided", "StopGradient", "Depends"}
            print(f"CENSUS {m} N={N}: nodes {sum(c.values())} dispatches {sum(v for k, v in c.items() if k not in VIEWS)}  top: " + ", ".join(f"{k}:{v}" for k, v in c.most_common(14)), flush=True)
    sys.exit(0)
for N in Ns:
    for m in MODS: t(N, m)
res = {(N, m): [] for N in Ns for m in MODS}
for p in range(PASS):
    for N in (Ns if p % 2 == 0 else Ns[::-1]):
        for m in MODS: res[(N, m)].append(t(N, m))
for m in MODS:
    print(f"{kind} ctx={CTX} {m:5s} ms/chain (median/min): " + "  ".join(f"N{N} {statistics.median(res[(N,m)]):.2f}/{min(res[(N,m)]):.2f}" for N in Ns), flush=True)

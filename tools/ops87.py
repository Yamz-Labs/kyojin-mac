"""Per-op decode table at depth N (Qwen3.5-4B, mlx4 or exl3 pack): ablation timing (interleaved blocks, one process, synthetic depth by tiling a real
1K prefill), measured ms vs the roof (bytes / BW). usage: curve83_ops.py KIND(mlx|exl3) MODEL N CFG(fp16|tq6|tq5) [BLOCKS] [STEPS] [OUT_JSONL] [VARIANTS,..]
Roof: weight bytes of the ablated modules / BW, KV bytes read per token at depth N / BW (BW = 139 GB/s measured read, results_mac/macair10_roofline.json)."""
import sys, os, time, json
import mlx.core as mx
import mlx.nn as nn
from kyojin_mlx import kvwin as KW, tq
import kvwin_kl as K
from mlx_lm.models import qwen3_5 as Q5, qwen3_next as QN
from mlx_lm.models.cache import KVCache

BW = 139.3e9
kind, mdir, N, cfgname = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B = int(sys.argv[5]) if len(sys.argv) > 5 else 7
S = int(sys.argv[6]) if len(sys.argv) > 6 else 8
out = sys.argv[7] if len(sys.argv) > 7 else None
ALL = ["full", "no_sdpa", "no_sdpa_kv", "no_attnmod", "no_gdn_core", "no_gdnmod", "no_mlp", "no_head", "floor"]
variants = sys.argv[8].split(",") if len(sys.argv) > 8 else ALL
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    from transformers import AutoTokenizer
    model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
else:
    from mlx_lm import load
    model, tok = load(mdir)
lm = model.language_model; hd = K.head(lm)
if os.environ.get("FUSE"):
    import fuse_lin; globals()["fuse_lin"] = fuse_lin; print("fused groups:", fuse_lin.fuse_model(model, tuple(os.environ["FUSE"].split(","))))
cfg = {"fp16": [0, 0, 0, 0, 0], "tq6": [6, 6, 0, 4, 256], "tq5": [5, 5, 0, 4, 256]}[cfgname]
REAL = os.environ.get("REAL") == "1"                         # real prefill to N (default: 1K prefill tiled to N, the decode time does not depend on the KV values)
P0 = N if REAL else 1024
ids = K.tokens(tok, P0, 0)
cache = K.cache_for(model, cfg)
for s0 in range(0, P0, 1024):
    h = lm.model(mx.array(ids[s0:s0 + 1024])[None], cache); mx.eval(h); mx.clear_cache()
y = mx.argmax(hd(h[:, -1:])[0], axis=-1)
reps = -(-N // P0)
for c in ([] if REAL else cache):                                              # tile the 1K pool to N rows: decode time does not depend on the KV values
    if isinstance(c, tq.TQKVCache):
        st = c.state
        pool = [mx.tile(x[:, :, :P0], (1, 1, reps) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]]
        c.state = pool + list(st[4:])
    elif isinstance(c, KVCache):
        c.keys = mx.tile(c.keys[:, :, :P0], (1, 1, reps, 1))[:, :, :N]; c.values = mx.tile(c.values[:, :, :P0], (1, 1, reps, 1))[:, :, :N]; c.offset = N
mx.eval([c.state for c in cache]); mx.clear_cache()
kv_bytes = sum(x.nbytes for c in cache if isinstance(c, (tq.TQKVCache, KVCache)) for x in (c.state if c.state else []) if isinstance(x, mx.array))

ABL = set()
def wrap(cls, name, fn):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        return fn(self, orig, *a, **k)
    setattr(cls, name, w)
wrap(QN.Qwen3NextAttention, "__call__", lambda self, o, x, *a, **k: x * 1 if "no_attnmod" in ABL else o(self, x, *a, **k))
wrap(QN.Qwen3NextMLP, "__call__", lambda self, o, x: x * 1 if "no_mlp" in ABL else o(self, x))
wrap(Q5.GatedDeltaNet, "__call__", lambda self, o, x, *a, **k: x * 1 if "no_gdnmod" in ABL else o(self, x, *a, **k))
wrap(Q5.DecoderLayer, "__call__", lambda self, o, x, *a, **k: x * 1 if "floor" in ABL else o(self, x, *a, **k))
orig_gdu = Q5.gated_delta_update
Q5.gated_delta_update = lambda q, k, v, a, b, A, dt, st, m, use_kernel=True: (v, st) if "no_gdn_core" in ABL else orig_gdu(q, k, v, a, b, A, dt, st, m, use_kernel=use_kernel)
orig_sdpa = QN.scaled_dot_product_attention
def sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
    return q if ("no_sdpa" in ABL or "no_sdpa_kv" in ABL) else orig_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)
QN.scaled_dot_product_attention = sdpa
orig_att = tq.TQKVCache.attend
tq.TQKVCache.attend = lambda self, qs, scale: qs if ("no_sdpa" in ABL or "no_sdpa_kv" in ABL) else orig_att(self, qs, scale)
for cls in (KVCache, tq.TQKVCache):
    o = cls.update_and_fetch
    cls.update_and_fetch = (lambda o: lambda self, k, v: (k, v) if "no_sdpa_kv" in ABL else o(self, k, v))(o)

def step(yv):
    hh = lm.model(yv[None], cache)
    if "no_head" in ABL or "floor" in ABL:
        return mx.argmax(hh[:, :, :8], axis=-1)[0]
    return mx.argmax(hd(hh)[0], axis=-1)
def block():
    global y
    if os.environ.get("FUSE"): fuse_lin.ON[0] = "nofuse" not in ABL
    t0 = time.perf_counter()
    for _ in range(S):
        y = step(y); mx.eval(y)
    return (time.perf_counter() - t0) / S * 1e3
def build_ms():
    t0 = time.perf_counter(); r = step(y); t = (time.perf_counter() - t0) * 1e3; mx.eval(r); return t

# weight bytes per module class (roof)
def nb(m):
    from mlx.utils import tree_flatten
    t = sum(v.nbytes for _, v in tree_flatten(m.parameters()))
    for sub in m.modules():                                   # Exl3Linear keeps its arrays private (not module parameters)
        if hasattr(sub, "_words"): t += sub._words.nbytes + sub._suh.nbytes + sub._svh.nbytes + (sub._bias.nbytes if sub._bias is not None else 0)
    return t
Ls = lm.model.layers
wb = dict(attn_lin=sum(nb(l.self_attn) for l in Ls if not l.is_linear), gdn_all=sum(nb(l.linear_attn) for l in Ls if l.is_linear),
          mlp=sum(nb(l.mlp) for l in Ls), head=nb(lm.model.embed_tokens) if lm.args.tie_word_embeddings else nb(lm.lm_head))
gdn_state = 0
for c in cache:
    if not isinstance(c, (tq.TQKVCache, KVCache)):
        gdn_state += sum(x.nbytes for x in c.cache if isinstance(x, mx.array))
for _ in range(3):
    for v in variants[:1]: block()
res = {v: [] for v in variants}
for r in range(B):
    for v in variants:
        ABL.clear(); ABL.add(v)
        res[v].append(block())
ABL.clear(); print("BLOCKS", {v: [round(t,1) for t in x] for v, x in res.items()})
bt = build_ms()
def host_probe(v, n=10):
    """CPU time of build + async_eval encode (returns when the whole step is encoded) vs the full step time: encode ~ step => host bound."""
    global y
    ABL.clear(); ABL.add(v); enc, tot = [], []
    for _ in range(n + 2):
        t0 = time.perf_counter(); yy = step(y); mx.async_eval(yy); t1 = time.perf_counter(); mx.eval(yy); t2 = time.perf_counter(); y = yy
        enc.append((t1 - t0) * 1e3); tot.append((t2 - t0) * 1e3)
    ABL.clear(); return sorted(enc[2:])[n // 4], sorted(tot[2:])[n // 4]
hp = host_probe("full")
def block_async(n=16):
    """mlx-lm style: the next step is encoded while the GPU runs the current one."""
    global y
    ABL.clear(); t0 = time.perf_counter(); nxt = step(y); mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n: nxt = step(cur); mx.async_eval(nxt)
        cur.item()
    y = cur; return (time.perf_counter() - t0) / n * 1e3
ta = sorted(block_async() for _ in range(5))[1]
st = {v: dict(min=min(x), p25=sorted(x)[len(x) // 4]) for v, x in res.items()}
rec = dict(async_ms=ta, host_encode_ms=hp[0], host_total_ms=hp[1], kind=kind, cfg=cfgname, N=N, kv_bytes=kv_bytes, gdn_state_bytes=gdn_state, wbytes=wb, build_ms=bt, stats=st, peak_gib=mx.get_peak_memory() / 2**30)
print(json.dumps(rec), flush=True)
if out: open(out, "a").write(json.dumps(rec) + "\n")
g = lambda v: st[v]["p25"]
print(f"== {kind} {cfgname} N={N}  full {g('full'):.2f} ms ({1e3/g('full'):.1f} tok/s)  KV read/token {kv_bytes/1e6:.0f} MB -> roof {kv_bytes/BW*1e3:.2f} ms")
if len(variants) == len(ALL):
    rows = [("sdpa (attn core)", g("full") - g("no_sdpa"), kv_bytes / BW * 1e3),
            ("kv write/ring (update_and_fetch)", g("no_sdpa") - g("no_sdpa_kv"), 0.0),
            ("full-attn layer (proj+norm+rope+sdpa+kv)", g("full") - g("no_attnmod"), wb["attn_lin"] / BW * 1e3 + kv_bytes / BW * 1e3),
            ("GDN layers total", g("full") - g("no_gdnmod"), (wb["gdn_all"] + 2 * gdn_state) / BW * 1e3),
            ("  of which GDN recurrent core", g("full") - g("no_gdn_core"), 2 * gdn_state / BW * 1e3),
            ("MLP", g("full") - g("no_mlp"), wb["mlp"] / BW * 1e3),
            ("lm_head", g("full") - g("no_head"), wb["head"] / BW * 1e3),
            ("floor (embed+norm+host+sync, no layers/head)", g("floor"), 0.0)]
    for n, m, r_ in rows: print(f"{n:48s} measured {m:7.2f} ms   roof {r_:6.2f} ms")
    print(f"sum of weight bytes {sum(wb.values())/1e9:.2f} GB -> total roof {(sum(wb.values())+kv_bytes+2*gdn_state)/BW*1e3:.2f} ms; build(python graph) {bt:.1f} ms; host: encode {hp[0]:.1f} ms of {hp[1]:.1f} ms step; ASYNC (mlx-lm style) {ta:.2f} ms = {1e3/ta:.1f} tok/s")

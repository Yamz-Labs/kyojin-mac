"""Multi-user decode per-op table (Qwen3.5-4B/9B, mlx4 or exl3 pack): N users x one token per step on the batched multiuser path.
Ablation timing (interleaved blocks over all (N, variant) pairs, one process, KV of every user = a real 1K prefill tiled to CTX), measured ms vs the roof.
usage: mu85_ops.py KIND(mlx|kmlx|exl3) MODEL CTX CFG(fp16|tq6) USERS(1,2,3,4) [BLOCKS] [STEPS] [OUT_JSONL] [VARIANTS,..]
Roof of an N-user step: (weight bytes once + N x (KV bytes read + 2 x GDN state bytes)) / BW, BW = 139.3 GB/s measured read."""
import sys, os, time, json
import mlx.core as mx
from kyojin_mlx import kvwin as KW, tq, multiuser as MU
import kvwin_kl as K
from mlx_lm.models import qwen3_5 as Q5, qwen3_next as QN
from mlx_lm.models.cache import KVCache

BW = 139.3e9
kind, mdir, CTX, cfgname = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
Ns = [int(v) for v in sys.argv[5].split(",")]
B = int(sys.argv[6]) if len(sys.argv) > 6 else 7
S = int(sys.argv[7]) if len(sys.argv) > 7 else 8
out = sys.argv[8] if len(sys.argv) > 8 else None
ALL = ["full", "no_sdpa", "no_sdpa_kv", "no_attnmod", "no_gdn_core", "no_gdnmod", "no_mlp", "no_head", "floor"]
variants = sys.argv[9].split(",") if len(sys.argv) > 9 else ALL
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    from transformers import AutoTokenizer
    model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
elif kind == "kmlx":                                          # the served mlx4 path: mlxpack (step regions) + q4mma few-row linears
    from kyojin_mlx import mlxpack, q4mma as QM
    model, tok = mlxpack.load(mdir)
    if os.environ.get("KYOJIN_Q4MMA", "1") != "0": print("q4mma linears", QM.install(model))
else:
    from mlx_lm import load
    model, tok = load(mdir)
MU.install(model)
lm = model.language_model
cfg = {"fp16": [0, 0, 0, 0, 0], "tq6": [6, 6, 0, 4, 256]}[cfgname]
P0 = 1024
ids = K.tokens(tok, P0, 0)
base = K.cache_for(model, cfg)
for s0 in range(0, P0, 512):
    h = lm.model(mx.array(ids[s0:s0 + 512])[None], base); mx.eval(h); mx.clear_cache()
y0 = mx.argmax(MU.logits_last(model, mx.array(ids[-1:])[None], MU.fork_cache(base)), axis=-1)  # a plausible first token (value is irrelevant for timing)
reps = -(-CTX // P0)

def tiled(cache):
    c2 = MU.fork_cache(cache)
    for c in c2:
        if isinstance(c, tq.TQKVCache):
            st = c.state
            c.state = [mx.contiguous(mx.tile(x[:, :, :P0], (1, 1, reps) + (1,) * (x.ndim - 3))[:, :, :CTX]) for x in st[:4]] + list(st[4:])
        elif isinstance(c, KVCache):
            c.keys = mx.contiguous(mx.tile(c.keys[:, :, :P0], (1, 1, reps, 1))[:, :, :CTX]); c.values = mx.contiguous(mx.tile(c.values[:, :, :P0], (1, 1, reps, 1))[:, :, :CTX]); c.offset = CTX
    mx.eval([c.state for c in c2])
    return c2
state = {}
for N in Ns:
    users = [tiled(base) for _ in range(N)]
    state[N] = dict(bc=MU.batch_caches(users, prefix_len=0), y=mx.concatenate([y0] * N))
    if os.environ.get("MU95_PREFIX"):                      # variant "pool": the same users through mu_attn.PoolKV (shared prefix of MU95_PREFIX tokens)
        state[N]["bcp"] = MU.batch_caches([MU.fork_cache(u) for u in users], prefix_len=int(os.environ["MU95_PREFIX"]))
    mx.clear_cache()
    print(f"N={N} caches built, peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
c0 = tiled(base)
kv_bytes = sum(x.nbytes for c in c0 if isinstance(c, (tq.TQKVCache, KVCache)) for x in (c.state if c.state else []) if isinstance(x, mx.array))
gdn_state = sum(x.nbytes for c in c0 if not isinstance(c, (tq.TQKVCache, KVCache)) for x in c.cache if isinstance(x, mx.array))
del c0

ABL = set()
TOGGLES = {}                                                 # variant -> [(module, attr, value)]: module constants flipped for the block (graph is rebuilt each step)
if kind == "exl3":
    from kyojin_mlx import exl3_linear as EL
    TOGGLES = {"mma2": [(EL, "GEMV_MAX_ROWS", 1), (EL, "MMA_ROWS", 2)], "mma3": [(EL, "GEMV_MAX_ROWS", 2), (EL, "MMA_ROWS", 3)], "mma4": [(EL, "GEMV_MAX_ROWS", 3), (EL, "MMA_ROWS", 4)]}
    from kyojin_mlx import metal_src as MS
    TOGGLES.update({"half2": [(MS, "HALF_ROWS", 2)], "half3": [(MS, "HALF_ROWS", 3)], "half4": [(MS, "HALF_ROWS", 4)]})
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

def step(N):
    st = state[N]
    hh = lm.model(st["y"][:, None], st["bcp"] if ("pool" in ABL or os.environ.get("MU95_POOLALL")) and "bcp" in st else st["bc"])[:, -1:]
    if "no_head" in ABL or "floor" in ABL:
        r = mx.argmax(hh[:, 0, :8], axis=-1)
    else:
        lg = MU.head(lm, hh)
        r = mx.argmax(lg[:, -1], axis=-1)
    return r
PADA = mx.random.normal((2048, 2048)).astype(mx.float16); mx.eval(PADA)
def pad_work():
    return (PADA @ PADA) @ PADA
def block(N):
    st = state[N]
    sv = [(m, a, getattr(m, a)) for v in ABL for m, a, _ in TOGGLES.get(v, [])]
    for v in ABL:
        for m, a, x in TOGGLES.get(v, []): setattr(m, a, x)
    t0 = time.perf_counter()
    for _ in range(S):
        if "gpu_pad" in ABL:
            st["y"] = step(N); mx.eval(pad_work(), st["y"])
        elif "pad_only" in ABL:
            mx.eval(pad_work())
        else:
            st["y"] = step(N); mx.eval(st["y"])
    dt = (time.perf_counter() - t0) / S * 1e3
    for m, a, x in sv: setattr(m, a, x)
    return dt
def build_ms(N):
    t0 = time.perf_counter(); r = step(N); t = (time.perf_counter() - t0) * 1e3; mx.eval(r); state[N]["y"] = r; return t

def nb(m):
    from mlx.utils import tree_flatten
    t = sum(v.nbytes for _, v in tree_flatten(m.parameters()))
    for sub in m.modules():
        if hasattr(sub, "_words"): t += sub._words.nbytes + sub._suh.nbytes + sub._svh.nbytes + (sub._bias.nbytes if sub._bias is not None else 0)
    return t
Ls = lm.model.layers
wb = dict(attn_lin=sum(nb(l.self_attn) for l in Ls if not l.is_linear), gdn_all=sum(nb(l.linear_attn) for l in Ls if l.is_linear),
          mlp=sum(nb(l.mlp) for l in Ls), head=nb(lm.model.embed_tokens) if lm.args.tie_word_embeddings else nb(lm.lm_head))

for _ in range(3):
    for N in Ns: ABL.clear(); block(N)
res = {(N, v): [] for N in Ns for v in variants}
for r in range(B):
    order = Ns if r % 2 == 0 else Ns[::-1]
    for N in order:
        for v in variants:
            ABL.clear(); ABL.add(v)
            res[(N, v)].append(block(N))
ABL.clear()
def block_async(N, n=16):
    st = state[N]; t0 = time.perf_counter(); st["y"] = step(N); nxt = st["y"]; mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n: st["y"] = cur; nxt = step(N); mx.async_eval(nxt)
        mx.eval(cur)
    st["y"] = cur; return (time.perf_counter() - t0) / n * 1e3
def host_probe(N, n=10):
    """CPU time of build + async_eval encode (returns when the step is encoded) vs the full step: encode ~ step => host bound."""
    st = state[N]; enc, tot = [], []
    for _ in range(n + 2):
        t0 = time.perf_counter(); yy = step(N); mx.async_eval(yy); t1 = time.perf_counter(); mx.eval(yy); t2 = time.perf_counter(); st["y"] = yy
        enc.append((t1 - t0) * 1e3); tot.append((t2 - t0) * 1e3)
    return sorted(enc[2:])[n // 4], sorted(tot[2:])[n // 4]
for N in Ns:
    hp = host_probe(N)
    ta = sorted(block_async(N) for _ in range(5))[1]
    bt = build_ms(N)
    stt = {v: dict(min=min(res[(N, v)]), p25=sorted(res[(N, v)])[len(res[(N, v)]) // 4]) for v in variants}
    rec = dict(async_ms=ta, host_encode_ms=hp[0], host_total_ms=hp[1], kind=kind, cfg=cfgname, ctx=CTX, N=N, kv_bytes=kv_bytes, gdn_state_bytes=gdn_state, wbytes=wb, build_ms=bt, stats=stt, peak_gib=mx.get_peak_memory() / 2**30)
    print(json.dumps(rec), flush=True)
    if out: open(out, "a").write(json.dumps(rec) + "\n")
    g = lambda v: stt[v]["p25"]
    roof = lambda w, kv, gs: (w + N * (kv + 2 * gs)) / BW * 1e3
    print(f"== {kind} {cfgname} ctx={CTX} N={N}  full {g('full'):.2f} ms ({1e3*N/g('full'):.1f} agg tok/s)  async {ta:.2f} ms ({1e3*N/ta:.1f})  host encode {hp[0]:.1f} of {hp[1]:.1f} ms")
    if len(variants) == len(ALL):
        rows = [("sdpa (attn core)", g("full") - g("no_sdpa"), N * kv_bytes / BW * 1e3),
                ("kv write (update_and_fetch)", g("no_sdpa") - g("no_sdpa_kv"), 0.0),
                ("full-attn layer (proj+norm+rope+sdpa+kv)", g("full") - g("no_attnmod"), roof(wb["attn_lin"], kv_bytes, 0)),
                ("GDN layers total", g("full") - g("no_gdnmod"), roof(wb["gdn_all"], 0, gdn_state)),
                ("  of which GDN recurrent core", g("full") - g("no_gdn_core"), N * 2 * gdn_state / BW * 1e3),
                ("MLP", g("full") - g("no_mlp"), roof(wb["mlp"], 0, 0)),
                ("lm_head", g("full") - g("no_head"), roof(wb["head"], 0, 0)),
                ("floor (embed+norm+host+sync)", g("floor"), 0.0)]
        for n, m, r_ in rows: print(f"{n:44s} measured {m:7.2f} ms   roof {r_:6.2f} ms")
        print(f"weights {sum(wb.values())/1e9:.2f} GB, KV/user {kv_bytes/1e6:.0f} MB, GDN state/user {gdn_state/1e6:.1f} MB -> step roof {roof(sum(wb.values()), kv_bytes, gdn_state):.2f} ms; build {bt:.1f} ms")

"""roof96.py PACK CTX [BLOCKS] [STEPS]: per-op table of the plain one-row decode step (Qwen3.5 mlx pack, served step_regions path, fp16/bf16 KV).

Full-attention KV filled with random values up to CTX (timing only), GDN states from a real 256-token prefill.
Per op group (all layers of the step):
  iso_ms   the group's kernels alone, every layer's real weights and captured inputs, one eval (kernel throughput; independent kernels may overlap)
  abl_ms   step time minus the step with that group removed (compiled regions re-traced per variant, interleaved blocks): its in-step cost incl. gaps
  bytes    weights + scales + biases (linears), state read + write (GDN core), K + V read (SDPA); roof_ms = bytes / BW (BW measured here: 1 GiB read)
Plus: synced step ms, pipelined ms/token (async lookahead, served-like), dispatch census of the compiled step graph, sum(iso) vs step = idle/serial gap.
Prints one JSON line `ROOF96 {...}` and a markdown table."""
import os, sys, json, time, statistics, collections
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("../tests", "..", ".")]
import mlx.core as mx
import mlx.nn as nn
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import step_regions as SR, sdpa_dec as SD
from mlx_lm.models import qwen3_5 as Q5, qwen3_next as QN, base as _base

PACK, CTX = sys.argv[1], int(sys.argv[2])
BLOCKS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
ONLY = os.environ.get("ROOF96_VARIANTS")

# ---- read bandwidth (1 GiB, best of 5)
big = mx.random.randint(0, 2**31 - 1, (1 << 28,)).astype(mx.uint32); mx.eval(big)
bt = []
for _ in range(6):
    t0 = time.perf_counter(); mx.eval(big.max()); bt.append(time.perf_counter() - t0)
BW = big.nbytes / min(bt[1:]); del big; mx.clear_cache()

be, _ = load_backend(PACK, mtp=False, log=lambda *a: None)
lm = be.lm; tm = lm.model
L = tm.layers
attn_i = [i for i, l in enumerate(L) if not l.is_linear]
gdn_i = [i for i, l in enumerate(L) if l.is_linear]
cache = be.make_cache()
be.prefill_chunk(cache, list(range(1000, 1256)))
ca = [cache[i] for i in attn_i]
_, HK, _, DK = ca[0].keys.shape; DV = ca[0].values.shape[3]


def fill(N):
    A = -(-(N + 64) // 256) * 256
    for c in ca:
        dt = c.keys.dtype
        c.keys = mx.random.normal((1, HK, A, DK)).astype(dt); c.values = mx.random.normal((1, HK, A, DV)).astype(dt); c.offset = N
    mx.eval(*[a for c in ca for a in (c.keys, c.values)])


fill(CTX)
TOK = mx.array([[1000]], dtype=mx.int32)


def trim(n):
    for c in ca:
        c.trim(n)


def kv():
    return [a for c in ca for a in (c.keys, c.values)]


# ---- module names
name_of = {}
for n, m in lm.named_modules():
    name_of[id(m)] = n


def group_of_linear(n):
    if n.endswith("in_proj_qkv"): return "gdn.in_qkv"
    if n.endswith("in_proj_z"): return "gdn.in_z"
    if n.endswith("in_proj_b") or n.endswith("in_proj_a"): return "gdn.in_ba"
    if n.endswith("linear_attn.out_proj"): return "gdn.out"
    if n.endswith("q_proj"): return "attn.q"
    if n.endswith("k_proj") or n.endswith("v_proj"): return "attn.kv"
    if n.endswith("o_proj"): return "attn.o"
    if n.endswith("gate_proj") or n.endswith("up_proj"): return "mlp.gate_up"
    if n.endswith("down_proj"): return "mlp.down"
    if n.endswith("lm_head"): return "head"
    return "other_lin:" + n


ABL = set()          # active ablations (checked at trace time inside compiled regions, at run time in the eager attention core)
CAP = None           # capture list or None
_qlc = nn.QuantizedLinear.__call__


def qlin_call(self, x):
    g = group_of_linear(name_of.get(id(self), "?"))
    if CAP is not None:
        CAP.append((g, self, x))
    if g in ABL:                                 # a broadcast view of the input keeps the producers live (zeros would let compile drop them)
        return mx.broadcast_to(x[..., :1], (*x.shape[:-1], self.weight.shape[0]))
    return _qlc(self, x)


nn.QuantizedLinear.__call__ = qlin_call
_emb_lin = nn.QuantizedEmbedding.as_linear


def emb_lin(self, x):
    if CAP is not None:
        CAP.append(("head", self, x))
    if "head" in ABL:
        return mx.broadcast_to(x[..., :1], (*x.shape[:-1], self.weight.shape[0]))
    return _emb_lin(self, x)


nn.QuantizedEmbedding.as_linear = emb_lin
_gdu = Q5.gated_delta_update


def gdu(q, k, v, a, b, A, dt, st, m, use_kernel=True):
    if CAP is not None:
        CAP.append(("gdn.core", None, (q, k, v, a, b, A, dt, st, m)))
    if "gdn.core" in ABL:
        return v, st
    return _gdu(q, k, v, a, b, A, dt, st, m, use_kernel=use_kernel)


Q5.gated_delta_update = gdu
_gnorm = QN.Qwen3NextRMSNormGated.__call__


def gnorm(self, x, gate=None):
    if CAP is not None:
        CAP.append(("gdn.gnorm", self, (x, gate)))
    if "gdn.gnorm" in ABL:
        return x + gate * 0
    return _gnorm(self, x, gate)


QN.Qwen3NextRMSNormGated.__call__ = gnorm
_conv = nn.Conv1d.__call__


def conv(self, x):
    if CAP is not None:
        CAP.append(("gdn.conv", self, x))
    if "gdn.conv" in ABL:
        return x[:, -1:, :]
    return _conv(self, x)


nn.Conv1d.__call__ = conv
_rms = nn.RMSNorm.__call__


def rms(self, x):
    if CAP is not None:
        CAP.append(("norm", self, x))
    if "norm" in ABL:
        return x
    return _rms(self, x)


nn.RMSNorm.__call__ = rms
_sdpa = _base.scaled_dot_product_attention     # sdpa_dec wrapper (installed by the backend)


def sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
    if "sdpa" in ABL:
        return q[..., :v.shape[-1]] * 0
    return _sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)


_base.scaled_dot_product_attention = sdpa
_upd = type(ca[0]).update_and_fetch


def upd(self, k, v):
    if "kvwrite" in ABL:
        self.offset += 1
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
    return _upd(self, k, v)


type(ca[0]).update_and_fetch = upd


def step():
    lg = be._head(tm(TOK, cache)[:, -1:]).astype(mx.float32)
    return mx.argmax(lg[0, -1], -1)


def step_sync():
    y = step(); mx.eval(y, *kv()); trim(1)


# ---- capture one stock step (uncompiled) for the isolated op timings, GDN states restored afterwards
gdn0 = [[mx.array(x) for x in cache[i].cache] for i in gdn_i]
CAP = []
y = be._head(tm._kr_orig(TOK, cache)[:, -1:]); mx.eval(y)
caps = CAP; CAP = None
trim(1)
for j, i in enumerate(gdn_i):
    cache[i].cache = [mx.array(x) for x in gdn0[j]]
flat_in = []
for g, m, x in caps:
    flat_in += list(x) if isinstance(x, tuple) else [x]
mx.eval(*[a for a in flat_in if isinstance(a, mx.array)])


def nb_lin(m):
    return sum(getattr(m, k).nbytes for k in ("weight", "scales", "biases") if getattr(m, k, None) is not None)


groups = collections.OrderedDict()
for g, m, x in caps:
    d = groups.setdefault(g, dict(calls=[], bytes=0))
    d["calls"].append((m, x))
    if g in ("head",) or isinstance(m, nn.QuantizedLinear):
        d["bytes"] += nb_lin(m)
    elif g == "gdn.core":
        d["bytes"] += 2 * x[7].nbytes + sum(t.nbytes for t in x[:5])
    elif g == "gdn.conv":
        d["bytes"] += m.weight.nbytes + x.nbytes
    elif g == "gdn.gnorm":
        d["bytes"] += m.weight.nbytes + x[0].nbytes * 3
    elif g == "norm":
        d["bytes"] += m.weight.nbytes + x.nbytes * 2


def run_group(g):
    outs = []
    for m, x in groups[g]["calls"]:
        if g == "head": outs.append(_emb_lin(m, x))
        elif g == "gdn.core": outs.extend(_gdu(*x))
        elif g == "gdn.gnorm": outs.append(_gnorm(m, *x))
        elif g == "gdn.conv": outs.append(_conv(m, x))
        elif g == "norm": outs.append(_rms(m, x))
        else: outs.append(_qlc(m, x))
    return outs


kvb = sum(c.offset * HK * (DK + DV) * c.keys.dtype.size for c in ca)
a0 = L[attn_i[0]].self_attn
qs = mx.random.normal((1, a0.num_attention_heads, 1, DK)).astype(ca[0].keys.dtype); mx.eval(qs)


def run_sdpa():
    return [_sdpa(qs, c.keys[..., :c.offset, :], c.values[..., :c.offset, :], cache=c, scale=a0.scale, mask=None) for c in ca]


def tmin(f, n=9):
    ts = []
    for _ in range(n + 2):
        t0 = time.perf_counter(); mx.eval(*f()); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts[2:]), statistics.median(ts[2:])


floor = tmin(lambda: [mx.add(qs, 1)])[0]           # one tiny eval: commit + wait floor
iso = {}
for g in groups:
    b, m = tmin(lambda g=g: run_group(g))
    iso[g] = dict(ms=round(b - floor, 3), med=round(m - floor, 3), n=len(groups[g]["calls"]), bytes=groups[g]["bytes"])
b, m = tmin(run_sdpa)
iso["sdpa"] = dict(ms=round(b - floor, 3), med=round(m - floor, 3), n=len(ca), bytes=kvb)

# ---- census of the compiled step graph (dispatches = primitive nodes minus views)
from count_launches import graph_census, VIEWS
cen = graph_census(step()); trim(1)
for j, i in enumerate(gdn_i):
    cache[i].cache = [mx.array(x) for x in gdn0[j]]

# ---- ablation variants: one compiled plan per variant
variants = ["full", "head", "mlp.gate_up", "mlp.down", "gdn.in_qkv", "gdn.in_z", "gdn.in_ba", "gdn.out", "gdn.core", "gdn.conv", "gdn.gnorm",
            "attn.q", "attn.kv", "attn.o", "norm", "sdpa", "kvwrite", "all_lin"]
if ONLY:
    variants = ["full"] + [v for v in ONLY.split(",") if v != "full"]
LIN = ["head", "mlp.gate_up", "mlp.down", "gdn.in_qkv", "gdn.in_z", "gdn.in_ba", "gdn.out", "attn.q", "attn.kv", "attn.o"]
plans = {}


def abl_set(v):
    return set() if v == "full" else (set(LIN) if v == "all_lin" else {v})


for v in variants:
    ABL.clear(); ABL.update(abl_set(v))
    tm._kr_plan = SR._Plan(tm)
    step_sync(); step_sync()
    plans[v] = tm._kr_plan
ABL.clear()


def block(v):
    tm._kr_plan = plans[v]; ABL.clear(); ABL.update(abl_set(v))
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step_sync()
    dt = (time.perf_counter() - t0) / STEPS * 1e3
    ABL.clear(); return dt


def pipe(n=16):
    tm._kr_plan = plans["full"]
    y = step(); mx.async_eval(y)
    t0 = time.perf_counter()
    for _ in range(n):
        yn = mx.argmax(be._head(tm(y.reshape(1, 1).astype(mx.int32), cache)[:, -1:]).astype(mx.float32)[0, -1], -1); mx.async_eval(yn); y.item(); y = yn
    y.item(); dt = (time.perf_counter() - t0) * 1e3 / n; trim(n + 1); return dt


res = {v: [] for v in variants}; pp = []
for r in range(BLOCKS):
    for v in (variants if r % 2 == 0 else variants[::-1]):
        res[v].append(block(v))
    pp.append(pipe())
st = {v: (round(min(x), 3), round(statistics.median(x), 3)) for v, x in res.items()}
full = st["full"][1]
out = dict(pack=os.path.basename(PACK.rstrip("/")), ctx=CTX, bw_GBs=round(BW / 1e9, 1), floor_ms=round(floor, 3), step_sync=st["full"],
           pipe_ms=(round(min(pp), 3), round(statistics.median(pp), 3)), census=dict(dispatches=cen["dispatches"], by_type={k: v for k, v in cen["by_type"].items() if k not in VIEWS}),
           iso=iso, abl={v: round(full - st[v][1], 3) for v in variants if v != "full"}, abl_raw=st)
print("ROOF96", json.dumps(out), flush=True)
tot_b = sum(d["bytes"] for d in iso.values())
print(f"\n{out['pack']} ctx {CTX}: step sync {st['full'][0]:.2f}/{full:.2f} ms (best/med), pipelined {out['pipe_ms'][0]:.2f}/{out['pipe_ms'][1]:.2f} ms/tok, "
      f"BW {BW/1e9:.1f} GB/s, roof {tot_b/BW*1e3:.2f} ms, dispatches {cen['dispatches']}, sum iso {sum(d['ms'] for d in iso.values()):.2f} ms")
print("| op group | n | MB | roof ms | iso ms | iso GB/s | iso % roof | in-step ms (abl) |\n|---|---|---|---|---|---|---|---|")
for g, d in iso.items():
    r_ = d["bytes"] / BW * 1e3
    print(f"| {g} | {d['n']} | {d['bytes']/1e6:.1f} | {r_:.3f} | {d['ms']:.3f} | {d['bytes']/max(d['ms'],1e-6)/1e6:.0f} | {100*r_/max(d['ms'],1e-6):.0f} | {out['abl'].get(g, float('nan')):.3f} |")
if "all_lin" in out["abl"]:
    print(f"all linears removed: -{out['abl']['all_lin']:.2f} ms; kvwrite -{out['abl'].get('kvwrite', float('nan')):.3f} ms")

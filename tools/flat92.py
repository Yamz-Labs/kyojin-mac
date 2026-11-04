"""flat92.py CTX[,CTX...] [REPS]: where the decode step's depth cost goes (plain 1 row and MTP verify T rows), Qwen3.5 pack of KYOJIN_Q35MLX.

The full-attention caches are filled with random fp16 keys / values up to CTX (timing only: the kernels do not depend on the values; no prefill,
so 64K fits a 3 min job). The gated-delta states come from a real 256-token prefill. Per CTX, best + median ms (interleaved reps):
  fast        plain step through step_regions (solo KVCache, what prof89 times)
  served      plain step through the served path (engine batch -> RowsKV, backend.step)
  fast_z / served_z / v5_z   the same with every attention core replaced by zeros (SDPA read removed, KV write / rope / mask / graph kept)
  v5          MTP verify pass of 5 rows (_verify), trimmed back
  draft1      one MTP-layer draft row (draft KV filled to CTX too)
  sdpa1 / sdpa5   the attention cores alone, every full-attention layer, T = 1 / 5 (bytes = K + V read, GB/s)
  kvw1 / kvw1_shared   KV append of one row in every layer, buffer owned by the cache / buffer also held by another reference (copy on write)
  sync_parts  synced split of the fast step: regions (compiled), rope, kv write, mask, sdpa, head
depth cost outside SDPA = X_z(CTX) - X_z(first CTX)."""
import os, sys, json, time, statistics
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", s) for s in ("tests", "")]
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import sdpa_dec as SD
from kyojin_mlx.step_regions import _forward  # noqa: F401  (import check)
from mlx_lm.models import base as _base
from mlx_lm.models.base import create_attention_mask

CTXS = [int(x) for x in sys.argv[1].split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
ONLY = os.environ.get("FLAT92_ONLY")
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
tm = be.lm.model
attn_i = [i for i, l in enumerate(tm.layers) if not l.is_linear]
ids = list(range(1000, 1256))
cache = be.make_cache(); d = be.new_draft()
be.prefill_chunk(cache, ids, draft=d)
ca = [cache[i] for i in attn_i]
dt = ca[0].keys.dtype
_, HK, _, DK = ca[0].keys.shape
DV = ca[0].values.shape[3]
gdn0 = [[mx.array(x) for x in cache[i].cache] for i in range(len(cache)) if i not in attn_i]


def fill(N):
    A = -(-(N + 64) // 256) * 256
    for c in ca + ([d.mc] if d is not None else []):
        cdt = c.keys.dtype                      # trunk bf16 / fp16 per pack; the MTP layer runs fp16
        c.keys = mx.random.normal((1, HK, A, DK)).astype(cdt)
        c.values = mx.random.normal((1, HK, A, DV)).astype(cdt)
        c.offset = N
    if d is not None:
        d.mv = N
    mx.eval(*[a for c in ca for a in (c.keys, c.values)], *([d.mc.keys, d.mc.values] if d is not None else []))


def trim(n):
    for c in ca:
        c.trim(n)


def state():
    return [a for c in ca for a in (c.keys, c.values)]


TOK = mx.array([[1000]], dtype=mx.int32)


def fast():
    lg = be._head(tm(TOK, cache)[:, -1:]).astype(mx.float32); mx.eval(lg, *state()); trim(1)


def served():
    bc = be.batch([cache])
    lg = Q35Step(bc); mx.eval(lg, *state()); trim(1)


def Q35Step(bc):
    from kyojin_mlx.serve.backends import Q35Backend
    return Q35Backend.step(be, bc, TOK[0])


bcs = be.batch_spec([cache])


def verify(T_):
    def f():
        p = be._verify(bcs, [mx.array([1000 + i for i in range(T_)], dtype=mx.int32)]); mx.eval(p.logits, *state()); trim(T_)
    return f


def draft1():
    y = be._mtp_rows(d, mx.array([1000], dtype=mx.int32), d.pend[-1][:, -1:]); head = be.dhead if be.dhead is not None else be._head
    mx.eval(mx.argmax(head(y)[0, 0], -1), d.mc.keys); d.mc.trim(1)


def sdpa_only(T_):
    a0 = tm.layers[attn_i[0]].self_attn
    q = mx.random.normal((1, a0.num_attention_heads, T_, DK)).astype(dt); mx.eval(q)
    def f():
        outs = []
        for c in ca:
            k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            o = SD.route(q, k, v, c, a0.scale, None if T_ == 1 else "causal")
            outs.append(o if o is not None else mx.fast.scaled_dot_product_attention(q, k, v, scale=a0.scale, mask=None if T_ == 1 else "causal"))
        mx.eval(*outs)
    return f


K1 = mx.random.normal((1, HK, 1, DK)).astype(dt); V1 = mx.random.normal((1, HK, 1, DV)).astype(dt); mx.eval(K1, V1)
HOLD = []


def kvw(shared):
    def f():
        HOLD.clear()
        if shared:
            HOLD.extend(c.keys[...] for c in ca); HOLD.extend(c.values[...] for c in ca); mx.eval(*HOLD)
        for c in ca:
            c.update_and_fetch(K1, V1)
        mx.eval(*state()); trim(1); HOLD.clear()
    return f


def pipe(n=16):
    """Engine-like decode: step k+1 is built and launched (async) before step k is read (served lookahead), argmax sampling. ms per token."""
    def f():
        bc = be.batch([cache]); y = mx.array([1000], dtype=mx.int32); mx.async_eval(y)
        t0 = time.perf_counter()
        for _ in range(n):
            yn = mx.argmax(Q35Step_t(bc, y), -1).astype(mx.int32); mx.async_eval(yn); y.tolist(); y = yn
        y.tolist(); PIPE.append((time.perf_counter() - t0) * 1e3 / n); trim(n)
    return f


def Q35Step_t(bc, y):
    from kyojin_mlx.serve.backends import Q35Backend
    return Q35Backend.step(be, bc, y)


PIPE = []
SYNC = {}


def sync_parts():
    """The step_regions forward with a sync after every group."""
    plan = tm._kr_plan; L = plan.layers; x = out = gate = None
    acc = {k: 0.0 for k in ("regions", "rope", "kvwrite", "mask", "sdpa", "head")}
    def tick(k, *arrs):
        mx.eval(*arrs); t = time.perf_counter(); acc[k] += (t - tick.t) * 1e3; tick.t = t
    tick.t = time.perf_counter()
    for r, (gl, fn) in enumerate(plan.regions):
        st = [s for i in gl for s in (cache[i][0], cache[i][1])]
        res = fn(*((TOK, *st) if r == 0 else (x, out, gate, *st)))
        ns = 1 if plan.regions[r] is plan.regions[-1] else 5
        head, new = res[:ns], res[ns:]
        for j, i in enumerate(gl):
            cache[i][0], cache[i][1] = new[2 * j], new[2 * j + 1]
        tick("regions", *res)
        if ns == 1:
            lg = be._head(head[0]).astype(mx.float32); tick("head", lg); break
        x, q, k, v, gate = head
        ai = plan.attn[r]; c, a = cache[ai], L[ai].self_attn
        q = a.rope(q, offset=c.offset); k = a.rope(k, offset=c.offset); tick("rope", q, k)
        k, v = c.update_and_fetch(k, v); tick("kvwrite", c.keys, c.values)
        mask = create_attention_mask(x, c); tick("mask", *([mask] if isinstance(mask, mx.array) else []))
        o = _base.scaled_dot_product_attention(q, k, v, cache=c, scale=a.scale, mask=mask)
        out = o.transpose(0, 2, 1, 3).reshape(1, 1, -1); tick("sdpa", out)
    trim(1)
    for gi, i in enumerate([i for i in range(len(cache)) if i not in attn_i]):
        cache[i].cache = [mx.array(x) for x in gdn0[gi]]
    for k, v in acc.items():
        SYNC.setdefault(k, []).append(v)


_route = SD.route


def zero_route(q, k, v, *a, **kw):
    return q[..., :v.shape[-1]] * 0


def zeroed(f):
    def g():
        SD.route = zero_route
        try:
            f()
        finally:
            SD.route = _route
    return g


fs = {"fast": fast, "served": served, "fast_z": zeroed(fast), "served_z": zeroed(served), "v5": verify(5), "v5_z": zeroed(verify(5)),
      "draft1": draft1, "draft1_z": zeroed(draft1), "sdpa1": sdpa_only(1), "sdpa5": sdpa_only(5), "kvw1": kvw(False), "kvw1_shared": kvw(True), "sync_parts": sync_parts, "pipe16": pipe(16)}
if ONLY:
    fs = {k: v for k, v in fs.items() if k in ONLY.split(",")}
for N in CTXS:
    fill(N); SYNC.clear(); PIPE.clear()
    for f in fs.values():
        f(); f()
    SYNC.clear(); PIPE.clear()
    res = {k: [] for k in fs}
    for _ in range(REPS):
        for k, f in fs.items():
            t0 = time.perf_counter(); f(); res[k].append((time.perf_counter() - t0) * 1e3)
    kvb = sum(c.offset * HK * (DK + DV) * mx.array(0, dtype=dt).itemsize for c in ca)
    out = {k: [round(min(v), 2), round(statistics.median(v), 2)] for k, v in res.items()}
    for k in ("sdpa1", "sdpa5"):
        if k in out:
            out[k + "_GBs"] = round(kvb / out[k][0] / 1e6, 1)
    if PIPE:
        out["pipe_ms_per_tok"] = [round(min(PIPE), 2), round(statistics.median(PIPE), 2)]; PIPE.clear()
    out["sync"] = {k: round(statistics.median(v), 2) for k, v in SYNC.items()}
    print("FLAT92", os.path.basename(T.PACK), N, "layers", len(ca), "kv_MB", round(kvb / 1e6), json.dumps(out), flush=True)

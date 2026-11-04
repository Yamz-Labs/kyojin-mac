"""step107.py CTX[,CTX...] [REPS]: in-model decode step at depth, fp16 KV vs TQ6 KV (tq6w 6:6:4:256) in ONE process, interleaved reps
(Qwen3.5 pack of KYOJIN_Q35MLX). Full-attention caches are synthetic (random fp16 K / V, a TQ pool tiled from 1024 random quantized rows; timing only,
no prefill, so 128K fits a short job); the gated-delta states come from a real 256-token prefill. Per CTX and mode, best / median ms:
  plain   one-row trunk step (model + head), trimmed back
  v5      MTP verify pass of 5 rows (backend._verify), trimmed back
  plain_z / v5_z   the same with every attention core (cache.attend / sdpa) returning zeros: what the step costs outside the attention read
The MTP draft layer keeps an fp16 cache in both modes (not timed here). Env STEP107_MODES (default fp16,tq) may add tq%KNOB=V (tq module knob)."""
import os, sys, json, time, statistics
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", s) for s in ("tests", "")]
os.environ.pop("KYOJIN_TQ", None)
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import tq

CTXS = [int(x) for x in sys.argv[1].split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
MODES = os.environ.get("STEP107_MODES", "fp16,tq").split(",")
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
tm = be.lm.model
attn_i = [i for i, l in enumerate(tm.layers) if not l.is_linear]
ids = list(range(1000, 1256))
caches = {}
for m in ("fp16", "tq"):
    if m == "tq":
        os.environ["KYOJIN_TQ"] = "6:6:4:256"
    c = be.make_cache(); be.prefill_chunk(c, ids, draft=be.new_draft()); mx.eval([a for x in c for a in x.state])
    caches[m] = c
os.environ.pop("KYOJIN_TQ", None)
a0 = caches["fp16"][attn_i[0]]
dt = a0.keys.dtype
_, HK, _, DK = a0.keys.shape
print("attn layers", len(attn_i), "Hk", HK, "D", DK, "tq cache", type(caches["tq"][attn_i[0]]).__name__, flush=True)


def fill(N):
    for i in attn_i:
        c = caches["fp16"][i]
        A = -(-(N + 64) // 256) * 256
        c.keys = mx.random.normal((1, HK, A, DK)).astype(dt); c.values = mx.random.normal((1, HK, A, DK)).astype(dt); c.offset = N
    mx.random.seed(2)
    base = tq.TQKVCache(6, 6, 4, 256); base.update_and_fetch(*(mx.random.normal((1, HK, 1024, DK)).astype(dt) for _ in range(2))); st = base.state
    r = -(-N // 1024)
    for i in attn_i:
        caches["tq"][i].state = [mx.tile(x, (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
    mx.eval([a for m in caches for i in attn_i for a in caches[m][i].state]); mx.clear_cache()


TOK = mx.array([[1000]], dtype=mx.int32)
VT = [mx.array([1000 + i for i in range(5)], dtype=mx.int32)]


def trim(c, n):
    for i in attn_i:
        c[i].trim(n)


def st(c):
    return [a for i in attn_i for a in ((c[i].kq[0], c[i].kq[1], c[i].vq[0], c[i].vq[1], c[i].rk, c[i].rv) if hasattr(c[i], "kq") else (c[i].keys, c[i].values))]


def plain(m):
    c = caches[m]
    def f():
        lg = be._head(tm(TOK, c)[:, -1:]).astype(mx.float32); mx.eval(lg, *st(c)); trim(c, 1)
    return f


from kyojin_mlx import step_regions as SR
TOK1 = mx.array([1000], dtype=mx.int32)


def srv(m, wide=True):
    """The served solo plain step: backend.step on backend.batch([cache]) (multiuser RowsKV of one cache), step_regions WIDE on / off."""
    c = caches[m]
    def f():
        w0 = SR.WIDE; SR.WIDE = wide
        try:
            lg = be.step(be.batch([c]), TOK1); mx.eval(lg, *st(c)); trim(c, 1)
        finally:
            SR.WIDE = w0
    return f


def v5(m):
    c = caches[m]; bc = be.batch_spec([c])
    def f():
        p = be._verify(bc, [VT[0]]); mx.eval(p.logits, *st(c)); trim(c, 5)
    return f


_att = tq.TQKVCache.attend
from kyojin_mlx import sdpa_dec as SD
_route = SD._route
_sdpa = mx.fast.scaled_dot_product_attention


def zeroed(f):
    def g():
        tq.TQKVCache.attend = lambda self, q, s: mx.zeros_like(q)
        SD._route = lambda q, *a, **k: mx.zeros_like(q)
        try:
            f()
        finally:
            tq.TQKVCache.attend = _att; SD._route = _route
    return g


def knobbed(m, f):
    if "%" not in m:
        return f
    kv = [p.split("=") for p in m.split("%")[1:]]
    def g():
        old = {k: getattr(tq, k) for k, _ in kv}
        for k, v in kv: setattr(tq, k, int(v))
        try:
            f()
        finally:
            for k, v in old.items(): setattr(tq, k, v)
    return g


_wp = tq.TQKVCache._write_pool
_uf = tq.TQKVCache.update_and_fetch
from kyojin_mlx.kvwin import WinKVCache


def _norot(self, keys, values):
    self._src = (keys.astype(mx.float32), values.astype(mx.float32)); return WinKVCache.update_and_fetch(self, keys, values)


def stubbed(f, nowrite=False, norot=False):
    def g():
        if nowrite: tq.TQKVCache._write_pool = lambda self, p, k, v: None
        if norot: tq.TQKVCache.update_and_fetch = _norot
        try:
            f()
        finally:
            tq.TQKVCache._write_pool = _wp; tq.TQKVCache.update_and_fetch = _uf
    return g


SRV = os.environ.get("STEP107_SRV") == "1"               # + srv / srv_w0 / srv_z: the served solo plain step, step_regions WIDE on / off / zeroed attention
PIPE = int(os.environ.get("STEP107_PIPE", "0"))         # > 0: + the served regime, PIPE solo plain steps pipelined (next graph built while the GPU runs the current one,
# like the engine lookahead) -> ms per token, and the host build time of one step (graph creation, no eval); WIDE off and on


def pipe(m, wide):
    c = caches[m]
    def f():
        w0 = SR.WIDE; SR.WIDE = wide
        try:
            bc = be.batch([c]); t = TOK1; b = 0.0
            mx.synchronize(); t0 = time.perf_counter()
            for i in range(PIPE):
                tb = time.perf_counter(); lg = be.step(bc, t); nt = mx.argmax(lg, axis=-1).astype(mx.int32); b += time.perf_counter() - tb
                mx.async_eval(nt)
                if i:
                    prev.item()
                prev, t = nt, nt
            prev.item(); dt_ = (time.perf_counter() - t0) / PIPE
            trim(c, PIPE)
        finally:
            SR.WIDE = w0
        return dt_ * 1e3, b / PIPE * 1e3
    return f


STUBS = os.environ.get("STEP107_STUBS") == "1"           # + tq plain_z without the pool write / without the K V rotation (where the fixed TQ cost goes)
for N in CTXS:
    fill(N)
    fns = {}
    if STUBS:
        fns["tq:plain_z_nowrite"] = stubbed(zeroed(plain("tq")), nowrite=True)
        fns["tq:plain_z_norot"] = stubbed(zeroed(plain("tq")), norot=True)
        fns["tq:plain_z_neither"] = stubbed(zeroed(plain("tq")), nowrite=True, norot=True)
        fns["tq:v5_z_nowrite"] = stubbed(zeroed(v5("tq")), nowrite=True)
        fns["tq:v5_z_norot"] = stubbed(zeroed(v5("tq")), norot=True)
        fns["tq:v5_z_neither"] = stubbed(zeroed(v5("tq")), nowrite=True, norot=True)
    for m in MODES:
        b = m.split("%")[0]
        fns[m + ":plain"] = knobbed(m, plain(b)); fns[m + ":v5"] = knobbed(m, v5(b))
        fns[m + ":plain_z"] = zeroed(plain(b)); fns[m + ":v5_z"] = zeroed(v5(b))
        if SRV:
            fns[m + ":srv"] = knobbed(m, srv(b)); fns[m + ":srv_w0"] = knobbed(m, srv(b, False)); fns[m + ":srv_z"] = zeroed(srv(b))
    for f in fns.values():
        f(); f()
    res = {k: [] for k in fns}
    for _ in range(REPS):
        for k, f in fns.items():
            mx.synchronize(); t0 = time.perf_counter(); f(); res[k].append((time.perf_counter() - t0) * 1e3)
    if PIPE:
        pf = {f"{m}:pipe_w{int(w)}": pipe(m, w) for m in ("fp16", "tq") for w in (False, True)}
        for f in pf.values():
            f()
        pr = {k: [] for k in pf}
        for _ in range(max(REPS // 2, 2)):
            for k, f in pf.items():
                pr[k].append(f())
        for k, x in pr.items():
            res[k] = [a for a, _ in x]; res[k.replace("pipe", "build")] = [b for _, b in x]
    out = {k: [round(min(x), 2), round(statistics.median(x), 2)] for k, x in res.items()}
    for k in out:
        print(f"N={N} {k:16s} best {out[k][0]:7.2f} med {out[k][1]:7.2f} ms", flush=True)
    print("STEP107", N, os.path.basename(T.PACK), json.dumps(out), flush=True)

"""ab96.py PACK [CTX] [BLOCKS] [STEPS] [NKL]: decfuse A/B on the plain 1-row decode step (compiled step regions), one process, interleaved.

Variants = decfuse classes switched at trace time (each gets its own compiled plan): off, proj, addnorm, gdn, all.
Quality: NKL teacher-forced decode steps on real text (corpus ids) after a 512-token prefill, every variant vs the stock uncompiled forward
(tm._kr_orig): KL mean / max, top-1 agreement, bit-equal step count. Speed: synced step and pipelined ms/token at CTX (random KV fill),
dispatch census of the step graph. Prints AB96 JSON + a table."""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx

from kyojin_mlx import decfuse as DF
from kyojin_mlx import step_regions as SR
from kyojin_mlx.serve.backends import load_backend
from count_launches import graph_census

PACK = sys.argv[1]
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
BLOCKS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
NKL = int(sys.argv[5]) if len(sys.argv) > 5 else 48
VARS = {"off": (0, 0, 0), "proj": (1, 0, 0), "addnorm": (0, 1, 0), "gdn": (0, 0, 1), "all": (1, 1, 1)}
if os.environ.get("AB96_VARIANTS"):
    VARS = {k: VARS[k] for k in os.environ["AB96_VARIANTS"].split(",")}

be, _ = load_backend(PACK, mtp=False, log=lambda *a: None)
tm = be.lm.model
attn_i = [i for i, l in enumerate(tm.layers) if not l.is_linear]


def setv(v):
    DF.PROJ, DF.ADDNORM, DF.GDN = (bool(b) for b in VARS[v])


plans = {}


def plan(v):
    setv(v)
    if v not in plans:
        plans[v] = SR._Plan(tm)
    tm._kr_plan = plans[v]


def logits(cache, tok, stock=False):
    t = mx.array([[tok]], dtype=mx.int32)
    h = tm._kr_orig(t, cache) if stock else tm(t, cache)
    return be._head(h[:, -1:]).astype(mx.float32)[0, -1]


ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json")))[: 512 + NKL + 1]
kl = {}
if NKL > 0:
    c0 = be.make_cache(); be.prefill_chunk(c0, ids[:512])
    ref = []
    for t in range(NKL):
        lg = logits(c0, ids[512 + t], stock=True); mx.eval(lg); ref.append(lg)
    del c0
    for v in VARS:
        plan(v)
        c1 = be.make_cache(); be.prefill_chunk(c1, ids[:512])
        ks, top, eq = [], 0, 0
        for t in range(NKL):
            lg = logits(c1, ids[512 + t]); r = ref[t]
            lp, lq = r - mx.logsumexp(r), lg - mx.logsumexp(lg)
            k = mx.sum(mx.exp(lp) * (lp - lq)); mx.eval(k)
            ks.append(max(float(k), 0.0)); top += int(mx.argmax(r).item() == mx.argmax(lg).item()); eq += int(mx.array_equal(r, lg).item())
        kl[v] = dict(kl_mean=float(statistics.mean(ks)), kl_max=float(max(ks)), top1=top / NKL, bit_equal=eq)
        print("KL", v, json.dumps(kl[v]), flush=True)
        del c1
    del ref
    mx.clear_cache()

# ---- speed at CTX
cache = be.make_cache()
be.prefill_chunk(cache, ids[:256])
ca = [cache[i] for i in attn_i]
_, HK, _, DK = ca[0].keys.shape; DV = ca[0].values.shape[3]
A = -(-(CTX + 64) // 256) * 256
for c in ca:
    dt = c.keys.dtype
    c.keys = mx.random.normal((1, HK, A, DK)).astype(dt); c.values = mx.random.normal((1, HK, A, DV)).astype(dt); c.offset = CTX
mx.eval(*[a for c in ca for a in (c.keys, c.values)])
TOK = mx.array([[1000]], dtype=mx.int32)


def kv():
    return [a for c in ca for a in (c.keys, c.values)]


def step():
    return mx.argmax(be._head(tm(TOK, cache)[:, -1:]).astype(mx.float32)[0, -1], -1)


def step_sync():
    y = step(); mx.eval(y, *kv())
    for c in ca:
        c.trim(1)


cen = {}
for v in VARS:
    plan(v); step_sync(); step_sync()
    cen[v] = graph_census(step())["dispatches"]
    for c in ca:
        c.trim(1)


def block(v):
    plan(v)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step_sync()
    return (time.perf_counter() - t0) / STEPS * 1e3


def pipe(v, n=16):
    plan(v)
    y = step(); mx.async_eval(y)
    t0 = time.perf_counter()
    for _ in range(n):
        yn = mx.argmax(be._head(tm(y.reshape(1, 1).astype(mx.int32), cache)[:, -1:]).astype(mx.float32)[0, -1], -1); mx.async_eval(yn); y.item(); y = yn
    y.item(); dt = (time.perf_counter() - t0) * 1e3 / n
    for c in ca:
        c.trim(n + 1)
    return dt


names = list(VARS)
res = {v: [] for v in names}; pp = {v: [] for v in names}
for r in range(BLOCKS):
    for v in (names if r % 2 == 0 else names[::-1]):
        res[v].append(block(v)); pp[v].append(pipe(v))
out = dict(pack=os.path.basename(PACK.rstrip("/")), ctx=CTX, kl=kl, dispatches=cen,
           step=({v: (round(min(x), 3), round(statistics.median(x), 3)) for v, x in res.items()}),
           pipe=({v: (round(min(x), 3), round(statistics.median(x), 3)) for v, x in pp.items()}))
print("AB96", json.dumps(out), flush=True)
print(f"\n{out['pack']} ctx {CTX}\n| variant | dispatches | step ms best/med | pipelined ms best/med | KL mean | KL max | top1 | bit-equal |\n|---|---|---|---|---|---|---|---|")
for v in names:
    k = kl.get(v, {})
    print(f"| {v} | {cen[v]} | {out['step'][v][0]:.2f} / {out['step'][v][1]:.2f} | {out['pipe'][v][0]:.2f} / {out['pipe'][v][1]:.2f} | "
          f"{k.get('kl_mean', float('nan')):.2e} | {k.get('kl_max', float('nan')):.2e} | {k.get('top1', float('nan')):.3f} | {k.get('bit_equal', '-')}/{NKL} |")

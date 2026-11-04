"""rest_table.py KIND(q35|ds) PACK ROWS [--layers N]: decode "rest" table = every non-GEMV, non-attention-core launch of one step.

Graph census (dispatches by primitive, per layer) plus, for the fast ops and the custom Metal kernels that the step calls (recorded while the
uncompiled step builds its graph), count / us each (isolated replay, N independent copies in one eval) / bytes touched.
q35: Qwen3.5 mlx pack, one sequence, ROWS tokens in one forward (stock mlx-lm path; ROWS == 1 also shows the compiled decfuse regions census).
ds : DS4.1 slice pack, ROWS = users in one batched step (rows.step)."""
import argparse, collections, os, re, sys, tempfile, time

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("../tests", "..", ".")]
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000")
import numpy as np
import mlx.core as mx
from count_launches import graph_census

ap = argparse.ArgumentParser()
ap.add_argument("kind"); ap.add_argument("pack"); ap.add_argument("rows", type=int)
ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()

REC = []            # (name, fn, args, kwargs) of fast ops / custom kernels called while recording
RECORDING = [False]


def _wrap_fast(name):
    orig = getattr(mx.fast, name)

    def f(*args, **kw):
        out = orig(*args, **kw)
        if RECORDING[0]:
            REC.append((f"fast.{name}", orig, args, kw, out))
        return out
    setattr(mx.fast, name, f)


for n in ("rms_norm", "rope", "layer_norm"):
    _wrap_fast(n)
_orig_mk = mx.fast.metal_kernel


def _mk(*args, **kw):
    k = _orig_mk(*args, **kw)
    nm = kw.get("name", args[0] if args else "?")

    def call(*cargs, **ckw):
        out = k(*cargs, **ckw)
        if RECORDING[0]:
            REC.append((f"kernel.{nm}", k, cargs, ckw, out))
        return out
    return call


mx.fast.metal_kernel = _mk

from kyojin_mlx import patch as P  # noqa: E402  (after the wrappers: modules bind mx.fast.* at call time)


def build_q35():
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(a.pack, mtp=False, log=lambda *x: None)
    tm = be.lm.model
    if a.layers:
        tm.layers = tm.layers[:a.layers]
    cache = be.make_cache()
    be.prefill_chunk(cache, list(range(1000, 1128)))
    tok = mx.array([list(range(2000, 2000 + a.rows))])
    n = len(tm.layers)

    def step(stock):
        t = getattr(tm, "_kr_orig", None) if stock else tm
        h = t(tok, cache) if t is not None else tm(tok, cache)
        return be._head(h[:, -1:])

    def trim():
        for c in cache:
            if hasattr(c, "trim") and getattr(c, "offset", None) is not None:
                c.trim(a.rows)
    return step, trim, n


def build_ds():
    from kyojin_mlx import rows as R, spec as S
    model, spec, cfg = P.build_model(a.pack)
    P.patch_model(model, a.pack, spec, cfg)
    R.enable(model, spec)
    S.install(model, spec)
    if a.layers:
        model.language_model.layers = model.language_model.layers[:a.layers]
    tc = cfg["text_config"]
    rng = np.random.default_rng(1)
    cache, lg, _, _ = S.prefill(model, rng.integers(16, tc["vocab_size"] - 16, 135).tolist())
    tok = mx.array([int(mx.argmax(lg[0]).item())] * a.rows)
    n = len(model.language_model.layers)
    from kyojin_mlx import fuse_layer as FL

    def step(stock):
        FL.ENABLED = not stock                      # recorded pass: eager layers, so the recorded kernel inputs are concrete arrays
        c = R.fork(cache) if a.rows == 1 else R.batch([R.fork(cache) for _ in range(a.rows)])
        return R.step(model, c, tok)
    return step, (lambda: None), n


step, trim, nl = (build_q35 if a.kind == "q35" else build_ds)()
for _ in range(2):
    mx.eval(step(False)); trim()
out = step(False)
cen = graph_census(out, a.rows)
mx.eval(out); trim()
ts = []
for _ in range(8):
    o = step(False); t0 = time.perf_counter(); mx.eval(o); ts.append((time.perf_counter() - t0) * 1e3); trim()
print(f"## {a.kind} {os.path.basename(a.pack.rstrip('/'))} rows {a.rows} layers {nl}: step {min(ts):.2f} ms (med {np.median(ts):.2f}), "
      f"{cen['dispatches']} dispatches ({cen['dispatches'] / nl:.1f}/layer)")
print("census: " + ", ".join(f"{k} {v}" for k, v in list(cen["by_type"].items())[:24]))
if a.kind == "q35" and a.rows == 1:
    # uncompiled stock forward for the recorded ops
    pass
RECORDING[0] = True
o = step(True)
RECORDING[0] = False
census2 = graph_census(o, a.rows)
mx.eval(o); trim()
print(f"uncompiled/recorded graph: {census2['dispatches']} dispatches; census: " + ", ".join(f"{k} {v}" for k, v in list(census2["by_type"].items())[:24]))


def nbytes(x):
    if isinstance(x, (list, tuple)):
        return sum(nbytes(i) for i in x)
    return x.nbytes if isinstance(x, mx.array) else 0


agg = collections.OrderedDict()
for nm, fn, args, kw, outp in REC:
    key = (nm, tuple(tuple(x.shape) + (str(x.dtype),) for x in args if isinstance(x, mx.array)))
    e = agg.setdefault(key, dict(n=0, fn=fn, args=args, kw=kw, out=outp))
    e["n"] += 1
print("| op | calls | calls/layer | in-> shapes | MB moved | us each (isolated) |")
print("|---|---|---|---|---|---|")
tot_us = 0.0
for (nm, shp), e in agg.items():
    try:
        def run(m):
            outs = [e["fn"](*e["args"], **e["kw"]) for _ in range(m)]
            t0 = time.perf_counter(); mx.eval(outs); return time.perf_counter() - t0
        run(8)
        ta = min(run(a.reps) for _ in range(5)); tb = min(run(a.reps // 2) for _ in range(5))
        us = (ta - tb) / (a.reps - a.reps // 2) * 1e6
    except Exception as ex:               # noqa: BLE001
        us = float("nan")
    mb = (sum(nbytes(x) for x in e["args"]) + nbytes(e["out"])) / 1e6
    tot_us += us * e["n"] if us == us else 0
    print(f"| {nm} | {e['n']} | {e['n'] / nl:.2f} | {' '.join(str(s[:-1]) for s in shp)[:60]} | {mb:.3f} | {us:.1f} |")
print(f"recorded ops total: {tot_us / 1e3:.3f} ms of the {min(ts):.2f} ms step")

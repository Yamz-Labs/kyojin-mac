"""Per-op GPU time of the served DS4.1 path (real pack, load_vlm + CED + MTP): prefill chunk, verify round (1 + D rows) and plain step.
Prefill: every wrapped call (metal kernels, mx leaf ops, kyojin / mlx_vlm functions, module __call__) syncs on enter and exit; the table is the
exclusive ms of each name (its own time minus its wrapped children), so un-wrapped glue (elementwise, slicing) lands in the caller's row.
Decode: a sync per call would cost as much as the round, so one round is captured (leaf calls + their arguments) and each leaf is replayed alone,
REPS times back to back; table = GPU ms per round of each leaf, against the bare round wall. Compile is disabled (mx.disable_compile) only for
the wrapped phases: the bare walls are measured first, compiled as served.
usage: op_profile.py [PROMPT=1024] [D=3] [PF=8192]"""
import sys, os, time, collections, statistics, types, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx

P = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
D = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PF = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
assert PF < 16384
REPS = int(os.environ.get("REPS", "20"))
ROUNDS = 24

# ---- wrap the kernel factory before any kyojin import: every kernel object, wherever it is cached, is a _Kernel
MODE = ["off"]                                       # off | sync | host | capture | record | skip
STACK, EXCL, CALLS, CAP, LDEPTH = [], collections.defaultdict(float), collections.Counter(), [], [0]
OUTS, SKIP, SIDX, KEEP = collections.defaultdict(list), set(), collections.Counter(), []
OUTK = collections.defaultdict(list)                  # (name, input shapes) -> recorded outputs in call order: a leaf called at several widths replays the right one


def _arrs(x):
    out = []
    def walk(v):
        if isinstance(v, mx.array):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for e in v:
                walk(e)
        elif isinstance(v, dict):
            for e in v.values():
                walk(e)
    walk(x)
    return out


def _timed(name, fn, a, k, leaf):
    if MODE[0] == "capture":
        if not leaf:
            return fn(*a, **k)
        if not LDEPTH[0]:
            CAP.append((name, fn, a, k))
        LDEPTH[0] += 1
        try:
            return fn(*a, **k)
        finally:
            LDEPTH[0] -= 1
    if MODE[0] == "host":
        STACK.append([name, 0.0])
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            me = STACK.pop()
            EXCL[name] += dt - me[1]; CALLS[name] += 1
            if STACK:
                STACK[-1][1] += dt
    if MODE[0] == "record" and leaf:
        r = fn(*a, **k)
        OUTS[name].append(r)
        OUTK[(name, tuple(tuple(x.shape) for x in _arrs(a) + _arrs(k)))].append(r)
        return r
    if MODE[0] == "skip" and leaf and name in SKIP:
        ins = _arrs(a) + _arrs(k)
        key = (name, tuple(tuple(x.shape) for x in ins))
        if not OUTK[key]:                                                                 # a shape the recorded round never saw: run it
            return fn(*a, **k)
        i = SIDX[key]; SIDX[key] += 1
        r = OUTK[key][i % len(OUTK[key])]
        osh = k.get("output_shapes")
        if osh is not None and [tuple(x) for x in osh] != [tuple(o.shape) for o in r]:   # shape tracks the position: zeros of the asked shape
            r = [mx.zeros(tuple(sh), dtype=dt) for sh, dt in zip(osh, k["output_dtypes"])]
        if isinstance(r, mx.array):
            return mx.depends([r], ins)[0] if ins else r
        return mx.depends(list(r), ins) if ins else r                                     # the outputs wait for the inputs: the served chain stays serial
    if MODE[0] != "sync" or not leaf:
        return fn(*a, **k)
    mx.eval(_arrs(a) + _arrs(k))
    mx.synchronize()
    STACK.append([name, 0.0])
    t = time.perf_counter()
    try:
        r = fn(*a, **k)
        mx.eval(_arrs(r))
        mx.synchronize()
    finally:
        dt = time.perf_counter() - t
        me = STACK.pop()
    EXCL[name] += dt - me[1]
    CALLS[name] += 1
    if STACK:
        STACK[-1][1] += dt
    return r


class _Kernel:
    def __init__(self, k, name):
        self.k, self.name = k, f"kernel:{name}"

    def __call__(self, *a, **kw):
        return _timed(self.name, self.k, a, kw, True)


_mk = mx.fast.metal_kernel
def _metal_kernel(*a, **kw):
    return _Kernel(_mk(*a, **kw), kw.get("name", a[0] if a else "?"))
mx.fast.metal_kernel = _metal_kernel

LEAF = ["quantized_matmul", "gather_qmm", "gather_mm", "matmul", "addmm", "argpartition", "argsort", "topk", "softmax", "take_along_axis",
        "put_along_axis", "scatter_add", "conv1d", "einsum", "cumsum", "sort", "logsumexp"]
for n in LEAF:
    if hasattr(mx, n):
        f = getattr(mx, n)
        setattr(mx, n, (lambda f, n: lambda *a, **k: _timed(f"mx.{n}", f, a, k, True))(f, n))
for n in ["scaled_dot_product_attention", "rms_norm", "layer_norm", "rope"]:
    if hasattr(mx.fast, n):
        f = getattr(mx.fast, n)
        setattr(mx.fast, n, (lambda f, n: lambda *a, **k: _timed(f"fast.{n}", f, a, k, True))(f, n))

from kyojin_mlx.serve import backends as B   # noqa: E402

B.wire_memory()
be, cfg = B.load_vlm(os.path.expanduser("~/ds41/model"), mtp=True, ced=True)
from kyojin_mlx.serve.server import cache_limit_mb_for  # noqa: E402
mx.set_cache_limit(cache_limit_mb_for(mx.get_active_memory()) * 2 ** 20)   # as the server: an unbounded buffer cache evicts the Engram page cache (232 GiB peaks)
lm = be.lm
ids = np.random.default_rng(0).integers(16, 120000, P + PF + 8 * (D + 1) * (ROUNDS + 40)).tolist()
if os.environ.get("OPTEXT"):                           # OPTEXT=glob[,glob]: real text prompt (random ids route MoE/Engram unlike text)
    import glob
    from tokenizers import Tokenizer
    txt = "".join(open(f, errors="replace").read() for g in os.environ["OPTEXT"].split(",") for f in sorted(glob.glob(g)))
    t = Tokenizer.from_file(os.path.expanduser("~/ds41/model/tokenizer.json")).encode(txt).ids
    assert len(t) >= len(ids), (len(t), len(ids))
    ids = t[:len(ids)]


def wrap_functions():
    """Every module-level function of kyojin_mlx.* and mlx_vlm.models.deepseek_v41.*, replaced in every namespace that holds it."""
    funcs = {}
    for mn, m in list(sys.modules.items()):
        if m is None or not (mn.startswith("kyojin_mlx") or mn.startswith("mlx_vlm.models.deepseek_v41")) or mn.endswith("serve.server"):
            continue
        for an, v in list(vars(m).items()):
            if isinstance(v, types.FunctionType) and v.__module__ == mn and not an.startswith("__"):
                funcs[id(v)] = (v, f"{mn.split('.')[-1]}.{an}")
    W = {}
    for _, (v, name) in funcs.items():
        @functools.wraps(v)
        def w(*a, _v=v, _n=name, **k):
            return _timed(_n, _v, a, k, False)
        W[id(v)] = w
    n = 0
    for mn, m in list(sys.modules.items()):
        if m is None or not hasattr(m, "__dict__"):
            continue
        try:
            items = list(vars(m).items())
        except TypeError:
            continue
        for an, v in items:
            if id(v) in W and isinstance(v, types.FunctionType):
                setattr(m, an, W[id(v)]); n += 1
    classes = {type(mod) for _, mod in lm.named_modules()} | {type(mod) for _, mod in be.model.named_modules()}
    for cls in classes:
        if "__call__" in vars(cls):
            o = cls.__call__
            cls.__call__ = (lambda o, nm: lambda self, *a, **k: _timed(nm, lambda *aa, **kk: o(self, *aa, **kk), a, k, False))(o, f"mod:{cls.__name__}")
    return len(funcs), n, len(classes)


cache, d = be.make_cache(), be.new_draft()
for c0 in range(0, P, PF):                             # the prompt in PF-row chunks, as the server (one P-row chunk swaps at 32K+)
    be.prefill_chunk(cache, ids[c0:min(P, c0 + PF)], draft=d); mx.synchronize()
bc = be.batch([cache])
pos = [P]


def plain():
    lg = be.step(bc, mx.array([ids[pos[0]]], dtype=mx.int32))
    mx.eval(lg, *[a for c in bc if hasattr(c, "stacked_arrays") for a in c.stacked_arrays()], *KEEP)
    KEEP.clear()
    mx.synchronize()
    pos[0] += 1


def verify():
    ps = be.spec_verify_bc(bc, [ids[pos[0]:pos[0] + D + 1]])
    mx.eval(ps.logits, *KEEP)
    KEEP.clear()
    ps.commit([min(2, D)])
    be.settle(bc)
    mx.synchronize()
    pos[0] += min(2, D) + 1


def bare(fn, n):
    for _ in range(4):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


PFI = ids[P:P + PF]
assert len(PFI) == PF


def prefill():
    c, dd = be.make_cache(), be.new_draft()
    t = time.perf_counter(); be.prefill_chunk(c, PFI, draft=dd); mx.synchronize()
    return (time.perf_counter() - t) * 1e3


WALL = {"plain": bare(plain, ROUNDS), f"verify{D + 1}": bare(verify, ROUNDS)}
pw = [round(prefill()) for _ in range(3)]
WALL["prefill"] = min(pw[1:])
print(f"OPPROF bare (compile as served) plain {WALL['plain']:.2f} ms  verify{D + 1} {WALL[f'verify{D + 1}']:.2f} ms  prefill {PF} rows {pw} ms "
      f"= {PF * 1e3 / WALL['prefill']:.0f} tok/s", flush=True)

nf, nr, nc = wrap_functions()
print(f"OPPROF wrapped {nf} functions ({nr} bindings), {nc} module classes, kernels via metal_kernel factory", flush=True)


FULL = open(os.environ.get("OPFULL", "/tmp/opprof_full.out"), "w")


def table(title, tt, cn, wall, per=1, top=30):
    tot = sum(tt.values())
    head = f"OPPROF {title}: wall {wall:.2f} ms, wrapped sum {tot * 1e3 / per:.2f} ms ({tot * 1e3 / per / wall * 100:.0f} % of wall), {sum(cn.values()) / per:.0f} calls"
    print(head, flush=True); print(head, file=FULL)
    for i, (k, v) in enumerate(sorted(tt.items(), key=lambda kv: -kv[1])):
        line = f"  {k:56s} {v * 1e3 / per:9.3f} ms  {v * 1e3 / per / wall * 100:5.1f} %  calls {cn[k] / per:.0f}"
        print(line, file=FULL)
        if i < top:
            print(line, flush=True)
    FULL.flush()


if not os.environ.get("OPDEC"):                       # OPDEC=1: decode ablation only
    # ---- prefill, synced per call (ms-scale ops)
    mx.disable_compile()                                 # a sync inside a compiled trace throws; the bare walls above ran compiled
    MODE[0] = "sync"
    prefill()
    EXCL.clear(); CALLS.clear()
    w = prefill()
    MODE[0] = "off"
    table(f"PREFILL {PF} rows GPU leaves (each synced; wall - sum = unwrapped glue + host)", EXCL, CALLS, WALL["prefill"])
    print(f"OPPROF prefill leaf-synced wall {w:.0f} ms", flush=True)
    mx.enable_compile()                                  # host pass as served: compiled regions run without Python after their trace
    for label, fn in (("prefill", prefill), (f"verify{D + 1}", verify), ("plain", plain)):
        fn()
        EXCL.clear(); CALLS.clear()
        MODE[0] = "host"
        n = 1 if label == "prefill" else 8
        try:
            for _ in range(n):
                fn()
        finally:
            MODE[0] = "off"
        table(f"HOST {label} exclusive wall, no syncs (python + graph build + implicit waits)", EXCL, CALLS, WALL[label], per=n, top=15)

mx.disable_compile()
# ---- decode, ablation in the served chain: wall of a round where every call of one leaf name returns its recorded output (mx.depends on its inputs) instead of running
# (its inputs are still evaluated at the end of the round, so producers stay live); cost of the name = base wall - ablated wall, at served clock
for label, fn in [(l, f) for l, f in ((f"verify{D + 1}", verify), ("plain", plain)) if os.environ.get("OPONLY", l) == l]:   # OPONLY=plain: one step kind
    fn()
    OUTS.clear(); OUTK.clear(); MODE[0] = "record"
    try:
        fn()
    finally:
        MODE[0] = "off"
    mx.eval([_arrs(o) for v in OUTS.values() for o in v])
    names = [k for k in OUTS if k.startswith(("kernel:", "fast."))]
    def wall(skip, n=12):
        SKIP.clear(); SKIP.update(skip); MODE[0] = "skip"
        try:
            SIDX.clear()
            fn()
            ts = []
            for _ in range(n):
                SIDX.clear()
                t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        finally:
            MODE[0] = "off"; SKIP.clear()
        return statistics.median(ts)
    def safe(skip):
        try:
            return wall(skip)
        except Exception as e:                            # a leaf whose recorded output breaks a later host check: report, go on
            print(f"OPPROF ablation {label} {skip if len(skip) < 3 else 'all'} failed: {type(e).__name__} {str(e)[:80]}", flush=True)
            return float("nan")
    base = wall(())
    allk = safe(names)
    gain = {}
    for k in names:
        w = safe((k,))
        if w == w:
            gain[k] = max(0.0, (base - w) * 1e-3)
    names = list(gain)
    table(f"DECODE {label} ablation in the served chain (base {base:.2f} ms uncompiled, all leaves skipped {allk:.2f} ms = host + glue)", gain, {k: len(OUTS[k]) for k in names}, base, top=40)
# ---- decode: capture one round of top-level leaves, replay each alone
for label, fn in [(l, f) for l, f in ((f"verify{D + 1}", verify), ("plain", plain)) if os.environ.get("OPONLY", l) == l]:   # OPONLY=plain: one step kind
    CAP.clear()
    MODE[0] = "capture"
    try:
        fn()
    finally:
        MODE[0] = "off"
    groups = collections.OrderedDict()
    for name, f, a, k in CAP:
        sig = (name, tuple(tuple(x.shape) for x in _arrs(a) + _arrs(k))[:6])
        groups.setdefault(sig, [f, a, k, 0])[3] += 1
    gpu, cnt = collections.defaultdict(float), collections.Counter()
    for (name, _), (f, a, k, c) in groups.items():
        ts, r = [], REPS
        for _ in range(4):
            mx.synchronize()
            t = time.perf_counter()
            outs = [f(*a, **k) for _ in range(r)]
            mx.eval(_arrs(outs)); mx.synchronize()
            dt = time.perf_counter() - t
            ts.append(dt / r); del outs
            r = int(min(2000, max(r, r * 0.02 / max(dt, 1e-6))))   # grow to >= 20 ms per sample so the eval overhead amortizes
        ts = ts[1:]
        gpu[name] += statistics.median(ts) * c
        cnt[name] += c
    table(f"DECODE {label} replayed leaves (GPU per round; wall - sum = launch/host/glue)", gpu, cnt, WALL[label])
    print(f"OPPROF {label} leaf calls per round {len(CAP)} in {len(groups)} shapes", flush=True)
print("OPPROF note: replays are independent dispatches (overlap); in the served round the leaves are dependent (5.4 us/op floor, launch_floor.py)", flush=True)
print("OPPROF done", flush=True)

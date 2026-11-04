"""Per-op GPU class table of ONE prefill chunk of a served MoE/GDN pack at a given context depth (sync after every wrapped call; exclusive ms).
Wrapped: every metal kernel, leaf mx ops, mx.fast ops, module classes (their rows hold the un-wrapped glue). Class table groups names by regex.
usage: moepf2_prof.py PACK TEXT_FILE DEPTH [--chunk 2048] [--reps 2] [--list]    (profiled chunk = tokens [DEPTH, DEPTH+chunk))"""
import argparse, os, sys, time, collections, re
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "1")
import numpy as np, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("text"); ap.add_argument("depth", type=int)
ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--list", action="store_true"); a = ap.parse_args()
MODE = ["off"]; STACK = []; EXCL = collections.defaultdict(float); CALLS = collections.Counter(); SHP = {}

def _arrs(x):
    out = []
    def walk(v):
        if isinstance(v, mx.array): out.append(v)
        elif isinstance(v, (list, tuple)):
            for e in v: walk(e)
        elif isinstance(v, dict):
            for e in v.values(): walk(e)
    walk(x); return out

def _timed(name, fn, args, kw):
    if MODE[0] != "sync": return fn(*args, **kw)
    mx.eval(_arrs(args) + _arrs(kw)); mx.synchronize()
    STACK.append(0.0); t = time.perf_counter()
    try:
        r = fn(*args, **kw); mx.eval(_arrs(r)); mx.synchronize()
    finally:
        dt = time.perf_counter() - t; ch = STACK.pop()
    EXCL[name] += dt - ch; CALLS[name] += 1
    if name not in SHP: SHP[name] = [tuple(x.shape) for x in _arrs(args)][:4]
    if STACK: STACK[-1] += dt
    return r

class _Kernel:
    def __init__(self, k, name): self.k, self.name = k, "kernel:" + name
    def __call__(self, *x, **kw): return _timed(self.name, self.k, x, kw)
_mk = mx.fast.metal_kernel
mx.fast.metal_kernel = lambda *x, **kw: _Kernel(_mk(*x, **kw), kw.get("name", x[0] if x else "?"))
for n in ["quantized_matmul", "gather_qmm", "gather_mm", "matmul", "addmm", "argpartition", "argsort", "topk", "softmax", "take_along_axis", "put_along_axis", "scatter_add", "conv1d", "cumsum", "sort", "logsumexp"]:
    if hasattr(mx, n): setattr(mx, n, (lambda f, n: lambda *x, **k: _timed("mx." + n, f, x, k))(getattr(mx, n), n))
for n in ["scaled_dot_product_attention", "rms_norm", "layer_norm", "rope"]:
    if hasattr(mx.fast, n): setattr(mx.fast, n, (lambda f, n: lambda *x, **k: _timed("fast." + n, f, x, k))(getattr(mx.fast, n), n))

from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import multiuser as MU
from kyojin_mlx.foreign_names import gguf_serve_dir
pack = gguf_serve_dir(a.pack)
tok = AutoTokenizer.from_pretrained(pack)
ids = tok.encode(open(a.text).read())[:a.depth + a.chunk]
assert len(ids) == a.depth + a.chunk, len(ids)
be, _ = load_backend(pack, log=lambda *x: None); mx.clear_cache()
m = be.model; lm = m.language_model if "language_model" in m else m; trunk = MU._trunk(lm)
cn = collections.Counter(type(x).__name__ for _, x in m.named_modules())
if a.list: print(cn); sys.exit()
WRAP = {"SwitchGLU", "ForeignLinear", "RowLinear", "Conv1d", "Attention", "GatedDeltaNet", "SparseMoeBlock", "MLP", "RMSNorm", "Embedding", "StoredEmbedding"}
done = set()
for n, x in m.named_modules():
    c = type(x); nm = c.__name__
    if c in done: continue
    if any(w in nm for w in WRAP) and "Layer" not in nm and "Model" not in nm:
        done.add(c); o = c.__call__
        c.__call__ = (lambda o, nm: lambda self, *x, **k: _timed("mod:" + nm, o, (self,) + x, k))(o, nm)

def run(lo, hi, cache):
    t = mx.array(ids[lo:hi])[None]
    h = (trunk(t, cache=cache) if MU._kw_cache(type(trunk)) else trunk(t, cache))[0]
    mx.eval(h, *be.state(cache))
def pre(depth):
    cache = be.make_cache()
    for lo in range(0, depth, a.chunk): run(lo, min(lo + a.chunk, depth), cache)
    return cache
print(f"resident {mx.get_active_memory()/2**30:.1f} GiB", flush=True)
pre(a.chunk); MODE[0] = "off"
cache = pre(a.depth); run(a.depth, a.depth + a.chunk, cache)            # warm the chunk at depth (compile)
res = []
for r in range(a.reps):
    cache = pre(a.depth); mx.synchronize()
    t0 = time.perf_counter(); run(a.depth, a.depth + a.chunk, cache); mx.synchronize(); bare = time.perf_counter() - t0
    cache = pre(a.depth); EXCL.clear(); CALLS.clear(); MODE[0] = "sync"; t0 = time.perf_counter(); run(a.depth, a.depth + a.chunk, cache); wall = time.perf_counter() - t0; MODE[0] = "off"
    res.append((bare, wall, dict(EXCL), dict(CALLS)))
bare, wall, ex, ca = min(res, key=lambda r: r[0])
print(f"depth {a.depth} chunk {a.chunk}: bare {bare*1e3:.0f} ms ({a.chunk/bare:.0f} tok/s); synced {wall*1e3:.0f} ms; sum of rows {sum(ex.values())*1e3:.0f} ms")
for k, v in sorted(ex.items(), key=lambda kv: -kv[1])[:40]: print(f"{v*1e3:8.1f} ms {100*v/wall:5.1f} %  calls {ca[k]:4d}  {k}  {SHP.get(k)}")

CLS = [("routed expert GEMM (kqgmm)", r"kqgmm|SwitchGLU"), ("expert gather/silu perm", r"kq_gather_perm|kq_silu_perm"), ("router/top-k/sort/scatter", r"argsort|argpartition|softmax|take_along|cumsum|SparseMoeBlock|moe_"),
       ("shared expert + dense proj (deq8/ForeignLinear/RowLinear/MLP)", r"deq8|ForeignLinear|RowLinear|MLP"), ("attention (sdpa, rope, Attention glue)", r"scaled_dot|rope|Attention"),
       ("GDN chunk scan", r"gated_delta|gdn_nax"), ("GDN pre/conv/norm", r"GatedDeltaNet|conv1d|Conv1d|gdn_pf|RMSNormGated"), ("norms", r"rms_norm|RMSNorm"), ("embedding", r"Embedding")]
tot = collections.defaultdict(float); used = set()
for k, v in ex.items():
    for nm, rx in CLS:
        if re.search(rx, k): tot[nm] += v; break
    else: tot["other"] += v
print("CLASS TABLE (synced ms, includes ~0.1-0.3 ms sync cost per op; bare chunk %.0f ms)" % (bare * 1e3))
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]): print(f"  {v*1e3:7.1f} ms {100*v/sum(tot.values()):5.1f} %  {k}")

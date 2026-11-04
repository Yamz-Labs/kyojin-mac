"""Stage profile of a DS4.1 prefill chunk on a patched slice (ds2 = 2 layers, full dims). Each stage evals alone: sums are upper bounds.
usage: prof_prefill.py PACK T [reps]"""
import sys, time, collections, os
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P
COOL = float(os.environ.get('KYOJIN_BENCH_COOL', '0'))     # idle seconds before each timed stage (burst-state numbers)
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
model, spec, cfg = P.build_model(pack)
print("patched:", P.patch_model(model, pack, spec, cfg), flush=True)
lm = model.language_model
acc = collections.defaultdict(float)
def eva(o):
    mx.eval(o)
def wrap(obj, meth, name):
    cls = obj.__class__
    orig = getattr(cls, meth)
    def f(self, *a, **k):
        eva([v for v in list(a) + list(k.values()) if isinstance(v, mx.array)]); time.sleep(COOL); t0 = time.perf_counter(); o = orig(self, *a, **k); eva(o); acc[name] += time.perf_counter() - t0; return o
    obj.__class__ = type(cls.__name__, (cls,), {meth: f})
for i, l in enumerate(lm.layers):
    at = l.attn
    for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        wrap(getattr(at, n), "__call__", "lin." + n)
    for n in ("compressor", "indexer"):
        if getattr(at, n, None) is not None: wrap(getattr(at, n), "__call__", "attn." + n)
    wrap(at, "_sparse_attention", "attn.sparse_core"); wrap(at, "_window_part", "attn.window_part")
    wrap(l.ffn, "__call__", "ffn.total")
    if hasattr(l.ffn, "shared_experts"): wrap(l.ffn.shared_experts, "__call__", "ffn.shared")
    wrap(at, "__call__", "attn.total")
    wrap(l, "__call__", "layer.total")
import importlib
LM = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
def timed(fn, name):
    def f(*a, **k):
        eva([v for v in list(a) + list(k.values()) if isinstance(v, mx.array)]); time.sleep(COOL); t0 = time.perf_counter(); o = fn(*a, **k); eva(o); acc[name] += time.perf_counter() - t0; return o
    return f
B = LM.DeepseekV41Block
B.hc_pre = staticmethod(timed(B.__dict__["hc_pre"].__func__ if isinstance(B.__dict__["hc_pre"], staticmethod) else B.hc_pre, "mix.hc_pre"))
LM.hc_expand = timed(LM.hc_expand, "mix.hc_expand")
for l in lm.layers:
    wrap(l, "hc_mixes", "mix.hc_mixes"); wrap(l.attn_norm, "__call__", "norm"); wrap(l.ffn_norm, "__call__", "norm")
    if getattr(l, "engram", None) is not None: wrap(l.engram, "__call__", "engram")
lm._ensure_engram_hash()
if getattr(lm, "engram_hash", None) is not None: wrap(lm.engram_hash, "__call__", "engram.hash")
wrap(lm.head, "__call__", "head")
EM = importlib.import_module("mlx_vlm.models.deepseek_v41.engram")
EM._engram_gate_score = timed(EM._engram_gate_score, "engram.gate_score"); EM._engram_apply_gate = timed(EM._engram_apply_gate, "engram.apply_gate")
for l in lm.layers:
    if getattr(l, "engram", None) is not None:
        wrap(l.engram.embed, "__call__", "engram.embed"); wrap(l.engram.wkv, "__call__", "engram.wkv")
ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
def run():
    cache = lm.make_cache(); acc.clear(); t0 = time.perf_counter(); lg = lm(ids, cache=cache).logits; mx.eval(lg); return time.perf_counter() - t0, dict(acc)
run()
runs = [run() for _ in range(reps)]
stage_min = {k: min(r[1].get(k, 1e9) for r in runs) for k in runs[0][1]}      # min over runs per stage (the GPU alternates between a fast and a slow state)
print(f"T={T} whole forward (stage evals, min of {reps}) {min(r[0] for r in runs)*1e3:.1f} ms")
for k, v in sorted(stage_min.items(), key=lambda kv: -kv[1]): print(f"  {k:20s} {v*1e3:8.1f} ms")
print("peak GiB", round(mx.get_peak_memory() / 2**30, 2))
# in-process check: one dense linear chained, same process / memory state as the forward above
import numpy as np
m = lm.layers[0].attn.wq_b
for T2 in (T,):
    x = mx.array(np.random.default_rng(0).standard_normal((1, T2, m.in_features)).astype(np.float16)); mx.eval(x)
    best = 1e9
    for _ in range(8):
        t0 = time.perf_counter(); o = m.__class__.__mro__[1].__call__(m, x) if False else m(x); mx.eval(o); best = min(best, time.perf_counter() - t0)
    print(f"in-process wq_b T={T2}: min of 8 = {best*1e3:.1f} ms")

"""Stage profile of the REAL prompt path (ds2 slice, routed experts stubbed): every stage (kernel wrappers, linears, router, ...) evaluated alone after an idle gap, min over reps.
usage: prof_stage.py PACK T [reps] [cool_s]"""
import sys, time, os, subprocess, collections, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, pf_glue as PG, attn_prefill as AP, fuse as F
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5; COOL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
H = cfg["text_config"]["hidden_size"]
P._moe_rows = lambda moe, x2, ids2, w2, xsrc=None: mx.zeros((x2.shape[0], H), dtype=mx.float32)
lm = model.language_model; LM = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
acc = collections.defaultdict(list); depth = [0]
def ev(o): mx.eval(o)
ME = os.getpid()
def others():
    out = subprocess.run(["ps", "-axo", "pid,pcpu,command"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if ("python" in l or "pytest" in l) and str(ME) != l.split()[0] and "lockf" not in l and float(l.split()[1].replace(",", ".")) > 3.0]
def quiet(wait=60):
    t0 = time.time()
    while others():
        if time.time() - t0 > wait: return False
        time.sleep(1.0)
    return True
def mk(fn, name):
    def f(*a, **k):
        if depth[0]: return fn(*a, **k)
        ev([v for v in list(a) + list(k.values()) if isinstance(v, mx.array)])
        for _try in range(3 if (not name.startswith("attn.") or name == "attn.sparse_core") else 1):
            q0 = quiet(); time.sleep(COOL)
            depth[0] += 1; t0 = time.perf_counter(); o = fn(*a, **k); ev(o); dt = time.perf_counter() - t0; depth[0] -= 1
            if q0 and not others(): break
        acc[name].append(dt); return o
    return f
def wrap_mod(obj, meth, name):
    cls = obj.__class__; orig = getattr(cls, meth)
    obj.__class__ = type(cls.__name__, (cls,), {meth: (lambda self, *a, _f=mk(lambda self, *a, **k: orig(self, *a, **k), name), **k: _f(self, *a, **k))})
for n in ("q_rope", "o_rope_inv", "fq_t", "expand", "combine", "pre_norm_fq", "wo_a_prompt", "wq_b_rope"): setattr(PG, n, mk(getattr(PG, n), "pf." + n))
orig_sa = AP.sparse_attention; AP.sparse_attention = mk(orig_sa, "attn.sparse_core")
for i, l in enumerate(lm.layers):
    a = l.attn
    for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"): wrap_mod(getattr(a, n), "__call__", "lin." + n)
    wrap_mod(a.q_norm, "__call__", "norm.q"); wrap_mod(a.kv_norm, "__call__", "norm.kv")
    for n in ("compressor", "indexer"):
        if getattr(a, n, None) is not None: wrap_mod(getattr(a, n), "__call__", "attn." + n)
    wrap_mod(a, "_window_part", "attn.window_part")
    wrap_mod(l.ffn.gate, "__call__", "ffn.router")
    sh = l.ffn.shared_experts
    for n in ("gate_proj", "up_proj", "down_proj"): wrap_mod(getattr(sh, n), "__call__", "shared." + n)
    wrap_mod(l, "hc_mixes", "mix.hc_mixes")
fqo = LM.fake_quant_fp8_ue8m0
LM.fake_quant_fp8_ue8m0 = mk(fqo, "fq (standalone)"); LM.fake_quant_fp8_ue8m0._s5_memo = True
ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
def run():
    cache = lm.make_cache(); acc.clear(); t0 = time.perf_counter(); lg = lm(ids, cache=cache).logits; mx.eval(lg); return time.perf_counter() - t0, {k: list(v) for k, v in acc.items()}
run(); runs = [run() for _ in range(reps)]
sm = {k: sum(min(r[1][k][i] for r in runs) for i in range(len(runs[0][1][k]))) for k in runs[0][1]}; cnt = {k: len(v) for k, v in runs[0][1].items()}
print(f"T={T} whole (with stage evals) min {min(r[0] for r in runs):.2f} s; stage sum (2 layers) {sum(sm.values())*1e3:.0f} ms")
for k, v in sorted(sm.items(), key=lambda kv: -kv[1]): print(f"  {k:22s} {v*1e3:8.1f} ms  ({cnt[k]} calls, {v*1e3/cnt[k]:6.2f} per call)")

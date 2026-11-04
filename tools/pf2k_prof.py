"""Per-module-class inclusive GPU time of one prefill chunk (eval + synchronize around every call: no overlap, upper bound). usage: pf2k_prof.py MODEL_DIR rows"""
import sys, time, collections, mlx.core as mx, mlx.nn as nn
from kyojin_mlx import exl3_linear as EL, mlp_pf as MP
from kyojin_mlx.serve.backends import load_backend
from mlx_lm.models import qwen3_5 as Q5, qwen3_next as QN
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
acc = collections.defaultdict(float); cnt = collections.Counter(); on = [False]
depth = [0]
def wrap(cls, name, meth="__call__"):
    f = getattr(cls, meth)
    def w(self, *a, **k):
        if not on[0]:
            return f(self, *a, **k)
        mx.synchronize(); t0 = time.perf_counter(); depth[0] += 1
        out = f(self, *a, **k)
        mx.eval(out); mx.synchronize(); depth[0] -= 1
        acc[name] += time.perf_counter() - t0; cnt[name] += 1
        return out
    setattr(cls, meth, w)
wrap(Q5.GatedDeltaNet, "GDN(incl lin)"); wrap(Q5.Attention, "Attn(incl lin)"); wrap(nn.RMSNorm, "RMSNorm"); wrap(EL.Exl3Linear, "Exl3Linear(all)")
wrap(QN.Qwen3NextMLP, "MLP(incl lin)")
def go():
    c = be.make_cache(); mx.synchronize(); on[0] = False; t0 = time.perf_counter(); be.prefill_chunk(c, ids); return time.perf_counter() - t0
go(); go()
on[0] = True; c = be.make_cache(); t0 = time.perf_counter(); be.prefill_chunk(c, ids); tot = time.perf_counter() - t0
acc.clear(); cnt.clear()
c = be.make_cache(); t0 = time.perf_counter(); be.prefill_chunk(c, ids); tot = time.perf_counter() - t0
for k, v in sorted(acc.items()): print(f"{k:18s} {v*1e3:8.1f} ms  n={cnt[k]}")
print(f"total (synced) {tot*1e3:.1f} ms")

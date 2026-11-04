"""Per layer kind prefill time WITHOUT the routed experts (stubbed: free), so that real MoE numbers (bench_s6_ab, 384 experts) can be added.
Kinds: ds2 L0 = compressing layer with indexer (kv + index source, ratio 2), ds2 L1 = reuse layer (shared KV and top-k), ds2w L1 = window-only (ratio 0).
Layer times come from an eval between layers (2 per forward). Arms interleaved per round, order rotated, min over rounds.
 ON = prefill kernels (attention, window-only, mHC rows, mix), GLUE-OFF = attention kernels on, mHC + mix stock, OFF = everything stock (T <= off_max).
usage: bench_kinds.py PACK T,T,.. [rounds] [off_max_T]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, patch as P, attn_prefill as AP, mhc_prefill as MPF, fuse as F
pack, Ts = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 8; off_max = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
import os
COOL = float(os.environ.get("KYOJIN_BENCH_COOL", "3"))       # seconds of idle GPU before each timed layer (Low Power Mode: sustained load runs at ~45 % of the burst speed)
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
H = cfg["text_config"]["hidden_size"]
P._moe_rows = lambda moe, x2, ids2, w2, xsrc=None: mx.zeros((x2.shape[0], H), dtype=mx.float32)      # routed experts free
lm = model.language_model
times = {}
class Tm:
    pass
for i, l in enumerate(lm.layers):
    cls = l.__class__; orig = cls.__call__
    def f(self, h, pre_mix, image_mask, cache, _i=i, _o=orig):
        mx.eval(h, pre_mix); time.sleep(COOL); t0 = time.perf_counter(); o = _o(self, h, pre_mix, image_mask, cache); mx.eval(o); times[_i] = time.perf_counter() - t0; return o
    l.__class__ = type(cls.__name__, (cls,), {"__call__": f})
ARMS = {"ON": (True, True, True, 1024), "DENSE128": (True, True, True, 128), "GLUE-OFF": (True, False, False, 1024), "OFF": (False, False, False, 128)}
def arm(name):
    a, m, x, dmax = ARMS[name]
    EL.DENSE_FUSED_MAX = dmax
    AP.ENABLED = "1" if a else "0"; F.MHC = m; MPF.ENABLED = "1" if x else "0"
def run(T, ids):
    cache = lm.make_cache(); times.clear(); mx.eval(lm(ids, cache=cache).logits); return dict(times)
for T in Ts:
    ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
    arms = (["ON", "GLUE-OFF"] if not os.environ.get("KYOJIN_KINDS_D128") else ["ON", "DENSE128", "GLUE-OFF"]) + (["OFF"] if T <= off_max else [])
    for a in arms: arm(a); run(T, ids)
    res = {a: [] for a in arms}
    for r in range(rounds):
        for a in (arms[r % len(arms):] + arms[:r % len(arms)]):
            arm(a); res[a].append(run(T, ids))
    for a in arms:
        mins = {i: min(x[i] for x in res[a]) for i in res[a][0]}
        print(f"T={T:5d} {a:9s} " + "  ".join(f"L{i}: {v*1e3:7.1f} ms ({T/v:6.0f} tok/s)" for i, v in mins.items()) + f" | peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)

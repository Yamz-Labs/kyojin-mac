import sys, time, os, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, pf_glue as PG
T = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
model, spec, cfg = P.build_model(sys.argv[1]); P.patch_model(model, sys.argv[1], spec, cfg)
a = model.language_model.layers[1].attn; rope = a.rope
o = mx.array((np.random.default_rng(5).standard_normal((1, 64, T, 512)) * 2).astype(np.float16)); mx.eval(o)
PG.FA.rope_range(rope, 0, T)
def old(): return a.wo_b(PG.fq_t(a.wo_a(PG.o_rope_inv(o, rope, 0, 8))))
def new(): return a.wo_b(PG.wo_a_prompt(a.wo_a, o, rope, 0))
def old_wo_a_only(): return PG.fq_t(a.wo_a(PG.o_rope_inv(o, rope, 0, 8)))
def new_wo_a_only(): return PG.wo_a_prompt(a.wo_a, o, rope, 0)
assert bool(mx.array_equal(old(), new()).item())
for nm, fs in (("o_rope+wo_a+fq_t -> wo_b input", (old_wo_a_only, new_wo_a_only)),):
    ta, tb = [], []
    for fn in fs: mx.eval(fn())
    for i in range(11):
        for f, l in ((fs[0], ta), (fs[1], tb)) if i % 2 == 0 else ((fs[1], tb), (fs[0], ta)):
            time.sleep(1.0); t0 = time.perf_counter(); mx.eval(f()); l.append((time.perf_counter() - t0) * 1e3)
    print(f"{nm}: old min {min(ta):.1f} p25 {np.percentile(ta,25):.1f} | new min {min(tb):.1f} p25 {np.percentile(tb,25):.1f}")

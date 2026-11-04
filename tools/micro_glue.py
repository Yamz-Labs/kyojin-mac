"""Isolated A/B of op chains: old (stock) vs new (pf_glue); interleaved, min / p25 over N, each after 0.5 s idle. usage: micro_glue.py [T] [N]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import pf_glue as PG
from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
H, D, RD, G = 64, 512, 64, 8
rope = DeepseekV4RoPE(RD, 160000, {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536}, 1048576)
rng = np.random.default_rng(0)
qb = mx.array(rng.standard_normal((1, T, H * D)).astype(np.float16)); o = mx.array(rng.standard_normal((1, H, T, D)).astype(np.float16)); oa = mx.array(rng.standard_normal((1, G, T, 1024)).astype(np.float16))
mx.eval(qb, o, oa); PG.FA.rope_range(rope, 0, T)
from kyojin_mlx import fuse as F
from mlx_vlm.models.deepseek_v41 import language as LM
x = mx.array(rng.standard_normal((1, T, 5120)).astype(np.float16)); res = mx.array(rng.standard_normal((1, T, 4, 5120)).astype(np.float16))
post = mx.array(rng.uniform(0, 2, (1, T, 4)).astype(np.float32)); comb = mx.array(rng.uniform(0, 1, (1, T, 4, 4)).astype(np.float32)); y = mx.array(rng.standard_normal((1, T, 1, 5120)).astype(np.float32))
mx.eval(x, res, post, comb, y)
def old_expand(): return F.mhc_expand(x.reshape(T, 1, 5120), res.reshape(T, 1, 4, 5120), post.reshape(T, 1, 4), comb.reshape(T, 1, 4, 4))[0]
import mlx.nn as nn
h4 = mx.array(rng.standard_normal((1, T, 4, 5120)).astype(np.float16)); pre = mx.array(rng.uniform(0, 1, (1, T, 4)).astype(np.float32)); nrm = nn.RMSNorm(5120, eps=1e-6); nrm.weight = mx.array(rng.standard_normal(5120).astype(np.float16)); mx.eval(h4, pre, nrm.weight)
def old_pn():
    xn = nrm(F.mhc_pre(h4.reshape(T, 1, 4, 5120), pre.reshape(T, 1, 4)).reshape(1, T, 5120)); return [xn, fq(xn)]
from kyojin_mlx import mhc_prefill as MPF
fn24 = mx.array((rng.standard_normal((24, 4 * 5120)) * 0.02).astype(np.float32)); mx.eval(fn24)
def pn_mix_old():
    a = PG.pre_norm_fq(h4, pre, nrm.weight, 1e-6); return [a[0], a[1], MPF.mixes(h4, fn24, 1e-6)]
def pn_mix_new():
    a = PG.pre_norm_fq(h4, pre, nrm.weight, 1e-6); return [a[0], a[1], MPF.mixes_rs(h4, fn24, a[2], 1e-6)]
cases = {
 "pre_norm_fq + mix (own rs kernel vs fused rs)": (pn_mix_old, pn_mix_new),
 "hc_pre+norm+fq": (old_pn, lambda: list(PG.pre_norm_fq(h4, pre, nrm.weight, 1e-6))),
 "hc_expand (prompt)": (old_expand, lambda: PG.expand(x, res, post, comb)),
 "ffn combine": (lambda: (y.astype(mx.float32).sum(-2) + x.astype(mx.float32)).astype(x.dtype), lambda: PG.combine(y, x)),
 "q rope": (lambda: rope(qb.reshape(1, T, H, D).transpose(0, 2, 1, 3), 0), lambda: PG.q_rope(qb, rope, 0, H, D)),
 "o rope inv + group transpose": (lambda: rope(o, 0, inverse=True).reshape(1, G, -1, T, D).transpose(0, 1, 3, 2, 4).flatten(-2), lambda: PG.o_rope_inv(o, rope, 0, G)),
 "wo_a-out transpose + fq": (lambda: fq(oa.transpose(0, 2, 1, 3).flatten(-2)), lambda: PG.fq_t(oa)),
}
for k, (a, b) in cases.items():
    mx.eval(a(), b()); ta, tb = [], []
    for i in range(N):
        for f, l in ((a, ta), (b, tb)) if i % 2 == 0 else ((b, tb), (a, ta)):
            time.sleep(0.5); t0 = time.perf_counter(); mx.eval(f()); l.append((time.perf_counter() - t0) * 1e3)
    print(f"{k:32s} old min {min(ta):6.2f} p25 {np.percentile(ta,25):6.2f} | new min {min(tb):6.2f} p25 {np.percentile(tb,25):6.2f} ms")

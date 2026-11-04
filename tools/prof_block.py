"""Stage times of the non-attention parts of a DS4.1 block at prefill: python tools/prof_block.py PACK T [layer=1] [reps=3]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
import importlib
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
pack, T = sys.argv[1], int(sys.argv[2]); li = int(sys.argv[3]) if len(sys.argv) > 3 else 1; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model; blk = lm.layers[li]
rng = np.random.default_rng(0)
h = mx.array(rng.standard_normal((1, T, 4, 5120)).astype(np.float16)); pre = mx.array(rng.random((1, T, 4)).astype(np.float32)); mx.eval(h, pre)
rec = {}
def st(n, fn):
    t0 = time.perf_counter(); r = fn(); mx.eval(r); rec[n] = min(rec.get(n, 9e9), time.perf_counter() - t0); return r
for _ in range(reps + 1):
    a = st("hc_mixes_attn", lambda: blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base))
    x = st("hc_pre", lambda: blk.hc_pre(h, pre))
    xn = st("attn_norm", lambda: blk.attn_norm(x))
    ap = a
    ex = st("hc_expand", lambda: L.hc_expand(xn, h, a[1], a[2]))
    st("ffn_gate", lambda: blk.ffn.gate(xn, None))
    st("shared_expert", lambda: blk.ffn.shared_experts(xn))
    st("fq8_x", lambda: L.fake_quant_fp8_ue8m0(xn))
for k, v in rec.items(): print(f"  {k:14s} {1e3 * v:7.2f} ms")
print(f"T={T} sum {1e3 * sum(rec.values()):.1f} ms")

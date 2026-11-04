"""Ablations of the forked d-split kernel (results wrong, timing only). usage: attn_abl84.py [o] [BQ BK WM WN]"""
import sys, mlx.core as mx
import attn_ds84 as A
o = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
cfg = tuple(int(x) for x in sys.argv[2:6]) if len(sys.argv) > 5 else (128, 32, 8, 2)
L = 2048; N = o + L
q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
BAR = "threadgroup_barrier(mem_flags::mem_threadgroup);"
E = {"base": (),
     "no_barriers": ((BAR, ""),),
     "no_peer_reduce": (("for (short peer_group = 1; peer_group < WN; peer_group++) {", "for (short peer_group = 1; peer_group < 1; peer_group++) {"),),
     "no_xchg_at_all": ((BAR, ""), ("for (short peer_group = 1; peer_group < WN; peer_group++) {", "for (short peer_group = 1; peer_group < 1; peer_group++) {"), ("slot[base + ik * kEPF + i] = s[i];", ""), ("s[i] = s_xchg[row_group][0][base + ik * kEPF + i];", "")),
     "no_causal_mask": (("if (do_causal && kb >= kb_min_causal) {", "if (false) {"),),
     "no_softmax_exp": (("Stile.template row_bin_op<ExpSubOp>(new_max);", ""),),
     "no_V_mma": (("otile_t::NAXFrag_t::mma(\n            Otile.frag_at(0, id),", "if (false) otile_t::NAXFrag_t::mma(\n            Otile.frag_at(0, id),"),),
     }
res = {}
for rnd in range(2):
    for n, ed in E.items():
        t = A.bench(lambda: A.attention(q, k, v, 0.0625, *cfg, edits=ed, tag=n)); res.setdefault(n, []).append(t)
for n, ts in res.items():
    t = min(ts); print(f"{n:16s}: {t*1e3:6.2f} ms {fl/t/1e12:5.2f} TF", flush=True)

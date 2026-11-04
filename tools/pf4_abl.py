"""pf4: ablation of the forked hd256 dsplit kernel (what does each phase cost). usage: pf4_abl.py O"""
import sys, os, mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
import attn_ds84 as A
o = int(sys.argv[1]); L = 2048; N = o + L; hd = 256
q = mx.random.normal((1, 16, L, hd)).astype(mx.float16); k = mx.random.normal((1, 4, N, hd)).astype(mx.float16); v = mx.random.normal((1, 4, N, hd)).astype(mx.float16); mx.eval(q, k, v)
fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
BAR = "threadgroup_barrier(mem_flags::mem_threadgroup);"
V = {"stock": None, "fork": (),
 "noxchg": (("peer_group = 1; peer_group < WN", "peer_group = 1; peer_group < 1"), (BAR, "")),
 "noPV": (("for (short id = 0; id < TDh; id += 2) {", "for (short id = 0; id < 0; id += 2) {"),),
 "noQK": (("for (short ik = 0; ik < TK; ik += 2) {", "for (short ik = 0; ik < 0; ik += 2) {"),),
 "noexp": (("Stile.template row_bin_op<ExpSubOp>(new_max);", ""),),
 "noKVadv": (("K += BK * int(params->K_strides[2]);", ""), ("V += BK * int(params->V_strides[2]);", "")),
 "noKld": (("Ktile.load(K + K_load_off, int(params->K_strides[2]));", "Ktile.clear();"),),
 "noVld": (("Vtile.load(V + V_load_off, int(params->V_strides[2]));", "Vtile.clear();"),),
 "noKVld": (("Ktile.load(K + K_load_off, int(params->K_strides[2]));", "Ktile.clear();"), ("Vtile.load(V + V_load_off, int(params->V_strides[2]));", "Vtile.clear();")),
 "nomask": (("if (do_causal && kb >= kb_min_causal) {", "if (false) {"),),
}
st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
fs = {}
for n, e in V.items():
    fs[n] = st if e is None else (lambda e=e, n=n: A.attention(q, k, v, 0.0625, 64, 32, 4, 2, e, n))
for f in fs.values(): mx.eval(f())
ts = {n: [] for n in fs}
for r in range(4):
    for n, f in fs.items(): ts[n].append(A.bench(f, 2, 2))
for n in fs:
    s = sorted(ts[n]); print(f"o={o} {n:8s}: best {s[0]*1e3:6.2f} ms {fl/s[0]/1e12:5.2f} TF  med {s[len(s)//2]*1e3:6.2f}", flush=True)

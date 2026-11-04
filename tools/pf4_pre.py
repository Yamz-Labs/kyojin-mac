"""pf4: software prefetch of the next block's first NPRE K tiles during softmax + P@V (hides load latency of the QK phase). usage: pf4_pre.py O"""
import sys, os, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ds84 as A
import pf4_pipe as P
def build(name, npre):
    body = A._parts()[2]
    def rep(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    rep("  // Loop over KV seq length\n  for (int kb = 0; kb < kb_lim; kb++) {", f"""  constexpr short NPRE = {npre};
  NAXTile<T, 2, 1> Kpre[NPRE];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < NPRE; id++) Kpre[id].load(K + id * kU, int(params->K_strides[2]));
  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {{""")
    a = body.index("      for (short id = 0; id < TDh; id++) {\n        NAXTile<T, 2, 1> Ktile;"); b = body.index("metal::true_type{});\n      }", a) + len("metal::true_type{});\n      }")
    body = body[:a] + """      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        if (id < NPRE) { Ktile = Kpre[id]; } else { Ktile.load(K + id * kU, int(params->K_strides[2])); }
        stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{}, Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
      }""" + body[b:]
    rep("    // O = P @ V for this D slice.", """    if (kb + 1 < kb_lim) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < NPRE; id++) Kpre[id].load(K + BK * int(params->K_strides[2]) + id * kU, int(params->K_strides[2]));
    }
    // O = P @ V for this D slice.""")
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=A.header(True, True), source=A.PRE + body)
if __name__ == "__main__":
    o = int(sys.argv[1]); L = 2048; N = o + L; hd = 256
    q = mx.random.normal((1, 16, L, hd)).astype(mx.float16); k = mx.random.normal((1, 4, N, hd)).astype(mx.float16); v = mx.random.normal((1, 4, N, hd)).astype(mx.float16); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
    st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
    ref = st(); mx.eval(ref); fs = {"stock": st}
    base = P.build("pf4_basep", 0, 0); fs["fork"] = lambda: P.run(base, q, k, v)
    for npre in (2, 4, 8):
        kk = build(f"pf4_pre{npre}", npre); y = P.run(kk, q, k, v); mx.eval(y)
        print(npre, "max|d|", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), flush=True); fs[f"pre{npre}"] = (lambda kk=kk: P.run(kk, q, k, v))
    ts = {n: [] for n in fs}
    for r in range(4):
        for n, f in fs.items(): ts[n].append(A.bench(f, 2, 2))
    for n in fs:
        s = sorted(ts[n]); print(f"o={o} {n}: best {fl/s[0]/1e12:.2f} TF med {fl/s[len(s)//2]/1e12:.2f} ({s[0]*1e3:.2f} ms)", flush=True)

"""pf4: load batching variants of the stock-structure hd256 kernel (all K (V) fragments of a block loaded before the MMAs), A/B vs stock. usage: pf4_pipe.py O"""
import sys, os, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ds84 as A
K_OLD = A._parts()[2]
def cut(body, a_str, b_str, new):
    a = body.index(a_str); b = body.index(b_str, a) + len(b_str); return body[:a] + new + body[b:]
KB = ("      for (short id = 0; id < TDh; id++) {\n        NAXTile<T, 2, 1> Ktile;", "metal::true_type{});\n      }", """      for (short iz = 0; iz < 1; iz++) {
      NAXTile<T, 2, 1> Kall[TDh];
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) Kall[id].load(K + ik * kU * int(params->K_strides[2]) + id * kU, int(params->K_strides[2]));
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{}, Kall[id].frag_at(0, 0), Kall[id].frag_at(1, 0), metal::true_type{});
      }
      }""")
VB = ("        NAXTile<T, 1, 2> Vtile;", "metal::false_type{});\n      }\n    }\n\n    // Next block", """        NAXTile<T, 1, 2> Vtile;
        Vtile.load(V + ik * kU * int(params->V_strides[2]) + id * kU, int(params->V_strides[2]));
        otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{}, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
      }
    }

    // Next block""")
VB2 = ("    // O = P @ V for this D slice.", "    // Next block", """    // O = P @ V for this D slice (all V fragments of the block loaded first).
    {
      NAXTile<T, 1, 2> Vall[TDh / 2][TK];
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) Vall[id / 2][ik].load(V + ik * kU * int(params->V_strides[2]) + id * kU, int(params->V_strides[2]));
      }
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++)
          otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{}, Vall[id / 2][ik].frag_at(0, 0), Vall[id / 2][ik].frag_at(0, 1), metal::false_type{});
      }
    }

    // Next block""")
def build(name, kb, vb, align_q=True, align_k=True):
    body = K_OLD
    if kb: body = cut(body, *KB)
    if vb: body = cut(body, *VB2)
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=A.header(align_q, align_k), source=A.PRE + body)
def run(kk, q, k, v, BQ=64, BK=32, WM=4, WN=2):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    p = A.make_params(B, H, Hkv, qL, kL, D, BQ, BK, 0.0625); NQ = qL // BQ
    return kk(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN)],
              grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]
if __name__ == "__main__":
    o = int(sys.argv[1]); L = 2048; N = o + L; hd = 256
    q = mx.random.normal((1, 16, L, hd)).astype(mx.float16); k = mx.random.normal((1, 4, N, hd)).astype(mx.float16); v = mx.random.normal((1, 4, N, hd)).astype(mx.float16); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
    st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
    ref = st(); mx.eval(ref); fs = {"stock": st}
    for nm, kb, vb in [("base", 0, 0), ("kbatch", 1, 0), ("vbatch", 0, 1), ("kvbatch", 1, 1)]:
        kk = build("pf4_" + nm, kb, vb); y = run(kk, q, k, v); mx.eval(y)
        print(nm, "max|d|", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), flush=True); fs[nm] = (lambda kk=kk: run(kk, q, k, v))
    ts = {n: [] for n in fs}
    for r in range(4):
        for n, f in fs.items(): ts[n].append(A.bench(f, 2, 2))
    for n in fs:
        s = sorted(ts[n]); print(f"o={o} {n}: best {fl/s[0]/1e12:.2f} TF med {fl/s[len(s)//2]/1e12:.2f} ({s[0]*1e3:.2f} ms)", flush=True)

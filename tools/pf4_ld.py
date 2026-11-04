"""pf4: hd256 NAX dsplit attention with wide per-lane K/Q/V loads and O stores.
Stock NAX fragment loads give each lane 8 B pieces (4 halves) of a row: 8 rows x 32 B per instruction, 2 per frag. Here the head dim is permuted
inside each 32-column block (same permutation on Q and K, so QK^T is unchanged up to fp32 summation order; V columns are unpermuted at the store),
so that a lane's pieces of two consecutive fragments are contiguous: one 16 B load per row instead of two 8 B loads.
Aligned shapes only (qL % BQ == 0, kL % BK == 0); callers fall back to stock otherwise.  usage: pf4_ld.py O"""
import sys, os, functools, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ds84 as A

HELP = """
template <typename T, typename F>
METAL_FUNC void pf4_ld2(const device T* base, int ld, short fm, short p, thread F& a, thread F& b) {
#if PF4_NARROW
  const device T* s = base + fm * ld + p * 4;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; j++) { a[i * 4 + j] = s[i * 8 * ld + j]; b[i * 4 + j] = s[i * 8 * ld + 16 + j]; }
  }
#else
  const device T* s = base + fm * ld + p * 8;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; i++) {
    uint4 u = *(const device uint4*)(s + i * 8 * ld);
    metal::vec<T, 8> v = as_type<metal::vec<T, 8>>(u);
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; j++) { a[i * 4 + j] = v[j]; b[i * 4 + j] = v[4 + j]; }
  }
#endif
}
template <typename T, typename F>
METAL_FUNC void pf4_st2(device T* base, int ld, short fm, short p, thread F& a, thread F& b) {
  device T* s = base + fm * ld + p * 8;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; i++) {
    metal::vec<T, 8> v;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; j++) { v[j] = static_cast<T>(a[i * 4 + j]); v[4 + j] = static_cast<T>(b[i * 4 + j]); }
    *(device metal::vec<T, 8>*)(s + i * 8 * ld) = v;
  }
}
"""

NARROW = int(os.environ.get('PF4_NARROW', '0'))
def ld_body(body, kw=True, vw=True, qw=True, ow=True):
    def cut(a_str, b_str, new, incl_b=True):
        nonlocal body
        a = body.index(a_str); b = body.index(b_str, a) + (len(b_str) if incl_b else 0)
        body = body[:a] + new + body[b:]
    def rep(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    perm = kw or qw
    assert kw == qw
    if qw:
        cut("  NAXTile<T, 1, 1> Qtiles[TDh];", "  const short2 simd_coord", """  NAXTile<T, 1, 1> Qtiles[TDh];
  const short pf_m = otile_t::NAXFrag_t::get_coord().y, pf_p = otile_t::NAXFrag_t::get_coord().x / 4;
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id += 2)
    pf4_ld2(Q + id * kU, int(params->Q_strides[2]), pf_m, pf_p, Qtiles[id].frag_at(0, 0), Qtiles[id + 1].frag_at(0, 0));

""", incl_b=False)
    if kw:
        cut("      for (short id = 0; id < TDh; id++) {\n        NAXTile<T, 2, 1> Ktile;", "metal::true_type{});\n      }", """      for (short id = 0; id < TDh; id += 2) {
        NAXTile<T, 2, 1> KA, KB;
        STEEL_PRAGMA_UNROLL
        for (short a = 0; a < 2; a++)
          pf4_ld2(K + (ik + a) * kU * int(params->K_strides[2]) + id * kU, int(params->K_strides[2]), sm, sn / 4, KA.frag_at(a, 0), KB.frag_at(a, 0));
        stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{}, KA.frag_at(0, 0), KA.frag_at(1, 0), metal::true_type{});
        stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id + 1].frag_at(0, 0), metal::false_type{}, KB.frag_at(0, 0), KB.frag_at(1, 0), metal::true_type{});
      }""")
    if vw:
        cut("        NAXTile<T, 1, 2> Vtile;", "metal::false_type{});\n      }\n    }\n\n    // Next block", """        NAXTile<T, 1, 2> Vtile;
        pf4_ld2(V + ik * kU * int(params->V_strides[2]) + id * kU, int(params->V_strides[2]), sm, sn / 4, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1));
        otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{}, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
      }
    }

    // Next block""")
    if ow:
        rep("    Otile.store(O, int(params->O_strides[2]));", """    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2)
      pf4_st2(O + id * kU, int(params->O_strides[2]), sm, sn / 4, Otile.frag_at(0, id), Otile.frag_at(0, id + 1));""")
    return body

@functools.lru_cache(maxsize=None)
def kern(name, align_q, align_k, mode):
    _, _, body = A._parts()
    body = ld_body(body, *mode)
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=f"#define PF4_NARROW {int(NARROW)}\n" + A.header(align_q, align_k) + HELP, source=A.PRE + body)

def attention(q, k, v, scale, BQ=64, BK=32, WM=4, WN=2, mode=(True, True, True, True)):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    assert qL % BQ == 0 and kL % BK == 0 and D == 256
    kk = kern(f"pf4ld_{BQ}_{BK}_{WM}_{WN}_{''.join(str(int(m)) for m in mode)}", True, True, mode)
    p = A.make_params(B, H, Hkv, qL, kL, D, BQ, BK, scale); NQ = qL // BQ
    return kk(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN)],
              grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

if __name__ == "__main__":
    o = int(sys.argv[1]); L = int(sys.argv[2]) if len(sys.argv) > 2 else 2048; N = o + L; hd = 256
    dt = mx.float16
    q = mx.random.normal((1, 16, L, hd)).astype(dt); k = mx.random.normal((1, 4, N, hd)).astype(dt); v = mx.random.normal((1, 4, N, hd)).astype(dt); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
    st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
    ref = st(); mx.eval(ref); fs = {"stock": st}
    for m in [(True, True, True, True), (False, True, False, False), (True, False, True, False), (False, False, False, True)]:
        try:
            y = attention(q, k, v, 0.0625, mode=m); mx.eval(y)
            ref32 = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=0.0625, mask="causal")
            print(m, "max|d| vs stock", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), "vs fp32 ref", float(mx.abs(y.astype(mx.float32) - ref32).max()), "stock vs fp32", float(mx.abs(ref.astype(mx.float32) - ref32).max()), flush=True)
            fs[str(m)] = (lambda m=m: attention(q, k, v, 0.0625, mode=m))
        except Exception as e: print(m, "ERR", str(e)[-300:], flush=True)
    ts = {n: [] for n in fs}
    for r in range(4):
        for n, f in fs.items(): ts[n].append(A.bench(f, 2, 2))
    for n in fs:
        s = sorted(ts[n]); print(f"o={o} {n}: best {fl/s[0]/1e12:.2f} TF med {fl/s[len(s)//2]/1e12:.2f} ({s[0]*1e3:.2f} ms)", flush=True)

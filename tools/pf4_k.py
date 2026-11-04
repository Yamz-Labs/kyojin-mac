"""pf4: TQ-generalised copy of MLX hd256 NAX dsplit attention (TQ q-fragments per simdgroup: each K/V fragment load feeds TQ MMAs).
usage: pf4_k.py O   (A/B vs stock, interleaved)"""
import sys, os, functools, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ds84 as A

def tq_body(body):
    def rep(old, new, cnt=1):
        nonlocal body
        assert body.count(old) == cnt, (body.count(old), old)
        body = body.replace(old, new)
    rep('static_assert(TQ == 1, "Check TQ");', "")
    rep("s_xchg[WM][WN][TK * kEPF * 32]", "s_xchg[WM][WN][TQ * TK * kEPF * 32]")
    rep("NAXTile<T, 1, 1> Qtiles[TDh];", "NAXTile<T, TQ, 1> Qtiles[TDh];")
    rep("""        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),""", """        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++)
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(iq, ik),
            Stile.frag_at(iq, ik + 1),
            Qtiles[id].frag_at(iq, 0),""")
    a = body.index("    // Exchange all partial scores"); b = body.index("    // Scale S")
    body = body[:a] + """    // Exchange all partial scores and reduce them in a fixed order.
    threadgroup AccumType* slot = s_xchg[row_group][d_group];
    const short base = short(simd_lane_id) * (TQ * TK * kEPF);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < TQ * TK * kEPF; e++) slot[base + e] = Stile.elems()[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < TQ * TK * kEPF; e++) {
      AccumType s = s_xchg[row_group][0][base + e];
      STEEL_PRAGMA_UNROLL
      for (short peer_group = 1; peer_group < WN; peer_group++) s += s_xchg[row_group][peer_group][base + e];
      Stile.elems()[e] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

""" + body[b:]
    rep("""      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);""", """      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++)
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(iq, ik);""")
    rep("""      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii""", """      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++)
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(iq, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + iq * kU + ii""")
    rep("""        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),""", """        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++)
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(iq, id),
            Otile.frag_at(iq, id + 1),
            Stile.frag_at(iq, ik),""")
    return body

@functools.lru_cache(maxsize=None)
def kern(name, align_q, align_k, edits):
    _, _, body = A._parts()
    body = tq_body(body)
    for old, new in edits:
        assert old in body, old
        body = body.replace(old, new)
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=A.header(align_q, align_k), source=A.PRE + body)

def attention(q, k, v, scale, BQ, BK, WM, WN, edits=(), tag="tq"):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    kk = kern(f"pf4_{tag}_{BQ}_{BK}_{WM}_{WN}", qL % BQ == 0, kL % BK == 0, tuple(edits))
    p = A.make_params(B, H, Hkv, qL, kL, D, BQ, BK, scale); NQ = (qL + BQ - 1) // BQ
    return kk(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN)],
              grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

if __name__ == "__main__":
    o = int(sys.argv[1]); L = 2048; N = o + L; hd = 256
    q = mx.random.normal((1, 16, L, hd)).astype(mx.float16); k = mx.random.normal((1, 4, N, hd)).astype(mx.float16); v = mx.random.normal((1, 4, N, hd)).astype(mx.float16); mx.eval(q, k, v)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * hd
    st = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
    ref = st(); mx.eval(ref); fs = {"stock": st}
    for c in [(64, 32, 4, 2), (64, 32, 2, 2), (64, 32, 2, 4), (128, 32, 4, 2), (32, 32, 1, 2), (32, 32, 1, 4)]:
        try:
            y = attention(q, k, v, 0.0625, *c); mx.eval(y)
            print(c, "max|d|", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), flush=True); fs[str(c)] = (lambda c=c: attention(q, k, v, 0.0625, *c))
        except Exception as e: print(c, "ERR", str(e)[-160:], flush=True)
    ts = {n: [] for n in fs}
    for r in range(4):
        for n, f in fs.items(): ts[n].append(A.bench(f, 2, 2))
    for n in fs:
        s = sorted(ts[n]); print(f"o={o} {n}: best {fl/s[0]/1e12:.2f} TF med {fl/s[len(s)//2]/1e12:.2f} ({s[0]*1e3:.2f} ms)", flush=True)

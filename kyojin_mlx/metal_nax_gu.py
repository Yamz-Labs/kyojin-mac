# SPDX-License-Identifier: Apache-2.0
"""pf_moe_epi: gate + up + SwiGLU + routing weight of the prefill MoE in ONE kernel (two accumulator sets).

Ported from the unmerged perf/pfmoe branch (commits b99661d, 8deddcd, 562609b), made generic and re-based on the
current coalesced GEMM + mid code. One simdgroup owns the same output slab (bns columns, bmu * nmb rows) of the gate
and of the up projection. Per k step it decodes the gate weights into the right operand, runs the matmuls into the gate
accumulators, decodes the up weights and runs the matmuls into the up accumulators. Every output element keeps the
reduction order of the production `metal_nax.coop_source` kernel (same descriptor, same k loop, same fp32 accumulators),
so gate and up are bit-identical to `expert_gemm`. The epilogue rounds gate / up to half (as the production Y store),
then runs the `mid` kernel body per row (H128 + svh, SwiGLU, routing weight, fp8 fake quant, H128 with suh_down, k
permutation): the output is the permuted activation xh2 [n, N] that the down GEMM reads. Neither the [2P, inter] gate /
up product nor a separate mid launch is needed.

With `rowmap` (pf_moe_rowmap on as well) A is None: the gate rows are staged from x through the slot -> token map with
gate's suh (and the up rows with up's suh) inside the kernel, so no [2P, K] copy exists either.

A differs per projection (gate rows and up rows of xh were rotated with their own sign vectors), so A is one buffer
xh [2n, KD]: gate rows first. Grid (nt / (NSG * nt_sg) * 32 * NSG, ceil(maxrows / (bmu * nmb)), GL).
"""
from __future__ import annotations

import functools
import itertools

import mlx.core as mx

from . import metal_nax as NX
from . import metal_src as S

_ids = itertools.count()
_kc: dict = {}


def _mid_tail(limit: float, wpre: bool, fq: bool, alpha: float, beta: float) -> str:
    """The `mid` body after the gate / up values gu[2] exist: SwiGLU, routing weight, fp8 fake quant, rotation, permuted half store."""
    s = S.mid_source(limit, wpre, fq, alpha, beta)
    i = s.index("float4 g4 = min(")
    t = s[i:]
    old = "*((device float4*)(xh2 + ulong(p) * N + col)) = float4(v0, v1, v2, v3) * " + S.RSQRT_HAD + ";"
    assert t.count(old) == 1
    return t.replace(old, S.perm_store("Y + ulong(p) * N + blk * 128", "float4(v0, v1, v2, v3) * " + S.RSQRT_HAD, True))


@functools.lru_cache(maxsize=None)
def gu_source(Kg: int, Ku: int, cb: str, bmu: int, nmb: int, bns: int, bks: int, limit: float, wpre: bool, fq: bool,
              alpha: float, beta: float, skip_tail: bool = True, rowmap: bool = False) -> str:
    """Inputs: eids i32 [GL], rowoff u32 [GL + 1], tr0 tr1 u32 (gate, up trellis), off0 off1 u32 [E + 1],
    A half [2n, KD] (gate rows, up rows; k permuted) or, with rowmap, x half [T, KD], tok u32 [n vs slots],
    suh0 suh1 half [E, KD]; params u32 [KT, NT, GL, n], svh0 svh1 half [E, N], suhd half [E, N], wts f32 [n].
    Output Y half [n, N] (the permuted SwiGLU activation)."""
    nt_sg, nk = bns // 16, bks // 16
    nsg = 128 // bns
    nsub = nk * nt_sg

    def mm(proj: str, shape: str = "kt") -> str:
        """Matmuls of one projection for the nk k-tiles decoded into rB."""
        ta = "tAr" if rowmap else ("tAg" if proj == "" else "tAu")
        cp = "g" if proj == "" else "u"
        return " ".join(f"{'if (mg + %du < M) ' % (bmu * mb) if skip_tail and mb else ''}"
                        f"{{ auto sA = {ta}.slice(int({shape}) * 16, {(str(bmu * mb)) if rowmap else ('int(mg) + %d' % (bmu * mb))}); op.run(sA, rB, cT{cp}{mb}); }}"
                        for mb in range(nmb))

    if rowmap:
        assert nmb == 1, "row map staging keeps one 32-row block in threadgroup memory"
        body = []
        # rows past rblk (the padding of a tail block) are never transformed: their A rows stay zero, as when they were staged from a zero input
        body += [f"for (uint i = sg * 32u + lane; i < {bmu}u * 128u; i += {nsg * 32}u) atg[i] = half(0.0f);"]
        body += [f"for (uint kb = 0; kb < KT / 8u; ++kb) {{"]
        for pre, suh in (("", "suh0"), ("u", "suh1")):
            Kp = Kg if pre == "" else Ku
            tb = 256 * Kp
            body += [f"  for (uint r0 = sg * 4u; r0 < {bmu}u; r0 += {4 * nsg}u) {{",
                     "    float4 xv[4], sv[4];",
                     "    #pragma unroll",
                     "    for (uint j = 0; j < 4u; ++j) {",
                     "      const uint r = r0 + j;",
                     "      if (r < rblk) {",
                     "        const ulong o = ulong(tok[row0 + mg + r]) * KD + kb * 128u + lane * 4;",
                     "        xv[j] = float4(*((const device half4*)(x + o)));",
                     f"        sv[j] = float4(*((const device half4*)({suh} + ulong(e) * KD + kb * 128u + lane * 4)));",
                     "      } else { xv[j] = float4(0.0f); sv[j] = float4(0.0f); }",
                     "    }",
                     "    #pragma unroll",
                     "    for (uint j = 0; j < 4u; ++j) {",
                     "      const uint r = r0 + j;",
                     "      if (r >= rblk) continue;                    // rblk <= bmu: padded rows skip the butterfly",
                     "      float v0 = xv[j].x * sv[j].x, v1 = xv[j].y * sv[j].y, v2 = xv[j].z * sv[j].z, v3 = xv[j].w * sv[j].w;",
                     S._BUTTERFLY,
                     "{ const float4 pv = float4(v0, v1, v2, v3) * " + S.RSQRT_HAD + ";",
                     "        threadgroup half* pd = (threadgroup half*)(atg + r * 128) + (lane >> 2) * 16 + 8 * (lane & 1) + 2 * ((lane >> 1) & 1);",
                     "        pd[0] = half(pv.x); pd[1] = half(pv.y); pd[4] = half(pv.z); pd[5] = half(pv.w); }",
                     "    }",
                     "  }",
                     "  threadgroup_barrier(mem_flags::mem_threadgroup);",
                     f"  {{ uint B0 = {32 * Kp}u * nb + {Kp}u + 8u * G * {Kp}u + {tb}u - 16u; if (B0 >= {tb}u) B0 -= {tb}u;",
                     f"  for (uint ks = 0; ks < 8u; ks += {nk}u) {{"]
            for t in range(nk):
                kt = f"kb * 8u + ks + {t}u"
                for jj in range(nt_sg):
                    body += ["    " + l for l in NX._tile_decode(Kp, cb, f"tbase{pre} + (ulong({kt}) * NT + nt0 + {jj}u) * {8 * Kp}u",
                                                                 "rB", (t * nt_sg + jj) * 8)]
            body += ["    " + mm(pre, "ks"), "  } }", "  threadgroup_barrier(mem_flags::mem_threadgroup);"]
        body += ["}"]
        body_src = "\n".join(body)
    else:
        L = [f"for (uint kt = 0; kt < KT; kt += {nk}u) {{"]
        for pre, K in (("", Kg), ("u", Ku)):
            tb = 256 * K
            L += [f"  {{ uint B0 = {32 * K}u * nb + {K}u + 8u * G * {K}u + {tb}u - 16u; if (B0 >= {tb}u) B0 -= {tb}u;"]
            for t in range(nk):
                for jj in range(nt_sg):
                    L += ["    " + l for l in NX._tile_decode(K, cb, f"tbase{pre} + (ulong(kt + {t}u) * NT + nt0 + {jj}u) * {8 * K}u",
                                                              "rB", (t * nt_sg + jj) * 8)]
            L += ["    " + mm(pre), "  }"]
        L += ["}"]
        body_src = "\n".join(L)

    ra = "mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t"
    decl = "\n".join(f"  auto cT{p}{mb} = op.get_destination_cooperative_tensor<{ra}<decltype(tA0)>, {ra}<decltype(rB)>, float>();"
                     for p in "gu" for mb in range(nmb))
    zero = "\n".join(f"  for (uint16_t i = 0; i < cT{p}{mb}.get_capacity(); ++i) if (cT{p}{mb}.is_valid_element(i)) cT{p}{mb}[i] = 0.0f;"
                     for p in "gu" for mb in range(nmb))
    epi = []
    for mb in range(nmb):
        for p in "gu":
            epi.append(f"  for (uint16_t i = 0; i < cT{p}{mb}.get_capacity(); ++i) if (cT{p}{mb}.is_valid_element(i)) {{ const auto ix = cT{p}{mb}.get_multidimensional_index(i);"
                       f" fb{p}[uint(ix[1]) * 128 + sg * {bns}u + uint(ix[0])] = half(cT{p}{mb}[i]); }}")
        epi += ["  threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"  for (uint r = sg; r < {bmu}u; r += {nsg}u) {{",
                f"    if (mg + {bmu * mb}u + r >= M) continue;",
                f"    const uint p = row0 + mg + {bmu * mb}u + r;",
                "    float4 gu[2];",
                "    for (uint pj = 0; pj < 2; ++pj) {",
                "      const half4 hv = *((threadgroup half4*)((pj ? fbu : fbg) + r * 128 + lane * 4));",
                "      float v0 = float(hv.x), v1 = float(hv.y), v2 = float(hv.z), v3 = float(hv.w);",
                S._BUTTERFLY,
                "      const float4 sv = float4(*((const device half4*)((pj ? svh1 : svh0) + ulong(e) * N + col)));",
                "      gu[pj] = float4(v0, v1, v2, v3) * " + S.RSQRT_HAD + " * sv;",
                "    }",
                _mid_tail(limit, wpre, fq, alpha, beta),
                "  }",
                "  threadgroup_barrier(mem_flags::mem_threadgroup);"]
    store = "\n".join(epi)
    abuf = (f"threadgroup half atg[{bmu * nmb} * 128];\n"
            f"    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tAr((threadgroup half*)atg, dextents<int32_t, 2>(128, (uint)({bmu * nmb})));"
            ) if rowmap else ""
    adecl = (f"auto tA0 = tAr.slice(0, int(mg));") if rowmap else (
        "tensor<device half, dextents<int32_t, 2>, tensor_inline> tAg((device half*)(A + ulong(row0) * KD), dextents<int32_t, 2>(int(KD), int(M)));\n"
        "    tensor<device half, dextents<int32_t, 2>, tensor_inline> tAu((device half*)(A + (ulong(NN) + row0) * KD), dextents<int32_t, 2>(int(KD), int(M)));\n"
        "    auto tA0 = tAg.slice(0, int(mg));")
    return _T.format(BMU=bmu, BNS=bns, BKS=bks, NTSG=nt_sg, NSG=nsg, BODY=body_src, DECL=decl, ZERO=zero, STORE=store,
                     ABUF=abuf, ADECL=adecl)


_T = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint z = threadgroup_position_in_grid.z;
    const uint KT = params[0], NT = params[1], GL = params[2], NN = params[3];
    const uint e = uint(eids[z]);
    const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;
    const uint mg = threadgroup_position_in_grid.y * {BMU} * ROWSMB;
    const uint nt0 = (threadgroup_position_in_grid.x * {NSG} + sg) * {NTSG};
    if (mg >= M || nt0 >= NT) return;
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint blk = threadgroup_position_in_grid.x;
    const ulong col = ulong(blk) * 128 + lane * 4;
    const uint rblk = min((uint)({BMU} * ROWSMB), M - mg);
    const device uint* tbase = tr0 + off0[e];
    const device uint* tbaseu = tr1 + off1[e];
    {ABUF}
    threadgroup half fbg[{BMU} * 128];
    threadgroup half fbu[{BMU} * 128];

    constexpr auto desc = matmul2d_descriptor({BMU}, {BNS}, {BKS}, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    {ADECL}
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
{DECL}
{ZERO}
    const ushort i0 = 0;
    const auto ix0 = rB.get_multidimensional_index(i0);
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
{BODY}
{STORE}
"""


def fused_gu_swiglu(g, u, d, eids: mx.array, ex, rowoff: mx.array, A: mx.array | None, n: int, maxrows: int,
                    cfg: tuple[int, int, int, int], wts_s: mx.array, m, rowmap: tuple | None = None, out_rows: int | None = None) -> mx.array:
    """xh2 [n, N] half = SwiGLU(gate, up) of the rows (sorted by expert) of A [2n, KD]: gate / up / activation in one launch.
    ex: host expert ids of eids. g, u: gate / up stacks (one bitrate each over eids), d: down stack (its suh rotates the
    activation), m: the MoeGather (limit, wpre, fq, alpha, beta). rowmap=(x, tok, suh0, suh1): read x instead of A.
    out_rows (pf_moe_epi_cap): rows to ALLOCATE for the output (>= n; the kernel writes rows < n only). The class row count changes every layer, and the MLX buffer
    cache reuses a free buffer only within 2 pages of the request, so an exact-size output misses it on every layer; the callers' other pair buffers are bucketed
    (`moe_prefill.cap_rows`)."""
    GL = int(eids.shape[0])
    kgs, kus = set(g.kh[ex].tolist()), set(u.kh[ex].tolist())
    assert len(kgs) == 1 and len(kus) == 1, (kgs, kus)      # one gate / up bitrate per launch (the prefill plan runs one pass per bitrate class)
    Kg, Ku = int(kgs.pop()), int(kus.pop())
    bmu, nmb, bns, bks = cfg
    # Two accumulator sets double the register file: one 16-column tile per simdgroup (8 simdgroups = one 128-column block
    # per threadgroup, the epilogue's unit) keeps the occupancy. Measured on the Ultra, T=1024/E=64/5120x2304: bns 32 14.8 ms
    # vs bns 16 8.4 ms for the same gate/up/activation.
    bns = 16
    if bks % 64 == 0 and g.kt % (bks // 16):
        bks = 32
    assert bns in (16, 32) and g.nout == u.nout and g.kt == u.kt and g.cb == u.cb
    assert g.nout % 128 == 0, "one threadgroup owns whole 128-column blocks (the epilogue reads svh/suhd by block)"
    rm = rowmap is not None
    if rm:
        assert len(rowmap) == 4 and g.kin % 128 == 0
        bmu, nmb = 32, 1                                     # staging keeps one 32-row block (8 + 8 + 8 KB threadgroup)
    cbk = NX.dec_cb(g.cb)
    key = (Kg, Ku, cbk, bmu, nmb, bns, bks, m.limit, m.wpre, m.fq, m.alpha, m.beta, NX.SKIP_TAIL, rm)
    kern = _kc.get(key)
    if kern is None:
        src = gu_source(Kg, Ku, cbk, bmu, nmb, bns, bks, m.limit, m.wpre, m.fq, m.alpha, m.beta, NX.SKIP_TAIL, rm).replace("ROWSMB", str(nmb))
        names = ["eids", "rowoff", "tr0", "tr1", "off0", "off1"] + (["x", "tok", "suh0", "suh1"] if rm else ["A"]) + \
                ["params", "svh0", "svh1", "suhd", "wts"]
        kern = _kc[key] = mx.fast.metal_kernel(name=f"kyojin_gu_{cbk}_{Kg}_{Ku}_{bmu}_{nmb}_{bns}_{bks}_{'rm' if rm else 'a'}_{next(_ids)}",
                                               header=NX.HEADER_NAX, input_names=names, output_names=["Y"], source=src)
    nsg = 128 // bns
    params = mx.array([g.kt, g.nt, GL, n], dtype=mx.uint32)
    ins = [eids, rowoff, g.words, u.words, NX._device_array(g.offset), NX._device_array(u.offset)] + \
          (list(rowmap) if rm else [A]) + [params, g.svh, u.svh, d.suh, wts_s]
    return kern(inputs=ins, grid=(-(-g.nt // (nsg * (bns // 16))) * 32 * nsg, -(-maxrows // (bmu * nmb)), GL), threadgroup=(32 * nsg, 1, 1),
                output_shapes=[(max(n, out_rows or n), g.nout)], output_dtypes=[mx.float16])[0]

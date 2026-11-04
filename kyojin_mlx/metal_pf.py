# SPDX-License-Identifier: Apache-2.0
"""Metal sources for the MoE prefill path (S3, kyojin_mlx.moe_prefill). Derived from the decode-path templates in metal_src.py by
text substitution (asserted), so the per-element arithmetic of rot / mid / final is the same code as in the decode path; only the
index maps change (pairs sorted by expert, projection-major layout, half or float intermediates).

Pair order inside a group: pairs sorted by expert id. Layout [proj, pair]: row pj * NP + p.
  decode_pf   w[pj * G + g]  = Wr^T fp16 [N, KD] of expert eids[g] of projection pj (ragged K, one launch for a whole expert group)
  rot_pf      xh[pj * NP + p] = H128(x[tok[p]] * suh_pj[ids[p]]) / sqrt(128)
  mid_pf      xh2[p]         = H128(act(gate, up) [* w] [fq] * suh_down[e]) / sqrt(128), gate / up read from part[pj * NP + p]
  final_pf    y[r]           = sum_j [w_j *] H128(part[inv[r * TOPK + j]]) / sqrt(128) * svh[e]   (j in order, fp32)
The matmul between rot_pf and mid_pf (and after mid_pf) is mx.gather_mm on the decoded fp16 weights."""
from __future__ import annotations

import functools

from . import metal_src as S

HEADER_PF = S.HEADER_MOE

_DECODE_PF = r"""
    // Wr^T fp16 [N, KD] for expert eids[zz % G] of projection zz / G. One thread per (tile, run g), grid.y = proj * G + local expert.
    const uint gid = thread_position_in_grid.x;
    const uint zz = thread_position_in_grid.y;
    const uint KT = params[0], NT = params[1], G = params[2];
    const uint g = gid & 7, tile = gid >> 3;
    if (tile >= KT * NT) return;
    const uint e = uint(eids[zz % G]);
    TRSEL
    const uint K = uint(kk[e]);
    const uint kt = tile % KT, nt = tile / KT;                    // kt fastest: a simdgroup writes 4 consecutive 32-byte k-slices of 8 rows
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    device half* wb = w + ulong(zz) * N * KD;
    device half* wlo = wb + (ulong(nt) * 16 + g) * KD + kt * 16;
    device half* whi = wb + (ulong(nt) * 16 + 8 + g) * KD + kt * 16;
    const device uint* tbase = trellis + off[e];
    BODIES
"""

_DECODE_BODY = r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp = tbase + (ulong(kt) * NT + nt) * TW;
        LOADS
        RUN
    }"""


@functools.lru_cache(maxsize=None)
def decode_pf_source(Ks: tuple[int, ...], nproj: int, cb: str = "mul1") -> str:
    """Inputs: eids i32 [G], tr0 (tr1) u32, off0 (off1) u32 [E+1], kk0 (kk1) u8 [E], params u32 [KT, NT, G]. Output w half [nproj * G, N, KD]."""
    if nproj == 1:
        trsel = "const device uint* trellis = tr0; const device uint* off = off0; const device uint8_t* kk = kk0;"
    else:
        trsel = ("const bool pj = (zz / G) != 0u; const device uint* trellis = pj ? tr1 : tr0; "
                 "const device uint* off = pj ? off1 : off0; const device uint8_t* kk = pj ? kk1 : kk0;")
    bodies = []
    for K in Ks:
        loads, unpack = S.gen_loads(K)
        ld = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
        run = S.run_decode_lines(K, lambda p, kk, hi, e: f"{'whi' if hi else 'wlo'}[{kk}] = {e};", cb)
        b = _DECODE_BODY.replace("KV", str(K)).replace("LOADS", "\n        ".join(ld)).replace("RUN", "\n        ".join(run))
        bodies.append(b)
    chain = " else ".join(f"if (K == {K}u) {b}" for K, b in zip(Ks, bodies))
    return _DECODE_PF.replace("TRSEL", trsel).replace("BODIES", chain)


def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"template changed, {old!r} found {text.count(old)} times"
    return text.replace(old, new)


_ROT_PF = r"""
    // xh[pj * NP + p] = H128(x[tok[p]] * suh_pj[ids[p]]) / sqrt(128). One simdgroup per (128-block, proj * pair).
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint zz = threadgroup_position_in_grid.y;
    const uint KD = params[0], NP = params[1];
    if (blk * 128 >= KD) return;
    const uint p = zz % NP;
    const uint e = uint(ids[p]);
    const device half* sp = SUHSEL + ulong(e) * KD;
    const ulong o = ulong(tok[p]) * KD + blk * 128 + lane * 4;
    const float4 xv = float4(*((const device XV4*)(x + o)));
    const float4 sv = float4(*((const device half4*)(sp + blk * 128 + lane * 4)));
    float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
    *((device OV4*)(xh + ulong(zz) * KD + blk * 128 + lane * 4)) = OV4(float4(v0, v1, v2, v3) * RSQRT);
"""


@functools.lru_cache(maxsize=None)
def rot_pf_source(nproj: int, in_half: bool, out_half: bool, perm: bool = False, fq8: bool = False) -> str:
    """Inputs: x [T, KD], tok u32 [NP], ids i32 [NP], suh0 (suh1) half [E, KD], params u32 [KD, NP]. Output xh [nproj * NP, KD].
    perm: k permuted inside 16-blocks (the layout of the fused EXL3 + NAX GEMM, metal_nax). fq8: the fp8 round trip of x folded into the load (fp16 x)."""
    sel = "suh0" if nproj == 1 else "((zz / NP) != 0u ? suh1 : suh0)"
    t = _ROT_PF
    if perm:
        t = _sub(t, "*((device OV4*)(xh + ulong(zz) * KD + blk * 128 + lane * 4)) = OV4(float4(v0, v1, v2, v3) * RSQRT);",
                 S.perm_store("xh + ulong(zz) * KD + blk * 128", "float4(v0, v1, v2, v3) * RSQRT", out_half))
    src = (t.replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD).replace("SUHSEL", sel)
           .replace("XV4", "half4" if in_half else "float4").replace("OV4", "half4" if out_half else "float4"))
    assert not fq8 or in_half, "the folded fp8 round trip is for fp16 activations"
    return S.with_fq8(src) if fq8 else src


@functools.lru_cache(maxsize=None)
def mid_pf_source(limit: float, wpre: bool, fq: bool, alpha: float, beta: float, part_half: bool, out_half: bool, perm: bool = False,
                  bounded: bool = False) -> str:
    """Inputs: part [2 * NP, N] (gate rows then up rows), ids i32 [NP], wts f32 [NP], svh0, svh1, suhd half [E, N], params u32 [NP, N]
    (+ rows u32 [2] when bounded: the slot range this launch owns; the other rows return at once). Output xh2 [NP, N]."""
    s = S.mid_source(limit, wpre, fq, alpha, beta)
    s = _sub(s, "const uint TGK = params[0], N = params[1];", "const uint NP = params[0], N = params[1];")
    if bounded:                     # pf_moe_nosync: the class's row range is a device value (the sort's offsets never reach the host)
        s = _sub(s, "const uint p = threadgroup_position_in_grid.y;",
                 "const uint p = threadgroup_position_in_grid.y;\n    if (p < rows[0] || p >= rows[1]) return;")
    s = _sub(s, "for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p * 2 + pj) * TGK + kb) * N + col));",
             "acc = float4(*((const device PV4*)(part + (ulong(pj) * NP + p) * N + col)));")
    new = (S.perm_store("xh2 + ulong(p) * N + blk * 128", "float4(v0, v1, v2, v3) * " + S.RSQRT_HAD, out_half) if perm else
           "*((device OV4*)(xh2 + ulong(p) * N + col)) = OV4(float4(v0, v1, v2, v3) * " + S.RSQRT_HAD + ");")
    s = _sub(s, "*((device float4*)(xh2 + ulong(p) * N + col)) = float4(v0, v1, v2, v3) * " + S.RSQRT_HAD + ";", new)
    return s.replace("PV4", "half4" if part_half else "float4").replace("OV4", "half4" if out_half else "float4")


@functools.lru_cache(maxsize=None)
def final_pf_source(wpost: bool, part_half: bool, classes: int = 0) -> str:
    """Inputs: part [P, N] (pairs in sorted order), inv u32 [P] (original pair -> sorted row), ids i32 [P] (original order), wts f32 [P],
    svh half [E, N], params u32 [1, N, TOPK]. Output y f32 [T, N].
    classes > 0 (pf_moe_nosync): `part` is `classes` equal blocks of `params[3]` rows, one per bitrate class, and each block holds its slots at
    their GLOBAL slot index, so the right row is `c * cap + inv[p]`, `c` = the class whose slot range (cum[zb[c]] ..) holds it: extra inputs
    cum u32 [E+1] and zb u32 [classes]. Same arithmetic, same per-row order."""
    s = S.final_gather_source(wpost)
    if classes:
        s = _sub(s, "for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p) * TGK + kb) * N + col));",
                 "const uint sl = inv[p];\n        uint ci = 0;\n        for (uint j = 1; j < CLS; ++j) if (sl >= cum[zb[j]]) ci = j;\n"
                 "        acc = float4(*((const device PV4*)(part + (ulong(ci) * ulong(params[3]) + ulong(sl)) * N + col)));")
    else:
        s = _sub(s, "for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p) * TGK + kb) * N + col));",
                 "acc = float4(*((const device PV4*)(part + ulong(inv[p]) * N + col)));")
    return s.replace("PV4", "half4" if part_half else "float4").replace("CLS", str(classes))

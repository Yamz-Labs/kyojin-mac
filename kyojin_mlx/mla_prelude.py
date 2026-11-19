# SPDX-License-Identifier: Apache-2.0
"""Row preparation of an absorbed-MLA decode call in ONE launch (flag KYOJIN_MLA_PRELUDE_FUSE, default off).

After the three linears of a layer (q_a -> q_a_norm -> q_b, and kv_a) the stock fixed-state call runs ~8 small launches before the cache write: the kv_a norm of the latent, the two
rope calls (each on a strided slice: a copy first), the split / transpose copies of q, the concat of [latent | k_pe | pool value | pool gate], the saturating clamp and the dtype cast
of the row (n9 census, T = 1: swa layer 31 -> 22 dispatches with the fused attend, kpool layer 63 -> 54). Here one threadgroup per (row, part) does all of it (part 0 the cache row, parts 1..nh one head of q each; n9-b: a single threadgroup per row was 1.5 % slower than the stock launches):

  row    [1, 1, L, D]   latent = rms_norm(kv_a[:kv]) * w, k_pe = rope(kv_a[kv:]) (consecutive pairs, positions off + l), then the pool value / gate rows when given;
                        written in the cache dtype (fp16 with the saturating clamp to +-65504, or fp32): ready for the one slice_update.
  q_nope [1, nh, L, nope]  the q_b output split and transposed (contiguous).     q_pe [1, nh, L, rope]   the roped rest of q.
Generic by op class (any absorbed-MLA layer: DeepSeek V3 / V4.1, GLM, Hy4). The linears, q_a_norm, the cache write, the absorbs and the attend stay with their own levers.
Same maths as the stock ops (fp32 in the kernel: the stock path rounds the norm output to the activation dtype before the rope / cast)."""
from __future__ import annotations

import math
import os

import mlx.core as mx

from . import kernels as KN

ON = os.environ.get("KYOJIN_MLA_PRELUDE_FUSE", "0") not in ("0", "")      # flag mla_prelude_fuse (flags.py rewrites it at runtime)
NT = 128
DT = {mx.float16: "half", mx.float32: "float"}
_par: dict = {}

_SRC = r"""
    const uint g = threadgroup_position_in_grid.x;                  // one threadgroup per (row, part): part 0 = the cache row, part 1 + h = head h of q
    const uint l = g / uint(NH + 1), part = g % uint(NH + 1);
    const uint tid = thread_position_in_threadgroup.x;
    const int pos = off[0] + int(l);
    const float eps = par[0], lb = par[1];
    if (part == 0u) {
        const uint sg = tid / 32u, ln = tid % 32u;
        threadgroup float ps[NT / 32];
        float ss = 0.0f;
        for (uint d = tid; d < uint(KV); d += NT) { const float v = float(kva[l * uint(KV + RD) + d]); ss += v * v; }
        ss = simd_sum(ss);
        if (ln == 0u) ps[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < NT / 32; ++i) tot += ps[i];
        const float rs = metal::rsqrt(tot / float(KV) + eps);
        device OT* rw = row + ulong(l) * DROW;
        for (uint d = tid; d < uint(KV); d += NT) rw[d] = CVT(float(kva[l * uint(KV + RD) + d]) * rs * w[d]);
        for (uint i = tid; i < uint(RD / 2); i += NT) {
            const float x0 = float(kva[l * uint(KV + RD) + KV + 2 * i]), x1 = float(kva[l * uint(KV + RD) + KV + 2 * i + 1]);
            const float ang = float(pos) * metal::exp2(-(float(i) / float(RD / 2)) * lb);
            const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
            rw[KV + 2 * i] = CVT(x0 * c - x1 * s);
            rw[KV + 2 * i + 1] = CVT(x0 * s + x1 * c);
        }
        if (DX > 0) for (uint d = tid; d < uint(DX); d += NT) {
            rw[KV + RD + d] = CVT(float(vin[l * uint(DX) + d]));
            rw[KV + RD + DX + d] = CVT(float(gin[l * uint(DX) + d]));
        }
    } else {
        const uint h = part - 1u, HD = uint(NOPE + RD);
        const ulong sb = ulong(l) * uint(NH) * HD + ulong(h) * HD;
        for (uint e = tid; e < HD / 2u; e += NT) {
            const uint c = 2u * e;
            const float x0 = float(q[sb + c]), x1 = float(q[sb + c + 1]);
            if (c < uint(NOPE)) {
                const ulong dst = (ulong(h) * uint(L_) + l) * uint(NOPE) + c;
                qn[dst] = T(x0); qn[dst + 1] = T(x1);
            } else {
                const uint i = (c - uint(NOPE)) / 2u;
                const float ang = float(pos) * metal::exp2(-(float(i) / float(RD / 2)) * lb);
                const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
                const ulong dst = (ulong(h) * uint(L_) + l) * uint(RD) + (c - uint(NOPE));
                qp[dst] = T(x0 * cs - x1 * sn); qp[dst + 1] = T(x0 * sn + x1 * cs);
            }
        }
    }
"""


def eligible(x_rows: int, kva: mx.array, q: mx.array, buf_dtype, kv: int, rd: int, nope: int, why: dict | None = None) -> bool:
    def no(r):
        if why is not None:
            why["prelude_no_" + r] = why.get("prelude_no_" + r, 0) + 1
        return False
    if mx.default_device() != mx.gpu:
        return no("device")
    if kva.dtype not in DT or q.dtype != kva.dtype or buf_dtype not in DT:
        return no("dtype")
    if not 1 <= x_rows <= 16 or kva.shape[0] != 1 or rd % 2 or nope % 2 or kv % 2 or rd == 0:
        return no("shape")
    return True


def prelude(kva: mx.array, q: mx.array, w: mx.array, eps: float, base: float, off: mx.array, nh: int, kv: int, rd: int, nope: int, buf_dtype, vin=None, gin=None):
    """kva [1, L, kv + rd], q [1, L, nh * (nope + rd)] (the two linear outputs, one dtype), w [kv] -> (row [1, 1, L, D] in buf_dtype, q_nope [1, nh, L, nope], q_pe [1, nh, L, rd]);
    D = kv + rd (+ 2 dx with the pool value / gate rows vin, gin [1, L, dx])."""
    L = kva.shape[1]
    dx = 0 if vin is None else vin.shape[-1]
    D = kv + rd + 2 * dx
    key = ("mla_prelude", kva.dtype, buf_dtype)
    cvt = "CVT(v) (static_cast<OT>(metal::clamp(v, -65504.0f, 65504.0f)))" if buf_dtype == mx.float16 else "CVT(v) (static_cast<OT>(v))"
    src = "#define " + cvt + "\n" + _SRC
    kern = KN._kern(key, name=f"kyojin_mla_prelude_{DT[kva.dtype]}_{DT[buf_dtype]}", input_names=["kva", "q", "w", "off", "par", "vin", "gin"], output_names=["row", "qn", "qp"], source=src)
    pk = (eps, base)
    par = _par.get(pk)
    if par is None:
        par = _par[pk] = mx.array([eps, math.log2(base)], dtype=mx.float32)
    z = mx.zeros((1,), dtype=kva.dtype)
    row, qn, qp = kern(inputs=[kva.reshape(-1), q.reshape(-1), w.astype(mx.float32), off.astype(mx.int32).reshape(-1), par, vin.reshape(-1) if dx else z, gin.reshape(-1) if dx else z],
                       template=[("T", kva.dtype), ("OT", buf_dtype), ("NT", NT), ("L_", L), ("KV", kv), ("RD", rd), ("NOPE", nope), ("NH", nh), ("DX", dx), ("DROW", D)],
                       grid=(NT * L * (nh + 1), 1, 1), threadgroup=(NT, 1, 1),
                       output_shapes=[(1, 1, L, D), (1, nh, L, nope), (1, nh, L, rd)], output_dtypes=[buf_dtype, kva.dtype, kva.dtype])
    return row, qn, qp

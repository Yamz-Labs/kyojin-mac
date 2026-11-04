# SPDX-License-Identifier: Apache-2.0
"""Fewer launches per decode token (one row, compiled step regions), for every pack whose linears are MLX affine quantized (mlx4 / mlx8 packs).

The per-op table of the 1-row step (tools/roof96.py) puts every quantized GEMV at 96-100 % of the Air read bandwidth; what is left between the sum of
the kernels and the step is ~860 launches (~2 us each on the Air, a larger share of a ~3 ms step on the Ultra). Three op classes, shared by every
hybrid / dense decoder stack:
  1. same-input projections in one GEMV: q|k|v of attention, gate|up of the MLP, qkv|z|b|a of a gated-delta layer. Rows of an affine quantized
     matmul are independent (own scales, same K order), so the fused launch returns the same bits. The packed rows are concatenated once at load and
     the original modules keep views into the fused buffer (prefill and every other path run unchanged, no extra memory).
  2. residual add + RMSNorm in one launch (the add of every sub-layer feeds the next norm): writes the summed stream and its norm.
  3. gated-delta front end in one launch (conv state, depthwise conv, silu, q/k L2 norm, g / beta) plus the gated output norm in one (gdn_pf kernels,
     step kernel core).
KYOJIN_DECFUSE=0 restores the stock region bodies; KYOJIN_DECFUSE_PROJ / _ADDNORM / _GDN = 0 switch one class off."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

ENABLED = os.environ.get("KYOJIN_DECFUSE", "1") != "0"
PROJ = os.environ.get("KYOJIN_DECFUSE_PROJ", "1") != "0"
ADDNORM = os.environ.get("KYOJIN_DECFUSE_ADDNORM", "1") != "0"
GDN = os.environ.get("KYOJIN_DECFUSE_GDN", "1") != "0"
MOEFOLD = os.environ.get("KYOJIN_DECFUSE_MOEFOLD", "1") != "0"        # the MoE combine inside the next add + RMSNorm

# ---------------------------------------------------------------- 1. fused same-input projections


class FusedQ(nn.QuantizedLinear):
    """Concatenated quantized rows of several linears with the same input; parts(x) = the outputs of the originals (slices of one qmv)."""

    def parts(self, x):
        y = self(x)
        return [y[..., o:o + n] for o, n in self.spans]


def _fusable(mods):
    m0 = mods[0]
    return all(type(m) is nn.QuantizedLinear and "bias" not in m and (m.group_size, m.bits, getattr(m, "mode", "affine")) ==
               (m0.group_size, m0.bits, getattr(m0, "mode", "affine")) and m.weight.shape[1] == m0.weight.shape[1]
               and ("biases" in m) == ("biases" in m0) for m in mods)


def fuse(mods):
    """One FusedQ over `mods`; each original keeps views into the fused weight / scales / biases (no second copy). None if not fusable."""
    if not _fusable(mods):
        return None
    m0 = mods[0]
    sizes = [m.weight.shape[0] for m in mods]
    f = FusedQ(m0.weight.shape[1] * 32 // m0.bits, sum(sizes), bias=False, group_size=m0.group_size, bits=m0.bits,
               **({"mode": m0.mode} if hasattr(m0, "mode") else {}))
    names = ["weight", "scales"] + (["biases"] if "biases" in m0 else [])
    for nm in names:
        setattr(f, nm, mx.concatenate([getattr(m, nm) for m in mods], axis=0))
    if "biases" not in m0 and "biases" in f:
        f.pop("biases")
    mx.eval(f.parameters())
    o = 0
    for m, n in zip(mods, sizes):
        for nm in names:
            setattr(m, nm, getattr(f, nm)[o:o + n])
        o += n
    mx.eval([m.parameters() for m in mods])
    f.spans = [(sum(sizes[:i]), n) for i, n in enumerate(sizes)]
    f.freeze()
    return f


def prepare(tm):
    """Build the fused projections of a decoder stack `tm` (layers with linear_attn / self_attn and a dense mlp). Returns the number built."""
    if not (ENABLED and PROJ):
        return 0
    n = 0
    for layer in tm.layers:
        la, sa, mlp = getattr(layer, "linear_attn", None), getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        if la is not None and getattr(la, "_kr_fused", None) is None and all(hasattr(la, k) for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")):
            object.__setattr__(la, "_kr_fused", fuse([la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]))
            n += la._kr_fused is not None
        if sa is not None and getattr(sa, "_kr_fused", None) is None and all(hasattr(sa, k) for k in ("q_proj", "k_proj", "v_proj")):
            object.__setattr__(sa, "_kr_fused", fuse([sa.q_proj, sa.k_proj, sa.v_proj]))
            n += sa._kr_fused is not None
        if mlp is not None and getattr(mlp, "_kr_fused", None) is None and all(hasattr(mlp, k) for k in ("gate_proj", "up_proj", "down_proj")):
            object.__setattr__(mlp, "_kr_fused", fuse([mlp.gate_proj, mlp.up_proj]))
            n += mlp._kr_fused is not None
    return n


def _fused(m):
    return getattr(m, "_kr_fused", None) if (ENABLED and PROJ) else None

# ---------------------------------------------------------------- 2. residual add + RMSNorm


# Mirrors mx.fast.rms_norm for one row: 4 contiguous reads per thread, threadgroup 32 * ceil(H / 128), simd_sum then a sum of the simdgroup partials,
# rsqrt(mean + eps), out = w * T(x * inv). The summed stream h = T(x + r) is what the stock add writes.
_ADDNORM_SRC = """
    // fp16 stream: the sum saturates at the fp16 range (an inf would become a NaN row at the next norm); the sum of two halves rounds once, as the stock add does
    #define SUM(a, b) static_cast<T>(is_same<T, half>::value ? clamp((float)(a) + (float)(b), -65504.0f, 65504.0f) : (float)(a) + (float)(b))
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint ntg = threads_per_threadgroup.x;
    threadgroup float part[32];
    if (sg == 0) part[sl] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device T* xp = x + row * H;
    const device T* rp = r + row * H;
    float acc = 0.0f;
    for (uint i0 = lid * 4; i0 < H; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < H) {
                T hv = SUM(xp[i0 + i], rp[i0 + i]);
                hout[row * H + i0 + i] = hv;
                float f = (float)hv;
                acc += f * f;
            }
        }
    }
    acc = simd_sum(acc);
    if (sl == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = simd_sum(part[sl]);
        if (sl == 0) part[0] = metal::precise::rsqrt(t / (float)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = part[0];
    for (uint i0 = lid * 4; i0 < H; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < H) {
                T hv = SUM(xp[i0 + i], rp[i0 + i]);
                nout[row * H + i0 + i] = w[i0 + i] * static_cast<T>((float)hv * inv);
            }
        }
    }
"""
_k = {}
_eps = {}


def add_rms(x, r, norm):
    """(x + r, rms_norm(x + r)) in one launch; `norm` is an nn.RMSNorm. Falls back to the stock pair off-GPU or for odd dtypes."""
    if not (ENABLED and ADDNORM) or x.dtype not in (mx.float16, mx.bfloat16) or r.dtype != x.dtype or norm.weight.dtype != x.dtype or x.shape != r.shape:
        h = x + r
        return h, norm(h)
    if "an" not in _k:
        _k["an"] = mx.fast.metal_kernel(name="kyojin_add_rms", input_names=["x", "r", "w", "eps"], output_names=["hout", "nout"], source=_ADDNORM_SRC,
                                        header="#pragma METAL fp contract(off)\n")
    e = _eps.get(norm.eps)
    if e is None:
        e = _eps[norm.eps] = mx.array([norm.eps], dtype=mx.float32)
    H = x.shape[-1]
    rows = x.size // H
    tg = min(1024, 32 * (-(-H // 128)))
    h, n = _k["an"](inputs=[x, r, norm.weight, e], template=[("T", x.dtype), ("H", H)], grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1),
                    output_shapes=[x.shape, x.shape], output_dtypes=[x.dtype, x.dtype])
    return h, n


# The MoE block's weighted combine (sum_k score_k * y_k + sigmoid(shared gate) * shared) folded into the add + norm that follows it: the combined output is
# rounded to T (saturated on fp16) exactly as foreign_moe._combine writes it, then added to the stream; one launch and one pass less per MoE layer.
_ADDNORM_MOE_SRC = _ADDNORM_SRC.split("uint row")[0] + """
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint ntg = threads_per_threadgroup.x;
    threadgroup float part[32];
    if (sg == 0) part[sl] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device T* xp = x + row * H;
    float acc = 0.0f;
    for (uint i0 = lid * 4; i0 < H; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            uint c = i0 + i;
            if (c < H) {
                float a = 0.0f;
                for (uint k = 0; k < K; k++) a = fma(float(y[(row * K + k) * H + c]), scs[row * K + k], a);     // fused multiply-adds: the combine launch's contraction
                a = fma(sgs[row], float(sh[row * H + c]), a);
                T pv = static_cast<T>(is_same<T, half>::value ? clamp(a, -65504.0f, 65504.0f) : a);
                T hv = SUM(xp[c], pv);
                hout[row * H + c] = hv;
                float f = (float)hv;
                acc += f * f;
            }
        }
    }
    acc = simd_sum(acc);
    if (sl == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = simd_sum(part[sl]);
        if (sl == 0) part[0] = metal::precise::rsqrt(t / (float)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = part[0];
    for (uint i0 = lid * 4; i0 < H; i0 += ntg * 4) {
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < H) {
                T hv = hout[row * H + i0 + i];             // this thread wrote the element in the first pass
                nout[row * H + i0 + i] = w[i0 + i] * static_cast<T>((float)hv * inv);
            }
        }
    }
"""


def add_rms_moe(x, pend, norm):
    """add_rms(x, combine(pend), norm) in one launch; pend = foreign_moe.Pending. Same bits as the combine launch followed by add_rms."""
    y, scs, sgs, sh = pend.y, pend.scs, pend.sgs, pend.sh
    if not (ENABLED and ADDNORM) or x.dtype != mx.float16 or y.dtype != x.dtype or sh.dtype != x.dtype or norm.weight.dtype != x.dtype or x.ndim != 2 or x.shape[-1] != pend.H or x.shape[0] != pend.T:
        return add_rms(x, pend.value(), norm)
    if "anm" not in _k:
        _k["anm"] = mx.fast.metal_kernel(name="kyojin_add_rms_moe", input_names=["x", "y", "scs", "sgs", "sh", "w", "eps"], output_names=["hout", "nout"],
                                         source=_ADDNORM_MOE_SRC, header="#pragma METAL fp contract(off)\n")
    e = _eps.get(norm.eps)
    if e is None:
        e = _eps[norm.eps] = mx.array([norm.eps], dtype=mx.float32)
    H = pend.H
    rows = x.shape[0]
    tg = min(1024, 32 * (-(-H // 128)))
    h, n = _k["anm"](inputs=[x, y, scs, sgs, sh, norm.weight, e], template=[("T", x.dtype), ("H", H), ("K", pend.K)], grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1),
                     output_shapes=[x.shape, x.shape], output_dtypes=[x.dtype, x.dtype])
    return h, n


def norm_of(x, pend, norm):
    """Residual stream and its norm: pend is the not-yet-added sub-layer output (None = x is already the stream; a foreign_moe.Pending = its combine is folded in)."""
    if pend is None:
        return x, norm(x)
    if hasattr(pend, "value"):
        x2 = x.reshape(-1, x.shape[-1])
        h, n = add_rms_moe(x2, pend, norm)
        return h.reshape(x.shape), n.reshape(x.shape)
    return add_rms(x, pend, norm)

# ---------------------------------------------------------------- bodies used by step_regions (one row)


def mlp(m, x, defer=False):
    """The MLP output; defer=True lets a fused MoE block hand back its combine as a Pending that norm_of folds into the next add + norm."""
    if defer and MOEFOLD and getattr(m, "_foreign_moe", False):
        return m._deferred_call(x)
    f = _fused(m)
    if f is None:
        return m(x)
    from mlx_lm.models.activations import swiglu
    g, u = f.parts(x)
    return m.down_proj(swiglu(g, u))


def _gdn_dec_ok(self, x, st):
    return (ENABLED and GDN and x.dtype in (mx.float16, mx.bfloat16) and self.sharding_group is None and self.head_k_dim == 128 and self.head_v_dim == 128
            and self.conv_kernel_size == 4 and self.dt_bias.dtype == x.dtype and self.norm.weight.size == 128 and st[1].dtype == mx.float32)


def gdn(self, x, st):
    """Gated-delta mixer for one row; st = (conv state, recurrent state). Returns (out, new states)."""
    B, S, _ = x.shape
    f = _fused(self)
    if f is not None:
        qkv, z, b, a = f.parts(x)
    else:
        from . import gdn_pf as _gp
        qkv, z = self.in_proj_qkv(x), self.in_proj_z(x)
        b, a = _gp.proj_ba(self, x)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    if _gdn_dec_ok(self, x, st):
        from mlx_lm.models.gated_delta import gated_delta_kernel

        from . import gdn_pf
        if gdn_pf._fuse_on(S):
            out, cn, state, _ = gdn_pf.fused_mixer(self, qkv, z, a, b, st[0], st[1])
            return out, (cn, state)
        q, k, v, cn, g, beta = gdn_pf.pre(qkv, st[0], self.conv1d.weight, a, b, self.A_log, self.dt_bias, self.num_k_heads, self.num_v_heads)
        y, state = gated_delta_kernel(q, k, v, g, beta, st[1], None)
        return gdn_pf.norm_out(self, y, z, x.dtype), (cn, state)
    from mlx_lm.models import qwen3_5 as Q5
    conv_input = mx.concatenate([st[0], qkv], axis=1)
    cn = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                                                         [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                                                         [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
    q, k = Q5.normalize_qk(q, k, inv_scale=self.head_k_dim ** -0.5, eps=1e-6)
    out, state = Q5.gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, st[1], None, use_kernel=True)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1)), (cn, state)


def qkv(a, h):
    f = _fused(a)
    if f is None:
        return a.q_proj(h), a.k_proj(h), a.v_proj(h)
    return tuple(f.parts(h))

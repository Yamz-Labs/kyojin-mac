# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x prompt-path mHC (hyper-connection) glue: the two seams of `HyperConnection.apply_branch` for prompt chunks (S > 8 rows, batch 1).

Stock (mlx_vlm deepseek_v4.hyper_connection, G0 profile at 8K rows: attn_hc 0.57 s + ffn_hc 0.48 s over 45 layers, 11-13 ms per call):
  collapse   y = x.astype(fp32) [S, 4, 4096] (536 MB written), z = rms_norm(y.flatten) (536 MB written), mixes = z @ fn.T as 32 tiled fp32 matmuls
             (24 columns: far from the matmul sweet spot), then the fused sinkhorn + collapse kernel;
  expand     `_hc_expand_op`: post * x + comb^T @ residual through fp32 [S, 4, 4096] temporaries.
Here
  mix        ONE kernel: mixes[s, :] = rsqrt(mean(x_s^2) + eps) * (x_s @ fn.T), x read in fp16 (exact in fp32), the product on simdgroup_matrix half MMAs with fp32
             accumulate; fn is split into fn_hi = half(fn) and fn_lo = half((fn - fn_hi) * 2048) so the weights keep ~fp32 precision (error ~1e-6 relative). The
             sinkhorn + collapse kernel of the stock path then runs unchanged on these mixes.
  expand     the DS4.1 prompt kernel (pf_glue.expand: one pass over x and the residual, fp32 inside, saturating at the fp16 range like the decode kernels).
Switches: KYOJIN_GLM_PF_HC (mix + expand, flag glm_pf_hc), KYOJIN_GLM_PF_HC_EXPAND=0 / KYOJIN_GLM_PF_HC_MIX=0 switch one seam off."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD
from . import hc_pf_fuse as HF

ENABLED = os.environ.get("KYOJIN_GLM_PF_HC", "0") != "0"
MIX = os.environ.get("KYOJIN_GLM_PF_HC_MIX", "1") != "0"
EXPAND = os.environ.get("KYOJIN_GLM_PF_HC_EXPAND", "1") != "0"
VIEW = os.environ.get("KYOJIN_HC_MIX_VIEW", "0") != "0"                    # [1, S, 4, D] -> [S, 4 D] as a reshape view (0 = `x[0]`, a Gather: a 268 MB copy per call at 8K rows)
NSG = int(os.environ.get("KYOJIN_GLM_PF_HC_NSG", "4"))                      # simdgroups (8 rows each) per threadgroup
RT = int(os.environ.get("KYOJIN_GLM_PF_HC_RT", "1"))                       # 8-row tiles per simdgroup
LO = os.environ.get("KYOJIN_GLM_PF_HC_LO", "1") != "0"                     # fn_lo pass (weights at ~fp32; 0 = fn_hi only, the fp16 weights)
MIN_ROWS = 16                                                             # S > 8 is the stock prompt path; the tile scheme below needs S >= 8
STATS = {"mix": 0, "expand": 0}

_HEADER = "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

# threadgroup = NSG simdgroups (default 4), 8 NSG rows (NSG tiles of 8) x 24 mix columns, K walked in chunks of 64: the x rows (32 x 64) and fn_hi / fn_lo (24 x 64) are staged through threadgroup
# memory with 16-byte coalesced loads, simdgroup g multiplies the 8 rows of tile g against all 24 columns. Tile t covers rows tb = min(8 t, S - 8) .. tb + 7 (the last tile
# overlaps its neighbour instead of reading past S; both write the same values). The row's sum of squares is taken from the staged chunk (thread = (row, quarter of 16 k)).
_MIX_SRC = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint S = prm[0];
    const float eps = epsv[0];
    constexpr uint NT = 32 * NSG;                           // threads
    constexpr uint NR = 8 * NSG * RT;                       // rows of the threadgroup
    threadgroup uint4 As4[NR * 9];                          // NR rows x 64 k (half), row stride 72 halves
    threadgroup uint4 Bh4[24 * 9], Bl4[24 * 9];             // fn_hi / fn_lo [24, 64 k]
    threadgroup float redh[NR * 24], redl[NR * 24];
    uint tb[NSG * RT];
    for (uint r = 0; r < NSG * RT; ++r) { tb[r] = min((tg * NSG * RT + r) * 8, S - 8); }
    simdgroup_matrix<float, 8, 8> ah[RT][3], al[RT][3];
    for (uint r = 0; r < RT; ++r) { for (uint n = 0; n < 3; ++n) { ah[r][n] = simdgroup_matrix<float, 8, 8>(0.0f); al[r][n] = simdgroup_matrix<float, 8, 8>(0.0f); } }
    const uint qd = NT / NR;                                // threads per row for the sum of squares (NT / NR = 4 / RT: 1, 2 or 4)
    const uint lr = tid / qd, q = tid % qd;
    const uint grow = tb[lr >> 3] + (lr & 7);
    threadgroup const half* Ah = (threadgroup const half*)As4;
    threadgroup const half* Bhh = (threadgroup const half*)Bh4;
    threadgroup const half* Blh = (threadgroup const half*)Bl4;
    float ss = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += 64) {
        for (uint idx = tid; idx < NR * 8; idx += NT) {
            const uint r = idx >> 3, c8 = idx & 7;
            const uint gr = tb[r >> 3] + (r & 7);
            As4[r * 9 + c8] = *((const device uint4*)(x + ulong(gr) * K + k0) + c8);
        }
        for (uint idx = tid; idx < 192; idx += NT) {
            const uint r = idx >> 3, c8 = idx & 7;
            Bh4[r * 9 + c8] = *((const device uint4*)(fnh + ulong(r) * K + k0) + c8);
            if (LO) { Bl4[r * 9 + c8] = *((const device uint4*)(fnl + ulong(r) * K + k0) + c8); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = q * (8 / qd); c < (q + 1) * (8 / qd); ++c) {         // this thread's share of the row's 64 elements, 8 per uint4
            const uint4 u = As4[lr * 9 + c];
            const half2* h = (const thread half2*)&u;
            for (uint j = 0; j < 4; ++j) { const float2 a = float2(h[j]); ss += a.x * a.x + a.y * a.y; }
        }
        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_matrix<half, 8, 8> A[RT], Bh[3], Bl[3];
            for (uint n = 0; n < 3; ++n) {
                simdgroup_load(Bh[n], Bhh + n * 8 * 72 + kk, 72, ulong2(0, 0), true);
                if (LO) { simdgroup_load(Bl[n], Blh + n * 8 * 72 + kk, 72, ulong2(0, 0), true); }
            }
            for (uint r = 0; r < RT; ++r) {
                simdgroup_load(A[r], Ah + (sg * RT + r) * 8 * 72 + kk, 72);
                for (uint n = 0; n < 3; ++n) {
                    simdgroup_multiply_accumulate(ah[r][n], A[r], Bh[n], ah[r][n]);
                    if (LO) { simdgroup_multiply_accumulate(al[r][n], A[r], Bl[n], al[r][n]); }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < RT; ++r) {
        for (uint n = 0; n < 3; ++n) {
            simdgroup_store(ah[r][n], redh + (sg * RT + r) * 8 * 24 + n * 8, 24);
            if (LO) { simdgroup_store(al[r][n], redl + (sg * RT + r) * 8 * 24 + n * 8, 24); }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = 1; o < qd; o <<= 1) { ss += simd_shuffle_xor(ss, o); }
    const float inv = metal::precise::rsqrt(ss / float(K) + eps);
    for (uint c = q * (24 / qd); c < (q + 1) * (24 / qd); ++c) {
        mixes[ulong(grow) * 24 + c] = (redh[lr * 24 + c] + (LO ? redl[lr * 24 + c] * (1.0f / 2048.0f) : 0.0f)) * inv;
    }
"""

_kern = None
_split: dict = {}


def _fn_split(fn):
    """(fn_hi, fn_lo) half [24, K]: fn = hi + lo / 2048 to ~2^-22 relative (cached per weight array)."""
    key = id(fn)
    e = _split.get(key)
    if e is None or e[0] is not fn:
        hi = fn.astype(mx.float16)
        lo = ((fn - hi.astype(mx.float32)) * 2048.0).astype(mx.float16)
        mx.eval(hi, lo)
        e = _split[key] = (fn, hi, lo)
    return e[1], e[2]


def mix_why(x, fn):
    """None when the prompt mix kernel serves x [S, K] / fn, else the reason (flag audit key; the guards of `mix_ok`, same order)."""
    if x.ndim != 2:
        return "x_ndim"
    if x.dtype != mx.float16:
        return "x_dtype=" + str(x.dtype).split(".")[-1]
    if x.shape[0] < MIN_ROWS:
        return f"rows<{MIN_ROWS}"
    if fn.ndim != 2 or fn.shape[0] != 24 or fn.shape[1] != x.shape[1] or x.shape[1] % 128 != 0:
        return "fn_shape"
    if fn.dtype not in (mx.float32, mx.float16):                  # fp16 = the served loader's storage: its values ARE the weights, fn_hi = fn exactly and fn_lo = 0 (no lo pass)
        return "fn_dtype=" + str(fn.dtype).split(".")[-1]
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "no_gpu"
    return None


def mix_ok(x, fn) -> bool:
    return mix_why(x, fn) is None


def mix(x, fn, eps: float):
    """x [S, K] fp16 (K = hc_mult * hidden), fn [24, K] fp32 or fp16 -> [S, 24] fp32 = rms_norm(x.f32, no weight, eps) @ fn.T."""
    global _kern
    S, K = x.shape
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="kyojin_glm_pf_hcmix", input_names=["x", "fnh", "fnl", "prm", "epsv"], output_names=["mixes"], source=_MIX_SRC, header=_HEADER)
    lo_on = LO
    if fn.dtype == mx.float16:                       # exact as is: the stock matmul promotes this fn to fp32 (same values), so the lo half is identically 0
        hi, lo, lo_on = fn, fn, False
    else:
        hi, lo = _fn_split(fn)
    STATS["mix"] += 1
    nt = -(-S // (8 * NSG * RT))
    return _kern(inputs=[x, hi, lo, mx.array([S], dtype=mx.uint32), mx.array([eps], dtype=mx.float32)], template=[("K", K), ("NSG", NSG), ("RT", RT), ("LO", int(lo_on))], grid=(nt * 32 * NSG, 1, 1), threadgroup=(32 * NSG, 1, 1),
                 output_shapes=[(S, 24)], output_dtypes=[mx.float32])[0]


TAG = "_kyojin_glm_pf"                  # set on the HyperConnection instances of the GLM model that installed the seams: the patched class methods act on those only


def _mine(hc) -> bool:
    return getattr(hc, TAG, False) is True


def _impl_call(orig, hcmod):
    def __call__(self, x):
        if not _mine(self):                                     # another model's HyperConnection (the class is shared process-wide): stock
            return orig(self, x)
        if ENABLED and MIX and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > hcmod.DECODE_BLOCK_SIZE:        # prompt chunks only: decode / verify rows are glm_hc's
            why = ("hc_mult" if self.hc_mult != 4 else "training" if self.training else "no_hc_kernel" if getattr(hcmod, "_hc_kernel", None) is None else None)
            if why is None:
                S, H, D = x.shape[1], x.shape[2], x.shape[3]
                xf = x.reshape(S, H * D) if VIEW else x[0].reshape(S, H * D)
                why = mix_why(xf, self.fn)
                if why is None:
                    FD.fire("pf_hc_mix")
                    mixes = mix(xf, self.fn, self.norm_eps)[None]
                    return hcmod._hc_kernel(x, None, mixes, self.scale, self.base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)
            FD.fb("pf_hc_mix", why)
        elif FD.AUDIT and ENABLED and MIX and x.ndim == 4 and x.shape[1] > hcmod.DECODE_BLOCK_SIZE:
            FD.fb("pf_hc_mix", f"batch={x.shape[0]}")
        return orig(self, x)
    return __call__


def _impl_apply(orig, hcmod):
    """apply_branch for prompt chunks: mix kernel, then sinkhorn + collapse + the sub-block's rms_norm in ONE kernel (hc_pf_fuse.collapse_norm), the branch, the expand seam."""
    def apply_branch(self, x, norm, branch, *args, **kwargs):
        if (_mine(self) and ENABLED and MIX and HF.NORM and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > hcmod.DECODE_BLOCK_SIZE and not self.training and self.hc_mult == 4
                and getattr(hcmod, "_hc_kernel", None) is not None):
            S, H, D = x.shape[1], x.shape[2], x.shape[3]
            xf = x.reshape(S, H * D)
            why = mix_why(xf, self.fn) or HF.norm_why(x, norm, self.hc_mult, D)
            if why is None:
                FD.fire("pf_hc_mix")
                FD.fire("pf_hc_norm")
                mixes = mix(xf, self.fn, self.norm_eps)[None]
                normed, post, comb = HF.collapse_norm(x, mixes, self.scale, self.base, norm.weight, norm.eps, self.hc_eps, self.sinkhorn_iters)
                output = branch(normed, *args, **kwargs)
                if isinstance(output, tuple):
                    return (hcmod.hc_expand(output[0], x, post, comb, use_kernel=True), *output[1:])
                return hcmod.hc_expand(output, x, post, comb, use_kernel=True)
            FD.fb("pf_hc_norm", why)
        return orig(self, x, norm, branch, *args, **kwargs)
    return apply_branch


def _expand_why(x, residual, post, comb, use_kernel, hcmod):
    if not use_kernel:
        return "use_kernel=False"
    if x.ndim != 3 or x.shape[0] != 1:
        return f"batch={x.shape[0]}" if x.ndim == 3 else "x_ndim"
    if residual.ndim != 4 or residual.shape[2] != 4:
        return "residual_shape"
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return "post_comb_dtype"
    if x.dtype not in (mx.float16, mx.bfloat16) or x.shape[2] % 8 != 0:
        return "x_dtype_hidden"
    if mx.default_device() != mx.gpu:
        return "no_gpu"
    return None


def _impl_expand(orig, hcmod):
    def hc_expand(x, residual, post, comb, *, use_kernel=True):
        if ENABLED and EXPAND and x.ndim >= 2 and x.shape[1] > hcmod.DECODE_BLOCK_SIZE:                                       # prompt rows only
            why = _expand_why(x, residual, post, comb, use_kernel, hcmod)
            if why is None:
                from . import pf_glue as PG
                STATS["expand"] += 1
                FD.fire("pf_hc_expand")
                return PG.expand(x, residual, post, comb)
            FD.fb("pf_hc_expand", why)
        return orig(x, residual, post, comb, use_kernel=use_kernel)
    return hc_expand


def _tag_instances(lm, hc) -> int:
    """Mark the HyperConnection instances of this GLM language model whose config the seams serve (hc_mult 4; the per-call norm check is HF.norm_why, which refuses a norm that is not MLX's RMSNorm)."""
    n = 0
    for m in lm.modules():
        if isinstance(m, hc.HyperConnection) and getattr(m, "hc_mult", None) == 4:
            object.__setattr__(m, TAG, True)
            n += 1
    return n


def install(model) -> int:
    """Patch HyperConnection.__call__ / apply_branch (class level, once per process, but acting only on the instances `_tag_instances` marked: a DeepSeek V4 or any other model that
    shares mlx_vlm's class runs the stock code) and hyper_connection.hc_expand (a free function: the same op for every hc_mult 4 model, shape / dtype guarded). Returns the number of seams installed."""
    if not ENABLED:
        return 0
    lm = getattr(model, "language_model", None)
    if not type(lm).__module__.startswith("mlx_vlm.models.glm5_next"):
        return 0
    from mlx_vlm.models.deepseek_v4 import hyper_connection as hc
    _tag_instances(lm, hc)
    if getattr(hc, "_kyojin_glm_pf", False):
        return 2
    n = 0
    if MIX:
        hc.HyperConnection.__call__ = _impl_call(hc.HyperConnection.__call__, hc)
        n += 1
    if EXPAND:
        hc.hc_expand = _impl_expand(hc.hc_expand, hc)
        n += 1
    if MIX:
        hc.HyperConnection.apply_branch = _impl_apply(hc.HyperConnection.apply_branch, hc)
        n += 1
    hc._kyojin_glm_pf = True
    return n

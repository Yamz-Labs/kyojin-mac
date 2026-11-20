# SPDX-License-Identifier: Apache-2.0
"""Hy4 (any absorbed-MLA skeleton) prompt attention over per-row key lists as ONE fused flash-style NAX kernel: scores (latent 512 + rope 64), online exp2 softmax with the
attention sink in the denominator, P x V, no [heads, rows, keys] score matrix in memory.

The Metal text is `glm_dsa_fused._SOURCE` (adapted from mlx-serve, MIT, itself from oMLX, Apache-2.0; see THIRD_PARTY_NOTICES.md) with the rope branch (`MSV_DSA_ROPE`) switched on and
four edits applied by string replacement (each asserted, so a change of the base text fails loudly instead of silently):
  1. the key row is read from ONE cache buffer of row stride RS (the stock cache row is [latent | k_pe | kpool value | kpool gate]); the roped key sits at column D of the same row;
  2. the sink of each head joins the softmax denominator (m / l / O rescaled once at the end), as `sc = concat(sc, sink)` does in the stock chain;
  3. nothing else: ids, causal test (`id > q_abs` dead), dead-tile skip, hi / lo P pieces and the fp32 accumulation are the base kernel's.
Use: the 12 indexer layers of Hy4 (32 heads, top_k 2048 keys per row from `foreign_skeleton_kpool.gather_slots`). The windowed layers (64 heads, 512 keys shared by neighbouring rows)
stay on the blocked fused-chain path: one row per threadgroup re-reads the key tile for every row, which only pays when the lists differ per row.

Flag KYOJIN_HY4_ATTN_FLASH (read per call, default ON since 19/11: a16 cell, NLL inside the null band, +12 % tok/s; needs the fp16 attention path, flag attn_fp16)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import glm_dsa_fused as GD

D, DR = 512, 64
_ON = "KYOJIN_HY4_ATTN_FLASH"


def on() -> bool:
    return os.environ.get(_ON, "1") not in ("0", "")


_ON_SWA = "KYOJIN_HY4_ATTN_FLASH_SWA"


def swa_on() -> bool:
    """Flag KYOJIN_HY4_ATTN_FLASH_SWA (read per call, default ON since 19/11, a16 cell): the windowed layers (64 heads, window keys per row) run the same kernel with per-row key lists = their window."""
    return os.environ.get(_ON_SWA, "1") not in ("0", "")


def window_ids(L: int, S: int, w: int):
    """ids [1, L, w] int32: row l sits at slot S - L + l and sees slots (S - L + l - w, S - L + l] (create_causal_mask window_size semantics); slots < 0 are -1 (dead)."""
    ids = (S - L) + mx.arange(L)[:, None] - (w - 1) + mx.arange(w)[None, :]
    return mx.where(ids >= 0, ids, -1).astype(mx.int32)[None]


def _edit(src: str, old: str, new: str, n: int = 1) -> str:
    assert src.count(old) == n, f"glm_dsa_fused source changed: {old[:60]!r} found {src.count(old)} times"
    return src.replace(old, new)


_SINK_EPI = r"""    float fsink[2] = {1.0f, 1.0f};
    if (HAS_SINK != 0) {
        for (short i = 0; i < 2; ++i) {
            const float s2 = float(sinks[head0 + fm + i * 8]) * 1.44269504088896341f;
            const float mf = max(m_run[i], s2);
            const float f = fast::exp2(m_run[i] - mf);
            l_run[i] = l_run[i] * f + fast::exp2(s2 - mf);
            fsink[i] = f;
        }
        for (short e = 0; e < 16; ++e) {
            const float f = fsink[(e >> 2) & 1];
            o0[e] *= f;
            o1[e] *= f;
            o2[e] *= f;
            o3[e] *= f;
        }
    }
"""


def source() -> str:
    s = GD._SOURCE
    s = "#define MSV_DSA_ROPE\n" + s
    s = _edit(s, "    constexpr int DR = 64;\n", "    constexpr int DR = 64;\n    const int RS = params[4];\n    const int HAS_SINK = params[5];\n")
    s = _edit(s, "kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;", "kr[tn][i] = kv + ulong(max(kp, 0)) * RS + fn;")
    s = _edit(s, "pr[tn][i] = pe + ulong(max(kp, 0)) * DR + fn;", "pr[tn][i] = kv + ulong(max(kp, 0)) * RS + D + fn;")
    s = _edit(s, "kv + ulong(max(kp0, 0)) * D + j4 * 128 + fn;", "kv + ulong(max(kp0, 0)) * RS + j4 * 128 + fn;")
    s = _edit(s, "kv + ulong(max(kp1, 0)) * D + j4 * 128 + fn;", "kv + ulong(max(kp1, 0)) * RS + j4 * 128 + fn;")
    s = _edit(s, "    for (short i = 0; i < 2; ++i) {\n        device T* orow", _SINK_EPI + "    for (short i = 0; i < 2; ++i) {\n        device T* orow")
    return s


_kern = None


def supported(qa, qpe, rows, ids, sinks) -> bool:
    """qa [1, H, T, 512] fp16, qpe [1, H, T, 64] fp16, rows [1, 1, N, RS] fp16 (RS >= 576), ids [1, T, W] int32 (-1 = none); the queries sit at the LAST T rows of the buffer."""
    return (GD.ENABLED and qa.ndim == 4 and qa.shape[0] == 1 and qa.shape[1] % 32 == 0 and qa.shape[3] == D and qa.dtype == mx.float16 and qpe.dtype == mx.float16
            and tuple(qpe.shape) == (1, qa.shape[1], qa.shape[2], DR) and rows.dtype == mx.float16 and rows.ndim == 4 and rows.shape[:2] == (1, 1) and rows.shape[3] >= D + DR
            and rows.shape[2] >= qa.shape[2] and ids.ndim == 3 and ids.shape[0] == 1 and ids.shape[1] == qa.shape[2] and ids.shape[2] >= 1 and ids.dtype == mx.int32
            and (sinks is None or (sinks.shape == (qa.shape[1],) and sinks.dtype == mx.float32)))


def attend(qa, qpe, rows, ids, scale: float, sinks=None):
    """softmax over (scale * (qa . latent[ids] + qpe . k_pe[ids]) [, sink]) @ latent[ids] -> [1, H, T, 512] fp16 (before v_b)."""
    global _kern
    _, H, T, _ = qa.shape
    N, RS, W = rows.shape[2], rows.shape[3], ids.shape[2]
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="kyojin_hy4_mla_flash", input_names=["q", "kv", "idx", "params", "scale", "qpe", "sinks"], output_names=["out"], source=source(), header=GD._HEADER)
    sk = sinks if sinks is not None else mx.zeros((1,), dtype=mx.float32)
    out = _kern(inputs=[mx.contiguous(qa[0]), rows.reshape(N, RS), ids[0], mx.array([T, N, W, N - T, RS, 0 if sinks is None else 1], dtype=mx.int32),
                        mx.array([scale], dtype=mx.float32), mx.contiguous(qpe[0]), sk],
                template=[("T", mx.float16)], grid=(T * 256, H // 32, 1), threadgroup=(256, 1, 1), output_shapes=[(H, T, D)], output_dtypes=[mx.float16])[0]
    return out[None]


def reference(qa, qpe, rows, ids, scale: float, sinks=None):
    """The same result from mx ops (fp32): used by the tests and as the fallback shape check."""
    _, H, T, _ = qa.shape
    N = rows.shape[2]
    r = rows[0, 0].astype(mx.float32)
    i0 = mx.maximum(ids[0], 0)                                                            # [T, W]
    kk = r[:, :D][i0]                                                                      # [T, W, 512]
    kp = r[:, D:D + DR][i0]                                                                # [T, W, 64]
    qpos = (N - T) + mx.arange(T)[:, None]
    ok = (ids[0] >= 0) & (ids[0] < N) & (ids[0] <= qpos)                                   # [T, W]
    sc = (mx.einsum("htd,twd->htw", qa[0].astype(mx.float32), kk) + mx.einsum("htd,twd->htw", qpe[0].astype(mx.float32), kp)) * scale
    sc = mx.where(ok[None], sc, -mx.inf)
    if sinks is not None:
        sc = mx.concatenate([sc, mx.broadcast_to(sinks[:, None, None], (H, T, 1))], axis=-1)
    p = mx.softmax(sc, axis=-1)[..., :ids.shape[2]]
    return mx.einsum("htw,twd->htd", p, kk)[None]

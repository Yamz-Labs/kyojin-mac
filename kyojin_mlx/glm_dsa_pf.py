# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x DSA prefill attention in latent space on the DS4.1 NAX gathered-key kernels (attn_prefill K1 / K3, online softmax).

GLM MLA has no rope (mla_use_nope): score(q, key j) = q_nope . W_k(latent_j) = embed_q(q) . latent_j and the value is W_v(latent_j), so with
q_lat = embed_q(q) [1, H, T, 512] the attention is MQA over ONE shared table (the latent, K = V, 512 wide) and the head projection of the output
moves after the sum: out = unembed_out(sum_j p_j latent_j). That is the shape of DS4.1's sparse pooled attention (64 heads, D = 512, one key list per
query), so the same kernels run it:

  stock   K = embed_q(latent) and V = unembed_out(latent) projected for EVERY head (2 x N x 64 x 256 values; above 32768 tokens re-projected from the
          whole latent at every chunk), then a SIMT kernel (mlx_vlm indexed_sparse_attention) reading K / V rows per (query, head);
  here    no projection of the table, the keys of a query (top-k + tail ids, -1 = none) are gathered once for all heads into NAX matmuls (fp32 accumulate),
          cost per layer independent of the context length N (the table is read by id only).

Same selection (the indexer's top-k ids), same softmax scale; no attention sink (the sink is -1e30). Rounding differs (fp16 q_lat / probabilities, fp32
accumulate): relative error vs the stock path about 1e-3.

Switch: KYOJIN_GLM_DSA_PF (flag glm_dsa_pf). Chunk of queries per launch group: KYOJIN_GLM_DSA_PF_CHUNK (default 256)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import archs as _A
from . import attn_prefill as AP
from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_DSA_PF", "0") != "0"
CHUNK = int(os.environ.get("KYOJIN_GLM_DSA_PF_CHUNK", "256"))
MIN_ROWS = 64                                                          # chunks of fewer rows (decode, verify, MTP) keep the stock code
NEG = -1e30

_dummy: dict = {}


def why_not(q, latent, topk):
    """None when the latent-space prompt attention serves this call, else the reason (flag audit key; the guards of `supported`, same order)."""
    B, H, T, _ = q.shape
    D = latent.shape[-1]
    if not ENABLED:
        return "flag_off"
    if T < MIN_ROWS:
        return f"rows<{MIN_ROWS}"                          # decode / verify call: checked first so the reasons below are prompt chunks only
    if B != 1:
        return f"batch={B}"
    if H not in (32, 64):
        return "heads"
    if latent.ndim != 4 or latent.shape[1] != 1 or D % 128 != 0:
        return "latent_shape"
    if topk.ndim != 3 or topk.shape[0] != 1 or topk.shape[1] != T:
        return "topk_shape"
    if q.dtype not in (mx.float16, mx.bfloat16):
        return "q_dtype=" + str(q.dtype).split(".")[-1]
    if not AP.available():
        return "no_nax"
    return None


def supported(q, latent, topk) -> bool:
    return why_not(q, latent, topk) is None


def key_ids(topk, n: int):
    """[T, NK] int32 ids into the latent table (-1 = none), NK padded to a multiple of 32."""
    t = topk[0]
    ids = mx.where((t >= 0) & (t < n), t, -1).astype(mx.int32)
    pad = -ids.shape[-1] % 32
    return mx.concatenate([ids, mx.full((ids.shape[0], pad), -1, dtype=mx.int32)], axis=-1) if pad else ids


def latent_attention(q_lat, latent, topk, scale: float, chunk: int | None = None):
    """q_lat [1, H, T, D] fp16, latent [1, 1, N, D] fp16, topk [1, T, K] (-1 = none) -> [1, H, T, D] fp16 = softmax(scale * q_lat . latent[ids]) @ latent[ids]."""
    _, H, T, D = q_lat.shape
    N = latent.shape[2]
    if chunk is None:                                                              # chunk = the two-kernel path explicitly (bench / tests)
        from . import glm_dsa_fused as GF
        tk = topk if topk.dtype == mx.int32 else topk.astype(mx.int32)
        if GF.supported(q_lat, latent, tk):
            return GF.latent_attention(q_lat, latent, tk, scale)
    ids = key_ids(topk, N)
    NK = ids.shape[-1]
    q0 = mx.contiguous(q_lat[0])
    tab = [latent[0, 0], _dummy.setdefault(D, mx.zeros((1, D), dtype=mx.float16))]
    sk = mx.full((H,), NEG, dtype=mx.float32)
    k1, k3 = AP._kernels_online(H, D, NK, scale, True, False)
    ntl = -(-NK // 128)
    cs = chunk or CHUNK
    outs = []
    for a in range(0, T, cs):
        tc = min(cs, T - a)
        ic = ids[a:a + tc]
        prm = mx.array([a, T, N], dtype=mx.uint32)
        P, MX, LX = k1(inputs=[q0] + tab + [ic, prm], grid=(ntl * 128, tc, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(tc * H, NK), (tc * H, ntl), (tc * H, ntl)], output_dtypes=[mx.float16, mx.float32, mx.float32])
        prm3 = mx.array([tc, ntl, N], dtype=mx.uint32)
        outs.append(k3(inputs=[P, MX, LX, sk] + tab + [ic, prm3], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0])
    return (outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1))[None]


def _impl_attend(orig):
    def _attend(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False):
        why = "last_only" if last_only else ("training" if self.training else why_not(q, latent, topk))
        if why is not None:
            FD.fb("dsa_pf", why)                          # "rows<64" = a decode / verify call, not a prompt chunk: the audit table reads it as not applicable
            return orig(self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only)
        FD.fire("dsa_pf")
        dt = q.dtype
        q_lat = self.embed_q(q).astype(mx.float16)                                  # [1, H, T, 512]
        tab = latent if latent.dtype == mx.float16 else latent.astype(mx.float16)
        o = latent_attention(q_lat, tab, topk, self.scale)
        out = self.unembed_out(o.astype(dt))                                        # [1, H, T, v_head_dim]
        batch, _, length, _ = q.shape
        return out.transpose(0, 2, 1, 3).reshape(batch, length, -1), topk
    return _attend


def install(model) -> int:
    """Replace `_attend` of Glm5NextAttention (once per process, class level: rows.enable wraps it afterwards). 1 when installed."""
    if not ENABLED:
        return 0
    try:
        L = _A.lang(_A.GLM5)
    except Exception:
        return 0
    cls = L.Glm5NextAttention
    layers = getattr(getattr(model, "language_model", None), "layers", None) or []
    if not any(isinstance(getattr(ly, "self_attn", None), cls) for ly in layers):
        return 0
    if getattr(cls, "_kyojin_dsa_pf", False):
        return 1
    cls._attend = _impl_attend(cls._attend)
    cls._kyojin_dsa_pf = True
    return 1

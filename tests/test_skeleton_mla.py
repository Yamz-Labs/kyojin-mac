# SPDX-License-Identifier: Apache-2.0
"""N1 MLA projections against the served GLM attention classes on synthetic tensors (CPU), then N3 (kpool) wired behind N1 at 1 and 4 rows.

References, given the same weights and inputs:
  rope form  mlx-vlm deepseek_v32 DeepseekV32Attention: the attention class GLM-5.x (glm_moe_dsa) is served with; indexer bypassed (topk None)
  gate+sink  mlx-vlm hy_v4 HyV4Attention (DeepseekV32Attention + element gate + learned sinks), indexer bypassed
  nope form  mlx-vlm glm5_next Glm5NextAttention: the class the GLM-5.3-Flash GGUF is served with (rope 0, fused qkv_a), shared-indexer layer fed
             the full causal top-k
Projections (q_lora residual, q_nope, roped q_pe, latent, roped k_pe) run the same ops in the same order: mx.array_equal.
Attention outputs: max abs diff at fp32 tolerance, because the references attend with mx.fast.scaled_dot_product_attention (fused kernel, its own
reduction order; glm5_next also one query at a time over gathered keys) and N1 with plain ops (scores, mask, sink column, precise softmax)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import foreign_skeleton_mla as MLA

D, NH, QL, KV, NOPE, ROPE, V = 64, 4, 48, 32, 24, 8, 16
TOL = 2e-5


def _fill(mod, rng):
    """Every parameter of `mod` <- a small normal tensor (norm weights around 1)."""
    flat = {}
    for k, a in nn.utils.tree_flatten(mod.parameters()):
        s = rng.standard_normal(a.shape).astype(np.float32)
        flat[k] = mx.array(1.0 + 0.1 * s if ("norm" in k and a.ndim == 1) else 0.2 * s)
    mod.update(nn.utils.tree_unflatten(list(flat.items())))
    return mod


def _ours(rope=ROPE, rng=None, **kw):
    return _fill(MLA.MlaAttention(D, NH, QL, KV, NOPE, rope, V, eps=1e-6, base=10000.0, **kw), rng)


def _copy(ref, m, fused=False):
    if fused:
        ref.qkv_a_proj.weight = mx.concatenate([m.q_a_proj.weight, m.kv_a_proj.weight], axis=0)
    else:
        ref.q_a_proj.weight, ref.kv_a_proj_with_mqa.weight = m.q_a_proj.weight, m.kv_a_proj.weight
    ref.q_a_layernorm.weight, ref.q_b_proj.weight, ref.kv_a_layernorm.weight = m.q_a_norm.weight, m.q_b_proj.weight, m.kv_a_norm.weight
    ref.embed_q.weight, ref.unembed_out.weight, ref.o_proj.weight = m.k_b, m.v_b, m.o_proj.weight


def _eq(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and mx.array_equal(a, b).item(), float(mx.max(mx.abs(a - b)).item())


def _close(a, b, tol=TOL):
    mx.eval(a, b)
    assert a.shape == b.shape
    d = float(mx.max(mx.abs(a - b)).item())
    assert d <= tol * max(1.0, float(mx.max(mx.abs(b)).item())), d


def _v32_cfg():
    return SimpleNamespace(hidden_size=D, num_attention_heads=NH, max_position_embeddings=4096, rope_theta=10000.0, q_lora_rank=QL,
                           qk_rope_head_dim=ROPE, kv_lora_rank=KV, v_head_dim=V, qk_nope_head_dim=NOPE, attention_bias=False, rope_scaling=None,
                           indexer_types=["shared"])


@pytest.fixture
def no_indexer(monkeypatch):
    from mlx_vlm.models.deepseek_v32 import language as v32
    from mlx_vlm.models.hy_v4 import language as _h4          # noqa: F401  (subclasses Indexer: import before the patch)
    monkeypatch.setattr(v32, "Indexer", lambda cfg: None)


def _causal(L):
    return mx.tril(mx.ones((L, L), dtype=mx.bool_))


@pytest.mark.parametrize("L", [1, 5])
def test_rope_form_vs_glm5_deepseek_v32(no_indexer, L):
    from mlx_vlm.models.deepseek_v32 import language as v32
    rng = np.random.default_rng(L)
    m = _ours(rng=rng)
    ref = v32.DeepseekV32Attention(_v32_cfg())
    ref.indexer = lambda *a, **k: None
    _copy(ref, m)
    x = mx.array(rng.standard_normal((2, L, D)).astype(np.float32))
    qr, q_nope, q_pe, lat, k_pe = m.project(x)
    _eq(qr, ref.q_a_layernorm(ref.q_a_proj(x)))
    rq = ref.q_b_proj(qr).reshape(2, L, NH, NOPE + ROPE).transpose(0, 2, 1, 3)
    _eq(q_nope, rq[..., :NOPE])
    _eq(q_pe, ref.rope(rq[..., NOPE:], 0))
    ckv = ref.kv_a_proj_with_mqa(x)
    _eq(lat, ref.kv_a_layernorm(ckv[..., :KV])[:, None])
    _eq(k_pe, ref.rope(ckv[..., KV:].reshape(2, L, 1, ROPE).transpose(0, 2, 1, 3), 0))
    _close(m(x, mask=_causal(L)), ref(x, mask=_causal(L)))


@pytest.mark.parametrize("L", [1, 5])
def test_gate_sinks_vs_hy_v4(no_indexer, L):
    from mlx_vlm.models.hy_v4 import language as H4
    rng = np.random.default_rng(10 + L)
    m = _ours(rng=rng, gate=True, sinks=True)
    ref = H4.HyV4Attention(_v32_cfg(), 0)
    _copy(ref, m)
    ref.linear_gate.weight, ref.learnable_sink_param = m.gate_proj.weight, m.sinks
    x = mx.array(rng.standard_normal((2, L, D)).astype(np.float32))
    out, _ = ref(x, mask=_causal(L))
    _close(m(x, mask=_causal(L)), out)


@pytest.mark.parametrize("L", [1, 4])
def test_nope_form_vs_glm5_next(L):
    from mlx_vlm.models.glm5_next import language as G
    rng = np.random.default_rng(20 + L)
    m = _ours(rope=0, rng=rng)
    cfg = SimpleNamespace(num_attention_heads=NH, q_lora_rank=QL, qk_nope_head_dim=NOPE, kv_lora_rank=KV, v_head_dim=V, indexer_types=["shared"],
                          hidden_size=D, attention_bias=False, rms_norm_eps=1e-6)
    ref = G.Glm5NextAttention(cfg, 0)
    _copy(ref, m, fused=True)
    x = mx.array(rng.standard_normal((2, L, D)).astype(np.float32))
    j = mx.arange(L)
    topk = mx.broadcast_to(mx.where(j[None, :] <= j[:, None], j[None, :], -1), (2, L, L)).astype(mx.int32)
    out, _ = ref(x, prev_topk_indices=topk)
    qr, q_nope, _, lat, _ = m.project(x)
    q_a, kv_a = mx.split(ref.qkv_a_proj(x), (QL,), axis=-1)
    # the reference runs ONE GEMM over the concatenated q_a | kv_a weight (its own blocking): equal at L = 1 (GEMV), fp32 rounding above
    chk = _eq if L == 1 else (lambda a, b: _close(a, b, 1e-6))
    chk(qr, ref.q_a_layernorm(q_a))
    chk(lat, ref.kv_a_layernorm(kv_a)[:, None])
    _close(m(x, mask=_causal(L)), out)


# ---------------------------------------------------------------- N3 behind N1: per-row offsets
KPOOL = dict(H=3, dx=16, kp=4, top_k=8, select_tail=True, n_rot=8, base=10000.0)


class _Cache:
    """Right-aligned batch cache (mlx-lm layout): keys [B, 1, S, W], offset int or int array [B]."""

    def __init__(self, keys=None, offset=0):
        self.keys, self.offset = keys, offset

    def update_and_fetch(self, k, v):
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=2)
        self.offset = self.offset + k.shape[2]
        return self.keys, v


def _batch_cache(solos, offs):
    S = max(offs)
    W = solos[0].keys.shape[-1] if any(c.keys is not None for c in solos) else None
    rows = []
    for c, o in zip(solos, offs):
        k = c.keys if c.keys is not None else mx.zeros((1, 1, 0, W))
        rows.append(mx.pad(k, [(0, 0), (0, 0), (S - o, 0), (0, 0)]))
    return _Cache(mx.concatenate(rows, axis=0), mx.array(offs, dtype=mx.int32))


@pytest.mark.parametrize("offs", [[13], [0, 5, 13, 30], [30, 2, 17, 9]])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("tail", [True, False])
def test_kpool_behind_mla_rows(offs, L, tail):
    rng = np.random.default_rng(sum(offs) + L)
    m = _ours(rng=rng, gate=True, sinks=True, kpool=dict(KPOOL, select_tail=tail))
    xs = [mx.array(rng.standard_normal((1, o + L, D)).astype(np.float32)) for o in offs]
    # per-row reference: the whole row in one call, no cache (positions 0 .. o + L - 1), its last L outputs
    refs = [m(x)[:, -L:] for x in xs]
    solos = []
    for x, o in zip(xs, offs):
        c = _Cache()
        if o:
            m(x[:, :o], cache=c)
        solos.append(c)
    W = next(c.keys for c in solos if c.keys is not None).shape[-1]
    for c in solos:
        if c.keys is None:
            c.keys = mx.zeros((1, 1, 0, W))
    bc = _batch_cache(solos, offs)
    out = m(mx.concatenate([x[:, -L:] for x in xs], axis=0), cache=bc)
    for b, r in enumerate(refs):
        _close(out[b:b + 1], r)


def test_kpool_behind_mla_selects():
    """The selection actually drops pools here (long row, top_k = 2 pools): the output differs from the same layer without N3's keep."""
    rng = np.random.default_rng(5)
    m = _ours(rng=rng, kpool=KPOOL)
    x = mx.array(rng.standard_normal((1, 40, D)).astype(np.float32))
    full = m(x)
    kp = m.kp
    m.kp = None
    dense = m(x, mask="causal")
    m.kp = kp
    assert float(mx.max(mx.abs(full[:, -8:] - dense[:, -8:])).item()) > 1e-3
    _close(full[:, :kp["kp"]], dense[:, :kp["kp"]])          # the first group sees only its own tail: same keys as causal


def test_keys_match_slot_map():
    from kyojin_mlx import foreign_skeleton_blocks as SB
    from kyojin_mlx import foreign_skeleton_hc as HC
    assert MLA.KEYS == SB.MLA_KEYS and HC.KEYS == SB.HC_KEYS
    assert MLA.fires({k + ".weight" for k in ("attn_kv_a_mqa", "attn_k_b", "attn_v_b")}) and not MLA.fires({"attn_q.weight"})

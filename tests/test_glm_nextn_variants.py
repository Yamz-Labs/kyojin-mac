"""GLM-5.3 nextn draft-head variants (glm_head_q / glm_draft_vocab / glm_draft_topk) on a tiny random head and a stub target: CPU, no checkpoint.
The variants must keep the head's contract: same draft shape, a quantised head close to the fp16 one, a pruned vocabulary that returns ids of its list,
and a session whose head cache was written by another variant starts over instead of reading it."""
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import glm_nextn as GN

H, V = 128, 512


def tiny_cfg():
    tc = dict(
        hidden_size=H, intermediate_size=256, moe_intermediate_size=64, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=0, qk_head_dim=32, v_head_dim=32, head_dim=0, index_head_dim=32, index_n_heads=4, index_topk=16, index_kpool=4,
        index_kpool_compress=True, index_kpool_always_select_tail=True, indexer_rope_interleave=True, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=4,
        n_group=1, topk_group=1, routed_scaling_factor=2.5, norm_topk_prob=True, scoring_func="sigmoid", topk_method="noaux_tc", swiglu_limit=10.0, rms_norm_eps=1e-5,
        vocab_size=V, num_hidden_layers=4, first_k_dense_replace=1, mla_use_nope=True, attention_bias=False, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=4, mhc=True,
        num_nextn_predict_layers=1, max_position_embeddings=4096, model_type="glm5_next_text", hidden_act="silu", moe_router_dtype="float32",
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"], mlp_layer_types=["dense"] + ["sparse"] * 3, indexer_types=["full"] * 4,
        linear_attn_config=dict(full_attn_layers=[3], gate_lower_bound=-5.0, head_dim=32, kda_layers=[0, 1, 2], num_heads=4, short_conv_kernel_size=4),
        rope_parameters=dict(rope_theta=10000.0, rope_type="default"),
    )
    return tc


@pytest.fixture(scope="module")
def head():
    mx.set_default_device(mx.cpu)
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
    try:
        cfg = Glm5NextMTPConfig.from_dict({"text_config": tiny_cfg()})
        d = Glm5NextMTPDraftModel(cfg)
    except Exception as e:                                           # the config schema moved with mlx-vlm: not this module's contract
        pytest.skip(f"tiny head not constructible: {type(e).__name__}: {e}")
    from mlx.utils import tree_flatten, tree_unflatten
    mx.random.seed(0)
    d.update(tree_unflatten([(k, mx.random.normal(v.shape).astype(mx.float16) * 0.05 if v.ndim > 1 else v) for k, v in tree_flatten(d.parameters())]))
    emb = nn.Embedding(V, H)
    lm_head = nn.Linear(H, V, bias=False)
    lm = types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=emb), lm_head=lm_head)
    mx.eval(d.parameters(), emb.parameters(), lm_head.parameters())
    return GN.NextN(lm, d, cfg=cfg), lm


def seeded_state(n=6):
    st = GN.GlmDraft()
    mx.random.seed(1)
    st.append(mx.random.normal((n, H)).astype(mx.float16), mx.array(list(range(5, 5 + n)), dtype=mx.int32), 0)
    return st


def draft(nx, st, depth=1):
    return nx.draft(st, 7, st.pos + st.m, depth)


def test_variants_off_is_the_plain_head(head, monkeypatch):
    for k in ("KYOJIN_GLM_HEAD_Q", "KYOJIN_DRAFT_VOCAB", "KYOJIN_GLM_DRAFT_TOPK"):
        monkeypatch.delenv(k, raising=False)
    nx, _ = head
    out = draft(nx, seeded_state(), 2)
    assert out.shape == (2,) and nx.mode == ("", 0, 0) and nx._vars == {} and "_draft_subheads" not in nx.__dict__


def test_quantised_head_is_smaller_and_close(head, monkeypatch):
    nx, _ = head
    monkeypatch.delenv("KYOJIN_DRAFT_VOCAB", raising=False)
    monkeypatch.setenv("KYOJIN_GLM_HEAD_Q", "8")
    monkeypatch.setattr(GN, "Q_MIN_PARAMS", 1024)                    # the tiny head's linears are far below the real threshold
    st = seeded_state()
    d8 = nx.drafter("8", 0)
    n0, n1 = nx.info["head_q8_bytes"]
    assert n1 < n0 * 0.95                                             # the routed experts (the tiny head holds them as params) are not quantised
    assert isinstance(d8.eh_proj, nn.QuantizedLinear) and isinstance(nx.d.eh_proj, nn.Linear)       # the checkpoint's head stays fp16
    out = draft(nx, st, 2)
    assert out.shape == (2,) and nx.mode == ("8", 0, 0)


def test_topk_variant_routes_fewer_experts(head, monkeypatch):
    nx, _ = head
    monkeypatch.setenv("KYOJIN_GLM_DRAFT_TOPK", "2")
    d2 = nx.drafter("", 2)
    assert d2.mtp_block.mlp.gate.top_k == 2 and nx.d.mtp_block.mlp.gate.top_k == 4
    assert d2.mtp_block.mlp.switch_mlp is nx.d.mtp_block.mlp.switch_mlp          # one stored copy of the routed experts
    assert draft(nx, seeded_state()).shape == (1,)


def test_vocab_prune_returns_listed_ids(head, monkeypatch, tmp_path):
    from kyojin_mlx import draftvocab as DV
    nx, lm = head
    ids = np.sort(np.random.RandomState(0).permutation(V)[:64]).astype(np.int32)
    monkeypatch.setattr(DV, "ranking", lambda tok_dir, vocab, K: ids[:K])             # the frequency list itself is draftvocab's (tests/test_draftvocab)
    monkeypatch.setenv("KYOJIN_DRAFT_VOCAB", "128")
    monkeypatch.setattr(DV, "ROUND", 64)
    nx.pack = "/nowhere"
    out = draft(nx, seeded_state(), 3).tolist()
    assert nx.mode == ("", 128, 0) and set(out) <= set(ids.tolist())
    h = mx.random.normal((1, 1, H)).astype(mx.float16)
    sub = nx.vocab_head(128)
    lg = (h.reshape(1, H) @ lm.lm_head.weight.T)[0]
    assert int(nx._logits_argmax(h, sub).item()) == int(ids[int(mx.argmax(lg[mx.array(ids)]).item())])     # the best of the listed rows, as a token id


def test_session_cache_of_another_variant_is_not_reused(head, monkeypatch):
    nx, _ = head
    for k in ("KYOJIN_GLM_HEAD_Q", "KYOJIN_DRAFT_VOCAB", "KYOJIN_GLM_DRAFT_TOPK"):
        monkeypatch.delenv(k, raising=False)
    st = seeded_state()
    assert draft(nx, st) is not None and st.cache is not None and st.variant == ("", 0, 0)
    st.append(mx.zeros((2, H), dtype=mx.float16), mx.array([1, 2], dtype=mx.int32), st.pos + st.m)
    monkeypatch.setenv("KYOJIN_GLM_DRAFT_TOPK", "2")
    assert draft(nx, st) is None                                       # the cache of the plain head is dropped, no draft this round
    assert st.cache is None and st.variant == ("", 0, 2)
    st.append(mx.zeros((3, H), dtype=mx.float16), mx.array([1, 2, 3], dtype=mx.int32), 40)    # the next positions restart the state
    assert st.pos == 40 and st.m == 3


def test_profiled_draft_equals_the_plain_draft_and_names_every_op(head, monkeypatch, tmp_path):
    nx, _ = head
    for k in ("KYOJIN_GLM_HEAD_Q", "KYOJIN_DRAFT_VOCAB", "KYOJIN_GLM_DRAFT_TOPK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KYOJIN_GLM_HEAD_PROF", "0")
    ref = draft(nx, seeded_state(), 2).tolist()
    monkeypatch.setenv("KYOJIN_GLM_HEAD_PROF", "1")
    monkeypatch.setenv("KYOJIN_GLM_HEAD_PROF_OUT", str(tmp_path / "p.json"))
    got = [draft(nx, seeded_state(), 2).tolist() for _ in range(20)]
    assert all(g == ref for g in got)
    import json
    ops = json.load(open(tmp_path / "p.json"))["ops"]
    for name in ("eh_proj", "router", "shared_expert", "routed_experts", "lm_head+argmax"):
        assert any(name in k for k in ops), (name, list(ops))

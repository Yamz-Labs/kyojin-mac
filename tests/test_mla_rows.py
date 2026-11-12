"""MLA attention (mlx-lm deepseek_v3 class) under the per-user batched cache: a RowsKV decode step of two users with different lengths equals each user alone."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models import deepseek_v3 as DS
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import multiuser as MU


def _model():
    mx.random.seed(0)
    a = DS.ModelArgs(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, n_shared_experts=1, n_routed_experts=4, kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=16, v_head_dim=16,
                     qk_nope_head_dim=16, num_experts_per_tok=2, first_k_dense_replace=1)
    m = DS.Model(a)
    m.set_dtype(mx.float16)                          # served dtype: the row-invariant linears are the fp16 ones
    mx.eval(m.parameters())
    return m


def test_mla_rows_decode_equals_each_user_alone():
    m = _model()
    MU.install(m)                                    # as Q35Backend: row-invariant linears (else B = 2 already differs from B = 1 by ~2e-3 in stock mlx-lm)
    assert MU.install_mla(m) == 2
    prompts = [[1, 5, 9, 3, 7], [2, 4]]
    users = []
    for p in prompts:
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([p]), cache=c))
        users.append(c)
    alone = []
    for u in users:                                  # reference: each user's next step alone, on a copy of its cache state
        c = make_prompt_cache(m)
        for dst, src in zip(c, u):
            dst.keys, dst.values, dst.offset = src.keys, src.values, src.offset
        alone.append(m(mx.array([[11]]), cache=c)[0, -1])
    bc = MU.batch_caches(users, 0)
    assert all(isinstance(c, MU.RowsKV) for c in bc)
    y = m(mx.array([[11], [11]]), cache=bc)[:, -1]
    mx.eval(y, *alone)
    for i in range(2):
        np.testing.assert_allclose(np.array(y[i], np.float32), np.array(alone[i], np.float32), rtol=0, atol=0)
    assert [u[0].offset for u in MU.unbatch_caches(bc, 2)] == [6, 3]


def test_mla_layers_keep_a_plain_kvcache_under_the_kv_pools(monkeypatch):
    """The KYOJIN_TQ pool pairs K and V of one head dim; the MLA latent cache (kv_lora_rank, rope dim) stays a KVCache."""
    from mlx_lm.models.cache import KVCache
    from kyojin_mlx import kvwin, tq

    class Pool:
        def __init__(self, **kw):
            pass

    m = _model()
    monkeypatch.setattr(tq, "config", lambda: {"x": 1})
    monkeypatch.setattr(tq, "TQKVCache", Pool)
    monkeypatch.setattr(kvwin, "install", lambda: None)
    assert kvwin._mla_layers(m, 2) == {0, 1}
    assert all(type(c) is KVCache for c in kvwin.make_prompt_cache(m))

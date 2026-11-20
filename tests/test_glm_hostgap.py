"""KYOJIN_GLM_HOSTGAP: the early submit of a decode step moves only where the graph is committed (CPU, tiny stock glm5_next stack, no model load)."""
import types

import mlx.core as mx
import pytest

from kyojin_mlx import glm_hostgap as HG

mx.set_default_device(mx.cpu)
N = 7


def _cfg():
    from mlx_vlm.models.glm5_next.config import TextConfig
    return TextConfig(vocab_size=64, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=N, num_attention_heads=4, num_key_value_heads=4,
                      n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, kv_lora_rank=16, q_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16,
                      qk_head_dim=16, linear_head_dim=16, linear_num_heads=4, index_topk=16, index_kpool=4, index_head_dim=16, index_n_heads=2,
                      layer_types=["linear_attention"] * N, mlp_layer_types=["sparse" if i % 2 else "dense" for i in range(N)], indexer_types=["full"] * N, hc_mult=4)


def _model():
    from mlx_vlm.models.glm5_next.language import Glm5NextTextModel
    mx.random.seed(0)
    m = Glm5NextTextModel(_cfg())
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


def _caches(m, rows, toks):
    from mlx_vlm.models.cache import ArraysCache
    cs = [ArraysCache(size=2) for _ in m.layers]
    m(toks, cache=cs)
    return cs


@pytest.fixture
def calls(monkeypatch):
    seen = []
    real = mx.async_eval
    monkeypatch.setattr(mx, "async_eval", lambda *a, **k: (seen.append(len(a)), real(*a, **k))[1])
    return seen


@pytest.mark.parametrize("rows", [1, 3])
def test_decode_step_is_bit_identical_and_submits_in_groups(calls, rows):
    m = _model()
    HG.install(types.SimpleNamespace(model=m))
    assert HG.install(types.SimpleNamespace(model=m)) == 0                        # idempotent
    pre = mx.random.randint(0, 64, (rows, 5))
    tok = mx.random.randint(0, 64, (rows, 1))
    HG.ENABLED, HG.FIRST, HG.EVERY, HG.ROWS = False, 1, 2, 1
    off = m(tok, cache=_caches(m, rows, pre))
    assert calls == []                                                            # flag off: no submit
    HG.ENABLED = True
    on = m(tok, cache=_caches(m, rows, pre))
    # layers built so far = 1, 3, 5 (never the last layer: the caller submits the logits)
    assert len(calls) == 3
    assert mx.array_equal(off, on).item()
    HG.ENABLED = False


def test_more_rows_than_the_cap_keep_their_own_submits(calls):
    m = _model()
    HG.install(types.SimpleNamespace(model=m))
    HG.ENABLED, HG.FIRST, HG.EVERY, HG.ROWS = True, 1, 2, 1
    pre = mx.random.randint(0, 64, (1, 5))
    cs = _caches(m, 1, pre)
    calls.clear()
    m(mx.random.randint(0, 64, (1, 2)), cache=cs)                                 # 2 rows of one session: a verify pass
    assert calls == []
    HG.ENABLED = False

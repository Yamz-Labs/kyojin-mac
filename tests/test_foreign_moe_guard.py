"""foreign_moe.install must not run a router the softmax routed kernel cannot compute exactly (CPU only, tiny blocks of the installed mlx_lm classes)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)
from mlx_lm.models import lfm2_moe as L2, qwen2_moe as Q2, qwen3_moe as Q3, qwen3_next as QN  # noqa: E402

from kyojin_mlx import fired, foreign_linear as FL, foreign_moe as FM  # noqa: E402

# ---- the wrong-kernel guard of foreign_moe.install ----
H, E, TOPK = 32, 8, 2


def _args(cls, **kw):
    base = dict(model_type="x", vocab_size=50, hidden_size=H, intermediate_size=16, moe_intermediate_size=16, num_hidden_layers=2, num_experts=E, num_experts_per_tok=TOPK,
                norm_topk_prob=True, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, shared_expert_intermediate_size=16)
    base.update(kw)
    import dataclasses
    for f in dataclasses.fields(cls):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in base:
            base[f.name] = 2                                  # a required field the block does not read here
    return cls(**{k: v for k, v in base.items() if k in cls.__dataclass_fields__})


def _qwen3(norm=True):
    a = _args(Q3.ModelArgs, decoder_sparse_step=1, mlp_only_layers=[], rms_norm_eps=1e-6, head_dim=16, rope_theta=1e4, tie_word_embeddings=False, norm_topk_prob=norm)
    return Q3.Qwen3MoeSparseMoeBlock(a)


def _qwen3next(norm=False):
    a = _args(QN.ModelArgs, norm_topk_prob=norm)
    return QN.Qwen3NextSparseMoeBlock(a)


def _qwen2():
    return Q2.Qwen2MoeSparseMoeBlock(_args(Q2.ModelArgs))


def _lfm2(bias, norm=True, scale=1.0):
    a = _args(L2.ModelArgs, use_expert_bias=bias, num_dense_layers=1, norm_eps=1e-5, conv_bias=False, conv_L_cache=3, layer_types=["conv", "full_attention"], norm_topk_prob=norm,
              routed_scaling_factor=scale)
    b = L2.Lfm2MoeSparseMoeBlock(a)
    if bias:
        b.expert_bias = mx.random.normal((E,), key=mx.random.key(3))
    return b


def _randomise(blk):
    blk.gate.weight = mx.random.normal(blk.gate.weight.shape, key=mx.random.key(11)) * 0.5
    return blk


@pytest.mark.parametrize("make", [_qwen3, lambda: _qwen3(False), _qwen3next, lambda: _qwen3next(True)])
def test_softmax_blocks_are_accepted(make):
    assert FM.route_why(_randomise(make())) is None


@pytest.mark.parametrize("make", [lambda: _lfm2(True), lambda: _lfm2(False), lambda: _lfm2(True, False), lambda: _lfm2(False, True, 2.5)])
def test_sigmoid_routers_are_refused(make):
    assert FM.route_why(_randomise(make())) is not None


def test_refusal_reasons():
    assert FM.route_why(_randomise(_lfm2(True))) == "selection_bias"
    assert FM.route_why(_randomise(_lfm2(False))) == "router_not_softmax"                 # sigmoid, no bias, renormalised: only the probe sees it
    assert FM.route_why(_randomise(_lfm2(False, True, 2.5))) == "grouped_or_scaled"
    assert FM.route_why(_randomise(_qwen2())) == "no_norm_topk_prob"                     # stock qwen2_moe never renormalises and has no attribute: the fused call would raise, so the block stays stock


def test_declared_gating_wins_and_probe_failure_is_a_refusal():
    b = _randomise(_qwen3())
    b.moe_gating = {"kind": "sigmoid", "bias": False, "renorm": True}
    assert FM.route_why(b) == "gating_sigmoid"
    b.moe_gating = {"kind": "softmax", "bias": True, "renorm": True}
    assert FM.route_why(b) == "selection_bias"
    c = _qwen3()
    c.top_k = 20                                                        # the block cannot run its own call
    assert FM.route_why(c).startswith("probe_failed")


def test_install_keeps_the_probe_free_of_side_effects_and_counts_refusals(monkeypatch):
    monkeypatch.setattr(FL, "ForeignSwitchGLU", type(_qwen3().switch_mlp))
    monkeypatch.setattr(FM, "ON", True)
    good, bad, bad2 = _randomise(_qwen3()), _randomise(_lfm2(True)), _randomise(_lfm2(False))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = good, bad, bad2
    m, logs = M(), []
    sw = (good.switch_mlp, bad.switch_mlp)
    fired.reset()
    n = FM.install(m, log=logs.append)
    assert n == 1 and getattr(type(good), "_foreign_moe", False) and not getattr(type(bad), "_foreign_moe", False) and not getattr(type(bad2), "_foreign_moe", False)
    assert good.switch_mlp is sw[0] and bad.switch_mlp is sw[1]                          # the probe put the experts back
    assert fired.STATS.get("foreign_moe.fb.router:selection_bias") == 1 and fired.STATS.get("foreign_moe.fb.router:router_not_softmax") == 1
    assert any("left on their own call" in s for s in logs)
    y = bad(mx.random.normal((1, 3, H)))                                                  # still the stock call, still runs
    assert y.shape == (1, 3, H)

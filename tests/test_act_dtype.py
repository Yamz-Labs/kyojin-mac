# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_FOREIGN_ACT_DTYPE: a tiny qwen4_exp whose conv1d / inject weights are F32 (as the GGUF stores them) promotes the hidden state to fp32; _cast_act keeps it in the half dtype. CPU only."""
import pytest

mx = pytest.importorskip("mlx.core")
mod = pytest.importorskip("mlx_vlm.models.qwen4_exp")
from mlx.utils import tree_flatten, tree_unflatten

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import foreign_qwen4exp_gguf as Q
from test_qwen4exp_gguf import _G, _kv


def _run(cast, dt):
    mx.set_default_device(mx.cpu)
    m = mod.Model(mod.ModelConfig.from_dict(Q.config(_G(_kv()), "qwen4exp")))
    lm = m.language_model
    mx.random.seed(0)
    w = {k: (mx.random.normal(p.shape) * 0.05).astype(mx.float32) if p.dtype == mx.float32 else p for k, p in tree_flatten(lm.parameters())}
    if cast:
        FL._cast_act(w, dt)
    lm.update(tree_unflatten([(k, (v.astype(dt) if v.dtype == mx.float32 and not FL._KEEP_F32.search(k) and cast is None else v)) for k, v in w.items()]))
    # the loader's 1-D rule: everything but the conv / inject weights is already half
    lm.update(tree_unflatten([(k, p.astype(dt) if (p.dtype == mx.float32 and not FL._KEEP_F32.search(k) and not any(s in k for s in ("conv1d.weight", "block_inject_weight"))) else p) for k, p in tree_flatten(lm.parameters())]))
    import mlx.nn as nn

    class Id(nn.Module):                              # the MoE (gather_mm) has no half kernel on the CPU: identity keeps the dtype test about the residual stream
        def __call__(self, x, *a, **k):
            return x
    for l in lm.model.layers:
        l.mlp = Id()
    lg = lm(mx.array([[1]]))
    lg = lg.logits if hasattr(lg, "logits") else lg
    mx.eval(lg)
    return lg.dtype, {k: p.dtype for k, p in tree_flatten(lm.parameters()) if k.endswith(("A_log", "dt_bias"))}


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_hidden_stays_half(dt):
    base, _ = _run(False, dt)
    assert base == mx.float32                        # the bug: F32 conv / inject weights promote everything
    out, keep = _run(True, dt)
    assert out == dt
    assert all(v == mx.float32 for v in keep.values()) and keep


def test_flag_parse(monkeypatch):
    monkeypatch.delenv("KYOJIN_FOREIGN_ACT_DTYPE", raising=False)
    assert FL.act_dtype(mx.float16) is None
    monkeypatch.setenv("KYOJIN_FOREIGN_ACT_DTYPE", "bf16")
    assert FL.act_dtype(mx.float16) == mx.bfloat16
    monkeypatch.setenv("KYOJIN_FOREIGN_ACT_DTYPE", "auto")
    assert FL.act_dtype(mx.float16) == mx.float16

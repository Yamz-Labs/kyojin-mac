"""The fused dense gated-MLP launch (install_dense_glu) computes down(silu(gate) * up): it must only replace modules whose own call is exactly that.
gelu-gated classes (gemma family) must keep their call; silu classes of our lanes keep the fused launch. CPU only."""
import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx.foreign_linear import silu_gated

mx.set_default_device(mx.cpu)


def _bare(modname, clsname, **attrs):
    try:
        C = getattr(importlib.import_module(modname), clsname)
    except Exception as e:                                           # class not shipped by this mlx-lm / mlx-vlm build
        pytest.skip(f"{modname}.{clsname}: {type(e).__name__}")
    o = C.__new__(C)
    nn.Module.__init__(o)
    for k in ("gate_proj", "up_proj", "down_proj"):
        o[k] = nn.Identity()
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


@pytest.mark.parametrize("modname,clsname", [
    ("mlx_lm.models.llama", "MLP"),
    ("mlx_lm.models.qwen3", "MLP"),
    ("mlx_lm.models.qwen3_moe", "MLP"),
    ("mlx_lm.models.glm4_moe", "MLP"),
    ("mlx_lm.models.deepseek_v3", "DeepseekV3MLP"),
    ("mlx_lm.models.deepseek_v2", "DeepseekV2MLP"),
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5MLP"),
])
def test_silu_classes_keep_the_fused_launch(modname, clsname):
    assert silu_gated(_bare(modname, clsname)) is True


@pytest.mark.parametrize("modname,clsname", [
    ("mlx_lm.models.gemma3_text", "MLP"),
    ("mlx_lm.models.gemma2", "MLP"),
    ("mlx_lm.models.gemma", "MLP"),
    ("mlx_vlm.models.gemma3.language", "MLP"),
])
def test_gelu_classes_keep_their_call(modname, clsname):
    assert silu_gated(_bare(modname, clsname)) is False


def test_activation_quant_class_keeps_its_call():
    """DS4.1's MLP fake-quantises the activations (fp8 ue8m0): not plain silu * up, so it keeps its own call."""
    assert silu_gated(_bare("mlx_vlm.models.deepseek_v41.language", "DeepseekV41MLP", swiglu_limit=10.0)) is False


def test_probe_failure_is_false():
    class Broken(nn.Module):
        def __call__(self, x):
            raise RuntimeError("no")
    o = Broken()
    for k in ("gate_proj", "up_proj", "down_proj"):
        o[k] = nn.Identity()
    assert silu_gated(o) is False

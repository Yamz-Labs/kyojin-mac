"""The fused dense gated-MLP launch computes silu(gate) * up: a gated MLP with another activation must be left alone (gemma classes use gelu)."""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
from mlx_lm.models import gemma2, gemma3_text, llama, qwen3  # noqa: E402

from kyojin_mlx import foreign_linear as FL  # noqa: E402


def test_probe_tells_silu_from_gelu():
    llama_mlp = llama.MLP(llama.ModelArgs(model_type="llama", hidden_size=16, num_hidden_layers=1, intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=8))
    q = qwen3.MLP(16, 32)
    g2 = gemma2.MLP(16, 32)
    g3 = gemma3_text.MLP(16, 32)
    assert FL.silu_gated(llama_mlp) and FL.silu_gated(q)
    assert not FL.silu_gated(g2) and not FL.silu_gated(g3)


def test_probe_is_false_when_it_cannot_run():
    class Odd:
        pass
    assert FL.silu_gated(Odd()) is False

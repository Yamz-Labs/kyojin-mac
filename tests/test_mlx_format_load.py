"""MLX-format quantized checkpoints (mlx-community layout: config.json "quantization" {group_size, bits, mode}, `<name>.weight` + `.scales` (+ `.biases` for affine))
of a text-only Qwen3: written to a tiny dir in each mode, loaded through serve.backends.load_q35, served on the plain Q35Backend (no Qwen3.5 wrapper), the linears stay
quantized (no fp16 copy) and the logits equal the stock mlx-lm model on the same files. Metal only; tiny shapes, no tokenizer (mlx_lm.load is stubbed to load_model)."""
import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")

MODES = {"affine": (64, 4), "mxfp4": (32, 4), "nvfp4": (16, 4)}


def _write(dst: Path, mode: str, gs: int, bits: int, kind: str = "qwen3"):
    from mlx.utils import tree_flatten
    from importlib import import_module
    qwen3 = import_module(f"mlx_lm.models.{kind}")
    cfg = {"model_type": kind, "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 256, "rope_theta": 10000.0, "max_position_embeddings": 512, "tie_word_embeddings": False}
    if kind == "qwen3_moe":                                                                                                          # stacked experts (SwitchGLU)
        cfg.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True)
    mx.random.seed(7)
    m = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=gs, bits=bits, mode=mode)
    mx.eval(m.parameters())
    dst.mkdir()
    mx.save_safetensors(str(dst / "model.safetensors"), dict(tree_flatten(m.parameters())))
    json.dump({**cfg, "quantization": {"group_size": gs, "bits": bits, "mode": mode}}, open(dst / "config.json", "w"))


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe"])
@pytest.mark.parametrize("mode", list(MODES))
def test_mlx_format_qwen3_loads_quantized_and_matches_mlx_lm(tmp_path, monkeypatch, mode, kind):
    gs, bits = MODES[mode]
    _write(tmp_path / mode, mode, gs, bits, kind)
    d = str(tmp_path / mode)
    names = set(mx.load(d + "/model.safetensors"))
    assert "model.layers.0.self_attn.q_proj.scales" in names and "lm_head.scales" in names and "model.embed_tokens.scales" in names     # mlx-lm names
    assert ("model.layers.0.self_attn.q_proj.biases" in names) == (mode == "affine")                                                  # biases: affine only

    import mlx_lm
    from mlx_lm.utils import load_model
    monkeypatch.setattr(mlx_lm, "load", lambda p, **k: (load_model(Path(p))[0], None))
    monkeypatch.setenv("KYOJIN_WARM_LOAD", "0")
    from kyojin_mlx.serve import backends as B
    from kyojin_mlx.serve.q35mlx import Q35MlxBackend
    be, cfg = B.load_q35(d, log=lambda *_: None)
    assert type(be) is B.Q35Backend and not isinstance(be, Q35MlxBackend) and cfg["quantization"]["mode"] == mode
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    qs = [x for _, x in be.model.named_modules() if isinstance(x, (nn.QuantizedLinear, QuantizedSwitchLinear))]
    assert qs and all(x.mode == mode and x.group_size == gs and x.bits == bits for x in qs)                                          # as stored, never dequantized

    ref = load_model(Path(d))[0]
    ids = list(range(5, 29))
    got = be.prefill_chunk(be.make_cache(), ids).reshape(-1)
    want = ref(mx.array(ids)[None])[0, -1]
    mx.eval(got, want)
    assert int(mx.argmax(got)) == int(mx.argmax(want))
    assert float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))) < 0.1 * float(mx.max(mx.abs(want.astype(mx.float32)))) + 1e-3

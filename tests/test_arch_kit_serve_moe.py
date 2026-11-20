"""Serve-path rows of an MoE architecture whose class names its stacked experts `switch_mlp` inside `block_sparse_moe` (not `mlp`): header-only, needs a real file.
Set KYOJIN_ARCHKIT_MODELS to a directory holding the granite-3.0-1b-a400m-instruct GGUF of any quantisation (skipped otherwise)."""
import glob
import os

import pytest

pytest.importorskip("mlx_lm")
D = os.environ.get("KYOJIN_ARCHKIT_MODELS")
FILES = sorted(glob.glob(os.path.join(D, "granite-3.0-1b-a400m-instruct*.gguf"))) if D else []


@pytest.mark.skipif(not FILES, reason="no granitemoe GGUF (KYOJIN_ARCHKIT_MODELS)")
def test_register_maps_experts_under_the_class_container_and_keeps_the_router_dense():
    from kyojin_mlx import foreign_archkit_serve as S, foreign_names as N
    N.ARCHS.pop("granitemoe", None)
    assert S.register(FILES[0]) == "granitemoe"
    A = N.ARCHS["granitemoe"]
    assert A.blk["ffn_gate_exps"] == "block_sparse_moe.experts.gate_proj" and A.xform == "granitemoe" and A.skip_nextn is False
    hf = N.gguf_name("blk.3.ffn_down_exps.weight", "granitemoe", None)
    assert hf == "model.layers.3.block_sparse_moe.experts.down_proj.weight"
    assert not N.is_linear(N.gguf_name("blk.3.ffn_gate_inp.weight", "granitemoe", None), 2, "granitemoe")      # the router stays a stored dense tensor
    assert N.is_linear("model.layers.3.self_attn.q_proj.weight", 2, "granitemoe")

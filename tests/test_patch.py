"""patch_model (kyojin_mlx.patch / archs). Name mapping needs no model. The smoke tests need the shrunk packs and mlx-vlm:
KYOJIN_SHRUNK=<dir holding ds2/ and glm2/> (made by tools/shrink_pack.py). Layer-level exactness: tools/layer_gate.py."""
import os

import pytest

from kyojin_mlx import archs as A


def test_target_names():
    # the model fuses q_a / kv_a (GLM) into qkv_a_proj: target() keeps the member name, FUSED says where it goes
    assert A.FUSED["q_a_proj"] == "qkv_a_proj" and A.FUSED["gate_proj"] == "gate_up_proj"
    for spec in A.ARCHS.values():
        assert A.target(spec, "model.layers.0.self_attn.o_proj") == A.target(spec.name, "model.layers.0.self_attn.o_proj")


def test_arch_of_and_tables():
    for spec in A.ARCHS.values():
        assert spec.name in A.ARCHS
        assert spec.layer_re
        if spec.moe is not None:
            assert spec.moe.convention in ("combine", "slot")
            assert sorted(spec.moe.roles.values()) == ["down", "gate", "up"]
    for fused, members in A.FUSED_ORDER.items():
        assert all(A.FUSED[m] == fused for m in members)


SHRUNK = os.environ.get("KYOJIN_SHRUNK")


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set")
@pytest.mark.parametrize("name", ["ds2", "glm2"])
def test_patch_shrunk(name):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from kyojin_mlx import patch as P
    pack = os.path.join(SHRUNK, name)
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing")
    model, spec, cfg = P.build_model(pack)
    rep = P.patch_model(model, pack, spec, cfg)
    assert rep["moe_layers"] >= 1 and rep["skipped"] == 0
    lm = model.language_model
    cache = lm.make_cache()
    vocab = cfg["text_config"]["vocab_size"]
    x = mx.array([[1, 100, vocab - 7, 5]])
    lg = lm(x[:, :3], cache=cache).logits
    ld = lm(x[:, 3:], cache=cache).logits
    mx.eval(lg, ld)
    assert bool(mx.all(mx.isfinite(lg.astype(mx.float32))).item()) and bool(mx.all(mx.isfinite(ld.astype(mx.float32))).item())
    assert ld.shape[-1] == vocab

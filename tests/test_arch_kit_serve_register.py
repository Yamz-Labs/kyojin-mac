"""gemma4 on the serve path (CPU only): foreign_archkit_serve.register() takes the same vetted config row as the CPU loader, behind either spelling of the switch, and a
file without the switch is refused as before. The per-layer embedding table is a lookup table, never a linear (it was handed to ForeignLinear, whose output width is the
vocabulary: the mlx-lm class then failed to unflatten it). KYOJIN_TEST_GEMMA4_E2B (a real gemma4 E2B GGUF) adds the same checks on that file; skipped when unset."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_archkit as K, foreign_archkit_serve as S, foreign_linear as FL, foreign_names as N  # noqa: E402
import test_arch_kit_gemma4 as T  # noqa: E402

REAL = os.environ.get("KYOJIN_TEST_GEMMA4_E2B")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in ("KYOJIN_GEMMA4_ROWS", "KYOJIN_KIT_UNPROVEN_ROWS"):
        monkeypatch.delenv(v, raising=False)
    saved = (dict(N.ARCHS), dict(N._CONFIG))
    N.ARCHS.pop("gemma4", None)
    N._CONFIG.pop("kit:gemma4", None)
    yield
    N.ARCHS.clear(); N.ARCHS.update(saved[0])
    N._CONFIG.clear(); N._CONFIG.update(saved[1])


@pytest.fixture
def tiny(tmp_path):
    m, cfg = T._model(T.DENSE)
    p = str(tmp_path / "g.gguf")
    T.write(p, m, cfg)
    return p


@pytest.mark.parametrize("switch", ["KYOJIN_GEMMA4_ROWS", "KYOJIN_KIT_UNPROVEN_ROWS"])
def test_register_accepts_with_either_switch(tiny, monkeypatch, switch):
    monkeypatch.setenv(switch, "1")
    assert S.register(tiny) == "gemma4"
    a = N.ARCHS["gemma4"]
    assert a.xform == "" and a.config == "kit:gemma4" and "model.embed_tokens_per_layer.weight" not in a.blk.values()


def test_register_refuses_without_the_switch(tiny):
    with pytest.raises(K.KitError, match="needs a vetted transform"):
        S.register(tiny)
    assert "gemma4" not in N.ARCHS


def test_per_layer_table_is_a_lookup_table_not_a_linear():
    assert not N.is_linear("model.embed_tokens_per_layer.weight", 2, "llama")
    assert N.is_linear("model.layers.0.mlp.up_proj.weight", 2, "llama")
    ti = SimpleNamespace(tname="Q4_0", ne=(8960, 262144))
    assert FL.embed_stored("model.embed_tokens_per_layer.weight", ti, "gemma4", SimpleNamespace(tensors={}))


@pytest.mark.skipif(not REAL, reason="KYOJIN_TEST_GEMMA4_E2B not set")
def test_real_file(monkeypatch):
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")
    assert S.register(REAL) == "gemma4"
    src = FL._read_gguf(REAL, "gemma4")
    assert "model.embed_tokens_per_layer" in src.embed and "model.embed_tokens_per_layer" not in src.packs

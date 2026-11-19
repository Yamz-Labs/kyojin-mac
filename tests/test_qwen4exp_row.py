# SPDX-License-Identifier: Apache-2.0
"""The unproven qwen4_exp serve row (modeldesc.UNPROVEN_DESCS): in force only with KYOJIN_KIT_UNPROVEN_ROWS; load_backend takes the stored-weights path then. CPU only."""
import json

import pytest

from kyojin_mlx import modeldesc as MD


def _pack(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(model_type="qwen4_exp", architectures=["Qwen4ExpForConditionalGeneration"])))
    return str(tmp_path)


def test_row_exists_only_with_the_switch(monkeypatch):
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    assert MD.backend_desc("qwen4_exp") is None and MD.by_architecture("Qwen4ExpForConditionalGeneration") is None
    assert MD.chat_desc("qwen4_exp").name == "qwen35"          # the qwen chat prefix, as before
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    assert MD.backend_desc("qwen4_exp").name == "qwen4_exp" and MD.backend_desc("qwen4_exp").backend == "generic_foreign"
    assert MD.backend_desc("qwen3_5_moe").name != "qwen4_exp"


def test_load_backend_switch_on_off(monkeypatch, tmp_path):
    from kyojin_mlx import foreign_linear as FL
    from kyojin_mlx.serve import backends as B
    monkeypatch.setenv("KYOJIN_WIRED_MB", "0")   # no wired limit: the CPU device has no max_recommended_working_set_size
    calls = []

    class Fake:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(FL, "is_foreign", lambda p: "safetensors")
    monkeypatch.setattr(FL, "load_foreign", lambda p, **kw: (calls.append(kw) or "M", {"model_type": "qwen4_exp"}))
    monkeypatch.setattr(B, "GenericBackend", Fake)
    monkeypatch.setenv("KYOJIN_BACKEND_FALLBACK", "0")
    p = _pack(tmp_path)
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    with pytest.raises(ValueError, match="no serving backend"):
        B.load_backend(p, log=lambda *a: None)
    assert not calls
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    be, cfg = B.load_backend(p, log=lambda *a: None)
    assert isinstance(be, Fake) and be.model == "M" and be.foreign and be.pack == p and calls and calls[0]["mtp"] is False

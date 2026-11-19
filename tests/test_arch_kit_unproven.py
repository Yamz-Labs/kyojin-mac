"""The one switch of the arch kit's unproven rows (KYOJIN_KIT_UNPROVEN_ROWS), both states on synthetic headers,."""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import day0_inspect as D  # noqa: E402

from kyojin_mlx import flags, foreign_archkit as K, foreign_cfgrows as CR, foreign_glm_gguf as GG, foreign_synth as SY, foreign_unproven as UP  # noqa: E402

ENV = "KYOJIN_KIT_UNPROVEN_ROWS"
HINT = f"unproven rows exist for this architecture: set {ENV} to try them"
BASE = [("token_embd.weight", (4, 8)), ("output_norm.weight", (8,)), ("blk.0.attn_norm.weight", (8,)), ("blk.0.ffn_norm.weight", (8,))]
# arch -> (verdict with the switch off = the verdict glm-speed gives, verdict with it on)
ARCHS = {"gemma4": ("ROW NEEDED", "READY"), "lfm2moe": ("ROW NEEDED", "READY"),
         "minimax-m3": ("MODEL CODE NEEDED", "ROW NEEDED"), "nemotron_h_moe": ("MODEL CODE NEEDED", "READY")}      # READY since the nemotron_h_moe config row + vetted line (still unproven: Metal identity not measured)


def _file(tmp_path, arch):
    kv = {"general.architecture": arch, f"{arch}.block_count": 2, f"{arch}.embedding_length": 8, f"{arch}.attention.head_count": 2, f"{arch}.context_length": 64,
          f"{arch}.expert_count": 4, f"{arch}.expert_used_count": 2, f"{arch}.attention.head_count_kv": 1}
    p = str(tmp_path / f"{arch}.gguf")
    SY.write_gguf(p, kv, [(n, sh, "F32", np.zeros(int(np.prod(sh)), np.float32).view(np.uint8)) for n, sh in BASE])
    return p


def test_the_switch_is_declared_default_off_and_gemma4_keeps_its_alias():
    assert flags.FLAGS["kit_unproven_rows"].env == ENV and flags.FLAGS["kit_unproven_rows"].default == "0"
    assert flags.FLAGS["gemma4_rows"].env == "KYOJIN_GEMMA4_ROWS"


def test_glm5_next_is_proven_ready_with_the_switch_off(tmp_path, monkeypatch):
    p = _file(tmp_path, "glm5-next")
    monkeypatch.delenv(ENV, raising=False)
    GG.register()
    r = D.inspect(p, use_census=True)
    assert not UP.has_unproven("glm5-next") and r["verdict"] == "READY" and HINT not in r["notes"], (r["verdict"], r["notes"])


def test_every_unproven_row_is_marked():
    assert CR.ROWS["gemma4"]["proven"] is False and CR.ROWS["lfm2moe"]["proven"] is False
    assert K.VETTED["gemma4"]["proven"] is False and K.VETTED["lfm2moe"]["proven"] is False
    assert set(UP.UNPROVEN_ALIASES) == {"minimax-m3", "nemotron_h_moe"} and K.MODEL_TYPE_ALIASES == {}
    assert all(UP.has_unproven(a) for a in ARCHS)
    assert not UP.has_unproven("qwen3") and not UP.has_unproven("gemma2")


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_day0_off_keeps_the_verdict_and_says_rows_exist_on_gives_the_new_one(tmp_path, monkeypatch, arch):
    p = _file(tmp_path, arch)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("KYOJIN_GEMMA4_ROWS", raising=False)
    GG.register()
    r = D.inspect(p, use_census=True)
    assert r["verdict"] == ARCHS[arch][0] and HINT in r["notes"], (r["verdict"], r["notes"])
    monkeypatch.setenv(ENV, "1")
    GG.register()
    try:
        r = D.inspect(p, use_census=True)
        assert r["verdict"] == ARCHS[arch][1] and HINT not in r["notes"], (r["verdict"], r["notes"])
    finally:
        monkeypatch.delenv(ENV)
        GG.register()


def test_off_leaves_no_row_anywhere(monkeypatch):
    from kyojin_mlx import foreign_names as N, foreign_xform as XF
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("KYOJIN_GEMMA4_ROWS", raising=False)
    GG.register()
    assert not CR.has_rows("gemma4") and not CR.has_rows("lfm2moe") and K.vetted("lfm2moe", "xform", None) is None and K.vetted("gemma4", "flags") == ()
    assert "glm5-next" in N.ARCHS and "glm5next" in N._CONFIG and "glm5next" in XF._SETS      # proven row: registered with the switch off
    assert K.model_type_candidates("minimax-m3") == [m for m in K.model_type_candidates("minimax-m3") if m != "minimax_m3_vl"]
    assert "nemotron_h" not in K.model_type_candidates("nemotron_h_moe")
    monkeypatch.setenv(ENV, "1")
    GG.register()
    try:
        assert CR.has_rows("gemma4") and CR.has_rows("lfm2moe") and K.vetted("lfm2moe", "xform") == "lfm2moe" and "glm5-next" in N.ARCHS
        assert "minimax_m3_vl" in K.model_type_candidates("minimax-m3") and "nemotron_h" in K.model_type_candidates("nemotron_h_moe")
    finally:
        monkeypatch.delenv(ENV)
        GG.register()


def test_gemma4_old_switch_still_turns_only_gemma4_on(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")
    assert CR.has_rows("gemma4") and not CR.has_rows("lfm2moe")

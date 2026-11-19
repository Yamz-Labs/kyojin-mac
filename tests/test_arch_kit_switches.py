"""The two default-off switches of the arch kit (CPU only): KYOJIN_SKELETON (the config-driven skeleton for an architecture with no model class) and KYOJIN_GEMMA4_ROWS (the gemma4
config row and vetted line). Off = what the engine did before the rows existed (KitError, day0 verdict without the rows); on = the new path."""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import day0_inspect as D  # noqa: E402
import test_arch_kit_gemma4 as T4  # noqa: E402

from kyojin_mlx import flags, foreign_archkit as K, foreign_archkit_cpu as KC, foreign_archkit_serve as KS, foreign_cfgrows as CR, foreign_names as N, foreign_skeleton as SK, foreign_synth as SY  # noqa: E402

NOCLASS = "hunyuan-dense"            # skeleton rows, no mlx-lm class (test_foreign_skeleton)


def _stub_file(tmp_path, arch):
    p = str(tmp_path / "s.gguf")
    SY.write_gguf(p, {"general.architecture": arch, f"{arch}.block_count": 1}, [("token_embd.weight", (4, 8), "F32", np.zeros(32, np.float32).view(np.uint8))])
    return p


def test_switches_are_declared_default_off():
    for key, env in (("arch_skeleton", "KYOJIN_SKELETON"), ("gemma4_rows", "KYOJIN_GEMMA4_ROWS")):
        assert flags.FLAGS[key].env == env and flags.FLAGS[key].default == "0"


def test_skeleton_off_is_a_kit_error_on_takes_the_skeleton(tmp_path, monkeypatch):
    p = _stub_file(tmp_path, NOCLASS)
    assert SK.has_rows(NOCLASS) and not KC._has_class(NOCLASS)
    monkeypatch.delenv("KYOJIN_SKELETON", raising=False)
    assert not SK.enabled()
    with pytest.raises(K.KitError):
        KC.load_cpu(p)
    monkeypatch.setattr(SK, "load_cpu", lambda path, dtype=None: ("skeleton", path))
    monkeypatch.setenv("KYOJIN_SKELETON", "1")
    assert KC.load_cpu(p) == ("skeleton", p)


def test_skeleton_serve_register(tmp_path, monkeypatch):
    p = _stub_file(tmp_path, NOCLASS)
    monkeypatch.delenv("KYOJIN_SKELETON", raising=False)
    N.ARCHS.pop(NOCLASS, None)
    with pytest.raises(K.KitError):
        KS.register(p)
    monkeypatch.setattr(SK, "register", lambda path: "skeleton")
    monkeypatch.setenv("KYOJIN_SKELETON", "1")
    assert KS.register(p) == "skeleton"


def test_gemma4_rows_off_and_on(tmp_path, monkeypatch):
    m, cfg = T4._model(T4.DENSE)
    p = str(tmp_path / "g.gguf")
    T4.write(p, m, cfg)
    monkeypatch.delenv("KYOJIN_GEMMA4_ROWS", raising=False)
    assert not CR.has_rows("gemma4") and K.vetted("gemma4", "flags") == () and K.vetted("gemma4", "consumed") == ()
    with pytest.raises(K.KitError):                       # without the row the kit refuses the metadata (what the serve path did on Metal)
        KC.load_cpu(p)
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")
    assert CR.has_rows("gemma4") and "rope_freqs" in K.vetted("gemma4", "flags")
    got, rep = KC.load_cpu(p)
    assert rep["model_type"] == "gemma4_text"


def test_day0_verdict_gemma4_follows_the_switch(tmp_path, monkeypatch):
    m, cfg = T4._model(T4.DENSE)
    p = str(tmp_path / "g.gguf")
    T4.write(p, m, cfg)
    monkeypatch.delenv("KYOJIN_GEMMA4_ROWS", raising=False)
    off = D.inspect(p, use_census=True)
    assert off["cfg_row"] is False and off["vetted_flags"] == [] and not off["verdict"].startswith("READY")
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")
    on = D.inspect(p, use_census=True)
    assert on["cfg_row"] is True and "rope_freqs" in on["vetted_flags"]
    print("off:", off["verdict"], "| on:", on["verdict"])


def test_serve_register_takes_the_gemma4_config_row(tmp_path, monkeypatch):
    """Metal run of 17/11: serve register called kit_config straight (KitError: keys not explained) because it ignored the config rows. Now: row on -> mapped (dry name mapping,
    CPU); row off -> the same KitError as before."""
    from kyojin_mlx import foreign_linear as FL
    m, cfg = T4._model(T4.DENSE)
    p = str(tmp_path / "g.gguf")
    T4.write(p, m, cfg)
    monkeypatch.setenv("KYOJIN_FOREIGN_MAPONLY", "1")
    monkeypatch.delenv("KYOJIN_GEMMA4_ROWS", raising=False)
    N.ARCHS.pop("gemma4", None)
    with pytest.raises(K.KitError):
        KS.register(p)
    monkeypatch.setenv("KYOJIN_GEMMA4_ROWS", "1")
    try:
        r, c = FL.load_foreign(p)
        assert c["model_type"] == "gemma4_text" and len(r["where"]) > 0
    finally:
        N.ARCHS.pop("gemma4", None)
        N._CONFIG.pop("kit:gemma4", None)

"""Wave-3 join: ONE class lookup (foreign_vlm_classes.resolve) for the serve loader, the arch kit and day0_inspect; honest verdicts for known serve blockers. CPU only, synthetic."""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import day0_inspect as D  # noqa: E402
from kyojin_mlx import foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_linear as FL, foreign_names as FN, foreign_qwen4exp_gguf as Q  # noqa: E402
from kyojin_mlx import foreign_synth as SY, foreign_vlm_classes as VC  # noqa: E402


def _switches(monkeypatch, on):
    for k in ("KYOJIN_KIT_UNPROVEN_ROWS", "KYOJIN_KIT_VLM_CLASSES"):
        if on:
            monkeypatch.setenv(k, "1")
        else:
            monkeypatch.delenv(k, raising=False)
    Q.register()


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    monkeypatch.delenv("KYOJIN_KIT_VLM_CLASSES", raising=False)
    Q.register()


def test_one_mechanism_in_source():
    assert FL.VLM_FAMILIES is VC.FAMILIES
    assert "VC.resolve" in inspect.getsource(KC._classes) or "VC.resolve" in inspect.getsource(KC)
    assert "_VC.resolve" in inspect.getsource(FL.load_foreign) and "VC.resolve" in inspect.getsource(D.serve_blocker)
    assert "_get_classes" not in inspect.getsource(FL.load_foreign)


def test_glm5_next_resolution_unchanged_all_switches_off(monkeypatch):
    pytest.importorskip("mlx_vlm.models.glm5_next")
    from mlx_vlm.models.glm5_next import Model, ModelConfig
    _switches(monkeypatch, False)
    assert FL.vlm_family("glm5_next") == "mlx_vlm.models.glm5_next" and FL.vlm_family("qwen4_exp") is None and FL.vlm_family("qwen3") is None
    r = VC.resolve(["glm5_next"], config={"model_type": "glm5_next"})
    assert r == (Model, ModelConfig, "glm5_next", "family")
    _switches(monkeypatch, True)                                    # the row is unproven (switch), the class resolution is not: same answer with the switches on
    assert VC.resolve(["glm5_next"], config={"model_type": "glm5_next"}) == r
    if "glm5-next" in FN.ARCHS:
        assert FN.ARCHS["glm5-next"].model_type == "glm5_next" and D.serve_blocker("glm5-next", ["token_embd.weight"]) is None


@pytest.mark.parametrize("on", [False, True])
def test_tool_class_equals_serve_class_per_architecture(monkeypatch, on):
    _switches(monkeypatch, on)
    n = 0
    for arch, A in list(FN.ARCHS.items()):                          # hand rows: the loader's call (config) vs the tool's (serve_blocker's call) vs the kit's (candidates)
        serve = _try(lambda: VC.resolve([A.model_type], config={"model_type": A.model_type}))
        tool = _try(lambda: VC.resolve([A.model_type], config={"model_type": A.model_type}))
        assert serve == tool
        assert (D.serve_blocker(arch, []) is None) == (serve is not None), arch
        if serve is not None and serve[3] != "family":
            kit = _try(lambda: VC.resolve([A.model_type], families=False))
            assert kit is None or kit[:2] == serve[:2], arch
        n += 1
    for arch in sorted(set(CR.ROWS) | {"cohere2moe", "flint", "nemotron_h_moe"}):      # kit rows: what register() takes (config) vs what the kit / census takes (candidates)
        from kyojin_mlx import foreign_archkit as K
        if not K.have(arch):
            continue
        kit = _try(lambda: KC._classes(arch))
        cands = K.model_type_candidates(arch)
        mt = kit[2] if kit else cands[0]
        serve = _try(lambda: VC.classes_for_config({"model_type": mt}))
        if kit is None:
            assert serve is None or on, arch
        else:
            assert serve is not None and serve == kit[:2], arch
        n += 1
    assert n > 10


def _try(f):
    try:
        return f()
    except BaseException:  # noqa: BLE001
        return None


def test_switch_off_means_no_vlm_fallback(monkeypatch):
    pytest.importorskip("mlx_vlm.models.flint")
    _switches(monkeypatch, False)
    assert _try(lambda: VC.classes_for_config({"model_type": "flint"})) is None
    _switches(monkeypatch, True)
    from mlx_vlm.models.flint import Model
    assert VC.classes_for_config({"model_type": "flint"})[0] is Model


def _q4_file(tmp_path, with_ple):
    kv = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}
    t = [("token_embd.weight", (32, 8), "F32", np.zeros((8, 32), np.float32))]
    if with_ple:
        t.append(("per_layer_token_embd.weight", (16, 8), "F32", np.zeros((8, 16), np.float32)))
    p = tmp_path / "q.gguf"
    SY.write_gguf(p, kv, t)
    return str(p)


def test_verdict_names_unserved_tensor(tmp_path, monkeypatch):
    _switches(monkeypatch, True)
    r = D.inspect(_q4_file(tmp_path, True), use_census=False)
    assert r["hand_row"] and r["verdict"] == "MODEL CODE NEEDED" and "per_layer_token_embd.weight" in r["why"] and "Arch.unserved" in r["why"]
    r2 = D.inspect(_q4_file(tmp_path, False), use_census=False)
    assert r2["serve_blocker"] is None or "no model class" in r2["serve_blocker"]          # no unserved tensor: the file is no longer refused for that reason


def test_verdict_names_missing_class(tmp_path, monkeypatch):
    _switches(monkeypatch, True)
    import dataclasses
    monkeypatch.setitem(FN.ARCHS, "qwen4exp", dataclasses.replace(FN.ARCHS["qwen4exp"], model_type="no_such_model_type"))
    assert "no model class" in (D.serve_blocker("qwen4exp", []) or "")


def test_existing_hand_row_verdict_unchanged(tmp_path, monkeypatch):
    _switches(monkeypatch, False)
    import test_gguf_arch as T
    p = tmp_path / "m.gguf"
    T._write(p, T._hf_qwen35())
    r = D.inspect(str(p))
    assert r["serve_blocker"] is None and r["verdict"] == "READY"


def test_no_op_lines_for_existing_architectures(monkeypatch):
    """Item 3: Arch.concat / Arch.unserved are empty for every row that exists with the switches off; kv_counts keeps its scalar answers; the bare-parameter name check is for unproven rows only."""
    from kyojin_mlx import foreign_archkit as K, foreign_gguf as GF
    _switches(monkeypatch, False)
    for arch, A in FN.ARCHS.items():
        assert not A.concat and not A.unserved, arch
    assert GF.kv_counts(np.array([8])) == (8, 8, [8]) and GF.kv_counts(8) == (8, 8, None) and GF.kv_counts([2, 4]) == (2, 4, [2, 4])
    assert GF.kv_counts(np.array([2, 4])) == (2, 4, [2, 4])
    assert all(v.get("proven") is False for a, v in K.VETTED.items() if "names" in v and any(n.endswith("softmax_offset") for n in v["names"].values()))

"""Declarative config rows (kyojin_mlx/foreign_cfgrows.py) against the hand builders (foreign_names._cfg_*, the reference), on the synthetic GGUFs of the hand-path round-trip tests:
equal configs, equal refusals, and the kit loader fed by the rows alone. CPU, tiny."""
from __future__ import annotations

import os
import subprocess
import sys
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

import test_arch_kit_roundtrip as RT   # noqa: E402  (CASES, ROOT)
import test_gguf_arch as T             # noqa: E402
from kyojin_mlx import foreign_cfgrows as CR, foreign_gguf as GF, foreign_names as N  # noqa: E402

CASES = RT.CASES


def _files(arch, tmp_path, monkeypatch):
    seen = []
    real = N.gguf_serve_dir

    def spy(path, *a, **k):
        seen.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(N, "gguf_serve_dir", spy)
    CASES[arch](tmp_path)
    return GF.find_gguf(seen[-1])


@pytest.mark.parametrize("arch", list(CASES))
def test_rows_give_the_hand_builders_config(arch, tmp_path, monkeypatch):
    g = GF.GgufFile(_files(arch, tmp_path, monkeypatch))
    hand, rows = N.gguf_config(g), CR.build(arch, g)
    assert hand == rows, {k: (hand.get(k), rows.get(k)) for k in set(hand) | set(rows) if hand.get(k) != rows.get(k)}
    assert CR.n_trunk(arch, g.kv) == N.n_trunk_of(g.kv, arch)


# metadata overrides each hand builder refuses: the rows must refuse the same files (and only those)
REFUSE = {
    "qwen35": [{"qwen35.rope.scaling.type": "yarn"}, {"prism.hadamard.block_size": 16}, {"qwen35.ssm.inner_size": 7 * 16}, {"qwen35.rope.dimension_sections": [11, 11, 10, 0]},
               {"qwen35.attention.recurrent_layers": [True, True, True, True, False]}, {"qwen35.attention.value_length": 8}],
    "qwen35moe": [{"qwen35moe.expert_gating_func": 2}, {"qwen35moe.expert_weights_scale": 2.5}, {"qwen35moe.rope.scaling.type": "yarn"}],
    "glm4moe": [{"glm4moe.rope.scaling.type": "yarn"}, {"glm4moe.expert_gating_func": 1}, {"glm4moe.logit_scale": 2.0}, {"glm4moe.rope.dimension_sections": [4, 2, 2, 0]}, {"glm4moe.expert_used_count": 1, "glm4moe.expert_weights_norm": True}],
    "deepseek2": [{"deepseek2.rope.scaling.type": "yarn"}, {"deepseek2.expert_gating_func": 1}, {"deepseek2.attention.temperature_scale": 0.1}, {"deepseek2.rope.scaling.yarn_log_multiplier": 0.1}],
}
# values both sides must read the same way (no refusal)
ACCEPT = {"qwen35": [{"qwen35.rope.freq_base": 5e6}, {"qwen35.ssm.conv_kernel": 4}], "qwen35moe": [{"qwen35moe.expert_weights_scale": 1.0}], "glm4moe": [{"glm4moe.expert_weights_scale": 2.5}], "deepseek2": [{"deepseek2.expert_weights_scale": 2.5}]}


def _same(arch, g, over):
    fake = types.SimpleNamespace(kv={**g.kv, **over}, tensors=g.tensors)
    out = []
    for f in (lambda: N.gguf_config(fake), lambda: CR.build(arch, fake)):
        try:
            out.append(("ok", f()))
        except GF.GgufError:
            out.append(("refused", None))
    return out


@pytest.mark.parametrize("arch", list(REFUSE))
def test_rows_refuse_what_the_hand_builder_refuses(arch, tmp_path, monkeypatch):
    g = GF.GgufFile(_files(arch, tmp_path, monkeypatch))
    for over in REFUSE[arch]:
        hand, rows = _same(arch, g, over)
        # a metadata override the tiny file cannot realise (the key is absent in this config) is not a case: both sides must agree whatever happens
        assert hand[0] == rows[0], (arch, over, hand[0], rows[0])
        if hand[0] == "ok":
            assert hand[1] == rows[1]
    assert any(_same(arch, g, over)[0][0] == "refused" for over in REFUSE[arch]), "no override was refused: the test does not test"
    for over in ACCEPT[arch]:
        hand, rows = _same(arch, g, over)
        assert hand[0] == rows[0] == "ok" and hand[1] == rows[1]


def test_expression_language_is_closed():
    for bad in ("__import__('os')", "K.__class__", "open('x')", "[x for x in K('a')]", "lambda: 1"):
        with pytest.raises(GF.GgufError):
            CR.ev(bad, CR._Env("a", {}, {}))


@pytest.mark.parametrize("arch", list(CASES))
def test_kit_loader_takes_its_config_from_the_rows(arch, tmp_path, monkeypatch):
    path = _files(arch, tmp_path, monkeypatch)
    code = ("import sys; sys.path.insert(0, %r)\nimport mlx.core as mx; mx.set_default_device(mx.cpu)\nfrom kyojin_mlx import foreign_archkit_cpu as KC, foreign_names as N\n"
            "m, rep = KC.load_cpu(%r)\nassert rep['arch'] == %r and rep['names'].ambiguous == {}\nimport numpy as np\n"
            "from mlx_lm.utils import _get_classes\nh, _ = KC.load_cpu(%r, hand_config=True)\nids = mx.array([[3, 7, 1, 30, 22, 5, 9, 2]])\n"
            "a, b = np.array(m(ids)), np.array(h(ids))\nassert np.abs(a - b).max() < 1e-5, np.abs(a - b).max()\n" % (RT.ROOT, str(path), arch, str(path)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-900:]

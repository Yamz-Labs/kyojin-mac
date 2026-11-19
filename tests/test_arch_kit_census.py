"""The census tool (tools/archkit_census.py) on a few architectures: class lookup, strict verdicts. CPU, tiny configs, no weights."""
import os
import sys

import pytest

pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import archkit_census as C  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return {r["arch"]: r for r in C.run({"olmo2", "qwen3", "gemma2", "gpt2", "llama", "deepseek2"})}


def test_zero_line_architectures_resolve_every_parameter(rows):
    for a in ("olmo2", "qwen3"):
        r = rows[a]
        assert r["verdict"] == "zero-line" and r["params"][0] == r["params"][1] and not r["req"] and not r["flags"] and not r["miss"] and not r["unmapped"], r


def test_flagged_or_hand_vetted_architectures_are_not_zero_line(rows):
    assert rows["gemma2"]["verdict"] == "needs vetted lines" and rows["gemma2"]["n_min"] == 3     # the 3 lines that exist today
    assert rows["llama"]["verdict"] == "needs vetted lines" and rows["deepseek2"]["verdict"] == "needs vetted lines"


def test_a_naming_difference_is_not_a_resolution(rows):
    r = rows["gpt2"]
    assert r["params"][0] == 0 and r["verdict"] != "zero-line" and r["req"]


def test_markdown_has_totals_on_top(rows):
    text = C.md(list(rows.values()))
    assert text.index("## Totals") < text.index("| architecture |")


def test_name_aliases_find_the_class_the_models_config_declares():
    import mlx_lm.models as pl
    import mlx_vlm.models as pv
    lm, vl = C.modules(pl), C.modules(pv)
    for arch, (src, mod) in {"minimax-m3": ("mlx_lm", "minimax_m3_vl"), "nemotron_h_moe": ("mlx_lm", "nemotron_h")}.items():
        assert (src, mod) in [c[:2] for c in C.find_classes(arch, lm, vl)], arch
    assert C.find_classes("beam", lm, vl) == []        # no family module of that name in either package: still no class


@pytest.fixture(autouse=True)
def _unproven_rows_on(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")       # these rows have no Metal proof yet: in force only behind the one switch (tests/test_arch_kit_unproven.py covers off)

"""tools/ced_gate.py on the tiny random DS4.1 model (loader injected: no pack on disk)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
import ced_gate as G  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402


class FakeModel:
    def __init__(self, lm):
        self.language_model = lm


def loader_for(**over):
    def load(model_dir, overrides):
        lm, cfg = M.make(**over)
        return FakeModel(lm), DS41, dict(vocab_size=M.CFG["vocab_size"]), None
    return load


def run(args, **over):
    with pytest.raises(SystemExit) as e:
        G.main(["mini", "--no-needle", *args], loader=loader_for(**over))
    return e.value.code


def test_window_one_passes_with_exact_tail(tmp_path, capsys):
    out = tmp_path / "gate.json"
    code = run(["--lengths", "96,160", "--chunk", "64", "--tails", "32", "--new", "8", "--dump", str(out)], sliding_window=1)
    txt = capsys.readouterr().out
    j = json.load(open(out))
    assert code == 0 and "VERDICT PASS" in txt, txt[-800:]
    assert all(r["kl_last"] <= r["kl_bound"] and r["answer_prefix"] >= 6 and r["rows_cut"] > 0 for r in j["rows"])


def test_big_tail_equals_exact_and_small_tail_is_measured(tmp_path, capsys):
    out = tmp_path / "gate.json"
    code = run(["--lengths", "128", "--chunk", "128", "--tails", "exact,16", "--new", "6", "--dump", str(out)], sliding_window=16)
    j = json.load(open(out))
    txt = capsys.readouterr().out
    exact = [r for r in j["rows"] if r["tail_arg"] == "exact"][0]
    assert exact["tail"] == 76 and exact["kl_last"] < 1e-6 and exact["top1"] == exact["top1_rows"]
    assert exact["floor_kl_last"] is not None
    assert "VERDICT" in txt


def test_failing_gate_exits_one(tmp_path, capsys):
    code = run(["--lengths", "256", "--chunk", "128", "--tails", "16", "--new", "24", "--kl-max", "1e-12", "--no-floor"], sliding_window=16)
    assert code == 1 and "VERDICT FAIL" in capsys.readouterr().out

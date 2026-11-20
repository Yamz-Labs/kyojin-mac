# SPDX-License-Identifier: Apache-2.0
"""Fires log: the periodic `[flags 60s]` serve line carries the lever totals (qdec_fast, absorb_gemv, fixed_gather); KYOJIN_HY4_FIRES_LOG=0 drops them. Pure python, no Metal."""
import sys
import time
import types

import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import fired as FD  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(FD, "AUDIT", True)
    monkeypatch.setattr(FD, "LOG", False)
    monkeypatch.setattr(FD, "FIRES", True)
    monkeypatch.setattr(FD, "_sink", None)
    FD.reset()
    yield
    FD.reset()


def test_gather_skip_reasons():
    assert MLA.gather_skip(0, 99999, False) == "off"
    assert MLA.gather_skip(8192, 4096, False) == "below_min"
    assert MLA.gather_skip(8192, 65536, True) == "fp16_copy"
    assert MLA.gather_skip(8192, 4096, True) == "below_min"


def test_totals_read_loaded_modules_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "kyojin_mlx.foreign_metal", types.SimpleNamespace(PATHS={"qdec_fast": 7}))
    monkeypatch.setitem(MLA.MlaAttention.PATHS, "absorb_gemv", 3)
    monkeypatch.setitem(MLA.MlaAttention.PATHS, "fixed_gather", 5)
    assert FD.totals() == {"qdec_fast": 7, "absorb_gemv": 3, "fixed_gather": 5}
    monkeypatch.delitem(sys.modules, "kyojin_mlx.foreign_metal")
    assert "qdec_fast" not in FD.totals()


def test_format_line_and_switch(monkeypatch):
    FD.fire("hy4_gather")
    FD.fb("hy4_gather", "below_min")
    FD.fb("hy4_gather", "below_min")
    line = FD.format_line(60, FD.summary(), {"qdec_fast": 7, "absorb_gemv": 3, "fixed_gather": 1})
    assert line == "[flags 60s] hy4_gather 1/2 (below_min:2) | totals qdec_fast=7 absorb_gemv=3 fixed_gather=1"
    monkeypatch.setattr(FD, "FIRES", False)
    assert FD.format_line(60, "x 1/0", {"qdec_fast": 7}) == "[flags 60s] x 1/0"


def test_reporter_emits_early_line_after_first_activity(monkeypatch):
    monkeypatch.setitem(MLA.MlaAttention.PATHS, "fixed_gather", 0)
    lines = []
    FD.start_reporter(lines.append, period=60.0)
    time.sleep(0.2)
    assert lines == []
    FD.fire("hy4_gather")
    monkeypatch.setitem(MLA.MlaAttention.PATHS, "fixed_gather", 4)
    deadline = time.time() + 10
    while not lines and time.time() < deadline:
        time.sleep(0.2)
    assert lines and lines[0].startswith("[flags 60s] hy4_gather 1/0") and "fixed_gather=4" in lines[0], lines

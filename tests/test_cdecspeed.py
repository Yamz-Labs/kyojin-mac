# SPDX-License-Identifier: Apache-2.0
"""tools/cdecspeed_mac.py on a CPU: the lever table against the flag table, the comparison it reports, the refusals and the no-Metal exit.
The Metal run (logits per lever, ms per step, kept bytes) is for the Mac; nothing here is evidence about a lever's speed or bits."""
import importlib.util
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def tool():
    spec = importlib.util.spec_from_file_location("cdecspeed_mac", os.path.join(ROOT, "tools", "cdecspeed_mac.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_lever_of_the_table_is_a_runtime_flag_and_its_value_validates():
    from kyojin_mlx import flags as FL
    t = tool()
    ok, gone = t.existing(t.ARMS, FL)
    assert gone == [] and set(ok) == set(t.ARMS)
    before = {n: FL.value(n) for n in t.ARMS}
    try:
        for n, v in t.ARMS.items():
            FL.set_many({n: v})
            assert FL.value(n) == str(v)
    finally:
        FL.set_many(before)


def test_the_hidden_stream_kernels_stay_on_their_shipped_defaults_in_every_arm():
    t = tool()
    assert t.BASE_FLAGS == {"kvchain": 1, "ropefuse": 1, "rope_fma": 3}
    assert not set(t.BASE_FLAGS) & set(t.ARMS)


def test_compare_reports_first_position_vocab_index_max_diff_and_argmax_flips():
    t = tool()
    a = np.zeros((5, 8), np.float32)
    a[:, 3] = 1.0
    assert t.compare(a, a.copy()) == (None, None, 0.0, 0, 0)
    b = a.copy()
    b[2, 6] = 0.5
    b[4, 3] = -1.0
    pos, vi, mad, npos, flips = t.compare(a, b)
    assert (pos, vi, npos, flips) == (2, 6, 2, 1) and mad == 2.0


def run(argv):
    old = sys.argv
    sys.argv = ["cdecspeed_mac.py"] + argv
    try:
        return tool().main()
    finally:
        sys.argv = old


def test_cap_above_3_5_gib_is_refused_before_anything_loads(capsys):
    assert run(["--max-gib", "4"]) == 3
    assert "refused" in capsys.readouterr().out


def test_without_metal_the_tool_exits_2(capsys):
    if mx.metal.is_available():
        pytest.skip("this machine has Metal")
    assert run([]) == 2
    assert "Metal not available" in capsys.readouterr().out

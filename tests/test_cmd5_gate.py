# SPDX-License-Identifier: Apache-2.0
"""tools/cmd5_identity_gate.py on a CPU: the comparison it reports, the arm table, the flag switches it uses, and the no-Metal exit. The Metal run is for the Mac."""
import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def gate():
    spec = importlib.util.spec_from_file_location("cmd5_gate", os.path.join(ROOT, "tools", "cmd5_identity_gate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_compare_reports_the_first_differing_position_index_and_flips():
    g = gate()
    a = np.zeros((5, 8), np.float32)
    a[:, 3] = 1.0
    assert g.compare(a, a.copy()) == (None, None, 0.0, 0, 0)
    b = a.copy()
    b[2, 6] = 0.5            # position 2, vocab 6, no argmax change
    b[4, 3] = -1.0           # position 4: argmax moves off index 3
    pos, vi, mad, npos, flips = g.compare(a, b)
    assert (pos, vi, npos, flips) == (2, 6, 2, 1) and mad == 2.0


def test_the_flag_switches_of_the_arms_exist_and_round_trip():
    from kyojin_mlx import flags
    g = gate()
    before = {n: flags.get_all()[n]["value"] for n in ("kvchain", "ropefuse", "rope_fma")}
    try:
        for fl in [g.BASE] + [a | {"rope_fma": f} for a in g.ARMS.values() for f in (0, 1, 3)]:
            flags.set_many(dict(g.BASE) | fl)
            from kyojin_mlx import dsv4_core as DC
            assert DC.KV_CHAIN == bool(fl.get("kvchain", 0)) and DC.ROPE_FUSE == bool(fl.get("ropefuse", 0))
            assert DC.ROPE_FMA in (0, 1, 2) and (fl.get("rope_fma", 0) == 0) <= (DC.ROPE_FMA == 0)       # 3 resolves to the measured form (0 on the CPU)
    finally:
        flags.set_many(before)


def test_without_metal_the_gate_exits_2(capsys):
    if mx.metal.is_available():
        pytest.skip("this machine has Metal")
    import sys
    old = sys.argv
    sys.argv = ["cmd5_identity_gate.py"]
    try:
        assert gate().main() == 2
    finally:
        sys.argv = old
    assert "Metal not available" in capsys.readouterr().out

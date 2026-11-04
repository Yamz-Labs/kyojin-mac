"""KYOJIN_DEC_IDX_FUSE (fused indexer rotary + fp4, dsv4_core.ROT_FQ4) and KYOJIN_IDX_FUSE (fused indexer select, dsv4_core.IDX_FUSE) are separate switches
(both used to assign IDX_FUSE, the second won and the first env was dead)."""
import subprocess
import sys

import pytest

pytest.importorskip("mlx.core")

CODE = "import kyojin_mlx.dsv4_core as C; print(C.ROT_FQ4, C.IDX_FUSE)"


def run(**env):
    base = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", **env}
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=base, timeout=120)
    return r.stdout.split() if r.returncode == 0 else r.stderr[-300:]


def test_the_two_switches_are_independent():
    assert run() == ["True", "True"]
    assert run(KYOJIN_DEC_IDX_FUSE="0") == ["False", "True"]
    assert run(KYOJIN_IDX_FUSE="0") == ["True", "False"]

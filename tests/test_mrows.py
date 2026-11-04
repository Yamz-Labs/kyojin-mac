"""Batched-core verify (rows.verify): several rows per session in one pass of the batched DS4.1 core equal the plain decode steps bit for bit,
and a commit leaves the state of the plain run. Needs a Mac (Metal) and the 2-layer DS4.1 slices (KYOJIN_SPEC_PACKS, see test_spec.py). Each case
runs tools/mrows_check.py, whose negative controls (rows see later rows, no rollback, wrong pool lengths) must fail."""
import os
import subprocess
import sys

import pytest

mx = pytest.importorskip("mlx.core")
from test_spec import CASES, pack_dir                                         # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = [("ds2m", [])]                                                        # ratio 2, window 128, 32 experts: the serving slice


@pytest.fixture(autouse=True)
def _solo_bits(monkeypatch):
    """These tests assert that a row keeps its solo bits: pin the NAX rows switch (ON by default, a rounding class) off."""
    from kyojin_mlx import exl3_linear as EL
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    monkeypatch.setenv("KYOJIN_ROWS_NAX", "0")                      # and in the subprocesses of the checks


def run_check(pack, *args):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "mrows_check.py"), pack, "--quiet", *args]
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("name,over", EXTRA + CASES, ids=[f"{n}-{'-'.join(o)[:40]}" for n, o in EXTRA + CASES])
def test_batched_verify_exact(name, over):
    r = run_check(pack_dir(name), "--steps", "30", *(["--set", *over] if over else []))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "EXACT" in r.stdout


@pytest.mark.parametrize("neg", ["fullring", "norollback", "nolens"])
def test_negative_controls_fail(neg):
    r = run_check(pack_dir("ds2m"), "--steps", "24", "--neg", neg)
    assert r.returncode == 1 and "MISMATCH" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def test_batched_verify_exact_with_parallel_engram_gather():
    r = run_check(pack_dir("engram_pack"), "--steps", "30", "--parallel-gather")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "EXACT" in r.stdout

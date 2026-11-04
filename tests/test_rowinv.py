"""Row-invariant dense products (rowinv.rowdot) and the multi-user batched step on mlx-vlm DS4.1 / GLM slices.
The model test needs packs: KYOJIN_MU_PACKS=<dir>[,<dir>...] (e.g. the 2-layer ds2 / glm2 slices); skipped otherwise. Needs a Mac (Metal)."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import rowinv as RI                                       # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def bits(a):
    a = np.array(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint16)


@pytest.mark.parametrize("K,N", [(2048, 16), (5120, 32), (20480, 24), (16384, 24), (1000, 24), (4096, 100)])
@pytest.mark.parametrize("xd,wd", [(np.float16, np.float16), (np.float32, np.float32), (np.float16, np.float32)])
def test_rowdot_row_invariant_and_accurate(K, N, xd, wd):
    rng = np.random.default_rng(K + N)
    w = mx.array((rng.standard_normal((N, K)) * 0.05).astype(wd))
    x = mx.array(rng.standard_normal((11, K)).astype(xd))
    one = [np.array(RI.rowdot(x[i:i + 1], w)) for i in range(11)]
    for B in (1, 2, 3, 4, 8, 11):                                         # 11 > ROWDOT_MAX: split launches
        y = RI.rowdot(x[:B], w)
        for i in range(B):
            assert np.array_equal(bits(y)[i], bits(one[i])[0]), (B, i)
    ref = np.array(x, dtype=np.float64) @ np.array(w, dtype=np.float64).T
    got = np.array(RI.rowdot(x[:4], w), dtype=np.float64)
    tol = 2e-3 if (xd == np.float16 and wd == np.float16) else 1e-5
    assert np.abs(got - ref[:4]).max() <= tol * np.abs(ref[:4]).max() + 1e-4


def test_rowdot_output_dtype():
    x16, w16 = mx.ones((2, 64), mx.float16), mx.ones((8, 64), mx.float16)
    assert RI.rowdot(x16, w16).dtype == mx.float16
    assert RI.rowdot(x16.astype(mx.float32), w16).dtype == mx.float32
    assert RI.rowdot(x16, w16.astype(mx.float32)).dtype == mx.float32


def test_rowmm_falls_back_for_prompts():
    x = mx.ones((1, 5, 64), mx.float16)                                   # S > 1: stock matmul
    w = mx.ones((8, 64), mx.float16)
    assert np.array_equal(np.array(RI.rowmm(x, w)), np.array(x @ w.T))


@pytest.mark.skipif(not os.environ.get("KYOJIN_MU_PACKS"), reason="needs KYOJIN_MU_PACKS")
@pytest.mark.parametrize("pack", [p for p in os.environ.get("KYOJIN_MU_PACKS", "").split(",") if p])
def test_batched_step_bitwise_equal_to_solo(pack):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "tools", "mu_check.py"), pack, "--users", "4", "--steps", "10"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = [l for l in r.stdout.splitlines() if l.startswith("N=")]
    assert len(lines) == 4 and all("equal to solo" in l and ": True; greedy tokens equal: True" in l for l in lines), r.stdout

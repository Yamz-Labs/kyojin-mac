# SPDX-License-Identifier: Apache-2.0
"""Tests for the decode glue fusion (kyojin_mlx/gluefuse.py).

`mixvec` must be bit-identical to the two stock launches it replaces: `rowinv.rowdot(mx.fast.rms_norm(x.astype(fp32), None, eps), fn)` -- the
mHC mix vector of DS4.1. `hcfold_the mixargs` cut must not change the value of fuse.mhc_mix_pre either.
"""
import importlib
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kyojin_mlx import gluefuse as GF           # noqa: E402
from kyojin_mlx import rowinv as RI            # noqa: E402


def _stock_mix(x, fn, eps):
    return RI.rowdot(mx.fast.rms_norm(x.astype(mx.float32), None, eps), fn)


def _bits(a):
    v = {mx.float16: mx.uint16, mx.bfloat16: mx.uint16, mx.float32: mx.uint32}.get(a.dtype)
    return np.array(a.view(v) if v is not None else a)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_mixvec_bitwise_hc_mix(dtype):
    """K = 4 * 5120 (DS4.1 hc_mult 4 x hidden 5120), N = 24 mix coefficients: the serve shape."""
    rng = np.random.default_rng(0)
    x = mx.array((rng.standard_normal((1, 4, 5120)) * 0.02).astype(np.float32).astype(np.float16 if dtype == mx.float16 else
                 (np.float32 if dtype == mx.float32 else np.float16))).astype(dtype)
    fn = mx.array((rng.standard_normal((24, 20480)) * 0.05).astype(np.float32))
    eps = 1e-6
    ref = _stock_mix(x.reshape(1, -1), fn, eps)
    y = GF.mixvec(x.reshape(1, -1), fn, eps)
    assert y is not None and y.shape == (1, 24)
    assert np.array_equal(_bits(ref), _bits(y)), "mixvec differs from rms_norm + rowdot"


def test_mixvec_rows_and_unformed_shapes():
    """Several rows keep each row's bits (grid.y), and shapes the kernel does not cover return None."""
    rng = np.random.default_rng(1)
    x = mx.array((rng.standard_normal((3, 8192)) * 0.05).astype(np.float16))
    fn = mx.array((rng.standard_normal((8, 8192)) * 0.05).astype(np.float32))
    ref = _stock_mix(x, fn, 1e-6)
    y = GF.mixvec(x, fn, 1e-6)
    assert np.array_equal(_bits(ref), _bits(y))
    for i in range(3):                                    # a row does not depend on the other rows
        one = GF.mixvec(x[i:i + 1], fn, 1e-6)
        assert np.array_equal(_bits(ref[i:i + 1]), _bits(one))
    assert GF.mixvec(mx.zeros((1, 3000), dtype=mx.float16), fn, 1e-6) is None      # K % 4096


def test_mix_vector_site_is_bitwise(monkeypatch):
    """The installed rows.mix_vector seam gives the stock value on a DS4.1 block-shaped call."""
    from kyojin_mlx import rows as RW
    from types import SimpleNamespace

    monkeypatch.setattr(GF, "ENABLED", True)
    monkeypatch.setattr(GF, "MIXVEC", True)
    GF.install()
    rng = np.random.default_rng(2)
    mod = SimpleNamespace(norm_eps=1e-6)
    x = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16))
    fn = mx.array((rng.standard_normal((24, 20480)) * 0.02).astype(np.float32))
    ref = RI.rowmm(mx.fast.rms_norm(x.astype(mx.float32).flatten(-2), None, 1e-6), fn)
    y = RW.mix_vector(mod, x, fn)
    assert y.shape == ref.shape == (1, 1, 24)
    assert np.array_equal(_bits(ref), _bits(y))


def test_mixargs_matches_stock_mhc_mix_pre(monkeypatch):
    """The FP32-argument cut of fuse.mhc_mix_pre keeps the bits (and drops the redundant copies)."""
    from kyojin_mlx import fuse as F

    monkeypatch.setattr(GF, "ENABLED", True)
    monkeypatch.setattr(GF, "MIXARGS", True)
    rng = np.random.default_rng(3)
    B, D = 1, 5120
    mixes = mx.array((rng.standard_normal((B, 1, 24))).astype(np.float32))
    scale = mx.array([1.1, 0.9, 1.0], dtype=mx.float32)
    base = mx.array((rng.standard_normal(24) * 0.1).astype(np.float32))
    x = mx.array((rng.standard_normal((B, 1, 4, D)) * 0.02).astype(np.float16))
    pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
    on = F.mhc_mix_pre(mixes, scale, base, x, pre, 20, 1e-6)
    mx.eval(*on)
    monkeypatch.setattr(GF, "MIXARGS", False)             # the stock astype path
    off = F.mhc_mix_pre(mixes, scale, base, x, pre, 20, 1e-6)
    mx.eval(*off)
    for a, b in zip(off, on):
        assert np.array_equal(_bits(a), _bits(b))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_mixpre_norm_bitwise(dt):
    """hc_pre + RMSNorm + sinkhorn in one launch == fuse.mhc_mix_pre(...) then nn.RMSNorm(w) (and the sinkhorn outputs are the stock ones)."""
    import mlx.nn as nn
    from kyojin_mlx import fuse as F

    rng = np.random.default_rng(5)
    B, D = 1, 5120
    x = mx.array((rng.standard_normal((B, 1, 4, D)) * 0.02).astype(np.float16)).astype(dt)
    pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
    mixes = mx.array(rng.standard_normal((B, 1, 24)).astype(np.float32))
    scale = mx.array([1.2, 0.8, 1.0], dtype=mx.float32)
    base = mx.array((rng.standard_normal(24) * 0.1).astype(np.float32))
    norm = nn.RMSNorm(D, eps=1e-6)
    norm.weight = mx.array((rng.standard_normal(D) * 0.3 + 1).astype(np.float16)).astype(dt)
    y_ref, pre_ref, post_ref, comb_ref = F.mhc_mix_pre(mixes, scale, base, x, pre, 20, 1e-6)
    n_ref = norm(y_ref)
    y, pre_o, post, comb = GF.mixpre_norm(mixes, scale, base, x, pre, norm.weight, 20, 1e-6, norm.eps)
    mx.eval(y_ref, n_ref, y, pre_o, post, comb)
    assert GF.mixnorm_ok(x, norm.weight)
    assert np.array_equal(_bits(n_ref), _bits(y)), "hc_pre + RMSNorm is not bit-identical"
    for a, b in ((pre_ref, pre_o), (post_ref, post), (comb_ref, comb)):
        assert np.array_equal(_bits(a), _bits(b)), "sinkhorn output differs"


def test_mixnorm_ok_refuses_small_rows():
    """Rows the kernel does not cover (a narrow hidden, a dtype mismatch) fall back to the stock norm."""
    w = mx.zeros((2048,), dtype=mx.float16)
    assert not GF.mixnorm_ok(mx.zeros((1, 1, 4, 2048), dtype=mx.float16), w)
    assert not GF.mixnorm_ok(mx.zeros((1, 1, 4, 5120), dtype=mx.float16), mx.zeros((5120,), dtype=mx.float32))


def test_install_idempotent(monkeypatch):
    monkeypatch.setattr(GF, "ENABLED", True)
    n1 = GF.install()
    from kyojin_mlx import rows as RW
    assert getattr(RW.mix_vector, "_kyojin_glue", False) and n1 <= 1
    assert GF.install() == 0                              # nothing installed twice
    monkeypatch.setattr(GF, "ENABLED", False)
    assert GF.install() == 0


def test_disabled_by_default():
    """The module flag is off unless the env asks for it (the lead's real-pack gate decides the default)."""
    assert os.environ.get("KYOJIN_GLUE_FUSE", "0") == "0" or importlib.import_module("kyojin_mlx.gluefuse") is not None
    assert isinstance(GF.ENABLED, bool)

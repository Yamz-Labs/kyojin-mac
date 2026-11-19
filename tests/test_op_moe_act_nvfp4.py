"""Activation kind "situ" of the NVFP4 expert pair_glu kernels (rows, g2m, inl, omx) against stock MLX ops on the same half-rounded projections.
Synthetic packs (Metal micro test). The swiglu form (act None) keeps its source: the existing tests of test_glm_moe_e.py cover it."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import glm_moe_e as GE  # noqa: E402
from kyojin_mlx import glm_moe_rows as R  # noqa: E402
from kyojin_mlx.moe import situ_ref  # noqa: E402
import test_glm_moe_e as TG  # noqa: E402

SB, SLB = 4.0, 25.0
ACT = ("situ", SB, SLB)


def _ulp(a, b):
    return int(np.abs(np.asarray(a.view(mx.int16)).astype(np.int32) - np.asarray(b.view(mx.int16)).astype(np.int32)).max())


def _setup(T, K):
    g, u, d = TG.mk(TG.H, TG.I, 1), TG.mk(TG.H, TG.I, 2), TG.mk(TG.I, TG.H, 3)
    rng = np.random.default_rng(T * 100 + K)
    rows = [list(rng.choice(TG.E, size=K, replace=False)) for _ in range(T)]
    inds = mx.array(np.array(rows, np.uint32).reshape(-1))
    x2 = mx.array((rng.normal(size=(T, TG.H)) * 0.5).astype(np.float16))
    tab = R.group(inds, T, K) if T > 1 else None
    xs = mx.repeat(x2, K, axis=0)                                           # slot s reads token s // K
    a = R.down(g, xs, inds, tab, T, K)                                      # gate_e(x) half [S, n]: the projection with no activation
    b = R.down(u, xs, inds, tab, T, K)
    want = situ_ref(a.astype(mx.float32), b.astype(mx.float32), SB, SLB).astype(mx.float16)
    sw = R.pair_glu(g, u, x2, inds, tab, T, K, TG.LIMIT)
    mx.eval(want, sw)
    return g, u, x2, inds, tab, want, sw


@pytest.mark.parametrize("T,K", [(1, 8), (2, 8), (3, 10), (4, 8)])
def test_situ_rows(T, K):
    g, u, x2, inds, tab, want, sw = _setup(T, K)
    got = R.pair_glu(g, u, x2, inds, tab, T, K, TG.LIMIT, ACT)
    print(f"rows T {T} K {K}: max half ulp vs mlx ops {_ulp(got, want)}")
    assert _ulp(got, want) <= 2 and not mx.array_equal(got, sw)
    assert mx.array_equal(R.pair_glu(g, u, x2, inds, tab, T, K, TG.LIMIT), sw)       # act None: the swiglu kernel as before


@pytest.mark.parametrize("kind", ["g2m", "inl", "omx"])
@pytest.mark.parametrize("T,K", [(1, 8), (2, 8), (4, 10)])
def test_situ_e_kinds(kind, T, K):
    """T 4 x K 10 = 40 slots: the wide (dedupe above 32 slots) variant."""
    g, u, x2, inds, tab, want, sw = _setup(T, K)
    got = GE.pair_glu(kind, g, u, x2, inds, T, K, TG.LIMIT, act=ACT)
    print(f"{kind} T {T} K {K}: max half ulp vs mlx ops {_ulp(got, want)}")
    assert _ulp(got, want) <= 2
    base = GE.pair_glu(kind, g, u, x2, inds, T, K, TG.LIMIT)
    assert _ulp(base, sw) <= 2 and not mx.array_equal(base, got)


def test_situ_act_source():
    src = "x half((a * sg_) * b) y"
    assert R.act_source(src, None) is src
    s = R.act_source(src, ACT)
    assert "tanh" in s and "(a * sg_) * b" not in s
    with pytest.raises(AssertionError):
        R.act_source(src, ("relu", 1.0, 1.0))
    with pytest.raises(AssertionError):
        R.act_source("no activation here", ACT)

# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_KQ_RT=n (2..4): the grouped expert kernel with decode sharing (one threadgroup = n row tiles of one expert, every weight k-block decoded once for all of them).
CPU part: the source transform builds for every K-class / nvfp4 variant and the (tb, off) groups of n * MT rows cover every sorted slot exactly once. Metal part (skipped without a GPU):
RT = n gives the same bytes as RT = 1 (same instructions per row, same order), ragged experts (0 rows, 1 row, MT, MT + 1, 2 MT, 2 MT + 1 rows) and the scatter / dep variants included."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_gguf as GF, foreign_kqgmm as KQ, foreign_linear as FL, foreign_synth as SY  # noqa: E402

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


@pytest.mark.parametrize("fmt", ("q3_k", "q4_k", "q5_k", "q6_k", "iq4_xs", "nvfp4"))
@pytest.mark.parametrize("scatter,dep", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize("rt", (2, 3, 4))
def test_rt_source_builds(fmt, scatter, dep, rt, monkeypatch):
    monkeypatch.setenv("KYOJIN_KQGMM_KLOW", "1")
    one, many = KQ.source(fmt, scatter, dep, 1), KQ.source(fmt, scatter, dep, rt)
    assert "cT1" not in one and "RTT" not in one and "RTT" in many
    for i in range(1, rt):
        assert many.count(f"op.run(sA{i}, rB0, cT{i})") == 1 and many.count(f"op.run(sA{i}, rB1, cT{i})") == 1
        assert f"mb{i} + m" in many and (f"orow[mb{i} + m]" in many) == scatter
    assert f"cT{rt}" not in many
    assert many.count("threadgroup float red[WS][RC * 32];") == 1 and many.count("threadgroup_barrier") == one.count("threadgroup_barrier") + 2 * (rt - 1)        # the epilogue (2 barriers) runs once per tile


def _groups(counts, mt, rt=2):
    """numpy emulation of the kernel's tile walk: (expert, first row, rows) of the two tiles of every group."""
    off = np.concatenate([[0], np.cumsum(counts)])
    tb = np.concatenate([[0], np.cumsum((counts + rt * mt - 1) // (rt * mt))])
    cover = np.zeros(off[-1], int)
    for ti in range(tb[-1]):
        ex = np.searchsorted(tb, ti, side="right") - 1
        tin = ti - tb[ex]
        mb = off[ex] + tin * rt * mt
        rall = min(rt * mt, off[ex + 1] - mb)
        for i in range(rt):
            ri = min(mt, rall - i * mt) if rall > i * mt else 0
            cover[mb + i * mt:mb + i * mt + ri] += 1
    return cover


@pytest.mark.parametrize("rt", (2, 3, 4))
@pytest.mark.parametrize("mt", (16, 32))
def test_groups_cover_every_slot_once(mt, rt):
    rng = np.random.default_rng(7)
    counts = np.array([0, 1, mt, mt + 1, 2 * mt, 2 * mt + 1, 3 * mt, 5 * mt + 3] + list(rng.integers(0, 4 * mt, 40)))
    assert (_groups(counts, mt, rt) == 1).all()


def _stack(fmt, E, n, k, seed=0):
    rng = np.random.default_rng(seed)
    d = FL.DevPack([GF.to_pack(fmt.upper(), SY.random_blocks(fmt.upper(), n, k, rng), k) for _ in range(E)])
    mx.eval(d.arrays())
    return d


@needs_gpu
@pytest.mark.parametrize("fmt", ("q3_k", "q4_k"))
@pytest.mark.parametrize("mt", (16, 32))
@pytest.mark.parametrize("variant", ("plain", "scatter", "dep"))
@pytest.mark.parametrize("rt", (2, 4))
def test_rt_bit_equal_to_rt1(fmt, mt, variant, rt, monkeypatch):
    monkeypatch.setenv("KYOJIN_KQGMM_KLOW", "1")
    E, n, k = 16, 96, 512
    dp = _stack(fmt, E, n, k, seed=3)
    rng = np.random.default_rng(11)
    counts = np.array([0, 1, mt, mt + 1, 2 * mt, 2 * mt + 1, 3 * mt - 1, 5, 3 * mt + 1, 4 * mt, 4 * mt + 1, 5 * mt + 7] + list(rng.integers(0, 3 * mt, E - 12)))
    es = np.repeat(np.arange(E), counts).astype(np.uint32)
    S = len(es)
    xp = mx.array((rng.standard_normal((S, k)) * 0.5).astype(np.float16))
    esm = mx.array(es)
    tb1, off = KQ.plan(esm, E, mt)
    tb2, off2 = KQ.plan(esm, E, rt * mt)
    assert np.array_equal(np.array(off), np.array(off2))
    kw1, kw2 = {}, {}
    if variant == "scatter":
        orow = mx.array(rng.permutation(S).astype(np.uint32)); kw1["orow"] = kw2["orow"] = orow
    if variant == "dep":
        dep = mx.zeros((4,), mx.float16); kw1["dep"] = kw2["dep"] = dep
    y1 = KQ.gmm(dp, xp, tb1, off, S, mt, RT=1, **kw1)
    y2 = KQ.gmm(dp, xp, tb2, off, S, mt, RT=rt, **kw2)
    a, b = np.array(y1), np.array(y2)
    assert np.isfinite(a.astype(np.float32)).all() and np.abs(a.astype(np.float32)).max() > 0
    assert np.array_equal(a, b)

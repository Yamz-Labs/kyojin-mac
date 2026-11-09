"""KYOJIN_MOE_CHAIN: mid in the tail of the gate/up gather GEMV and the combine in the tail of the down gather GEMV (3 launches per layer instead of 5).
Outputs must be bit-identical to the five-launch chain, plain and with the shared-expert slot (moe_sumoe)."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import moe as M, metal_src as S  # noqa: E402
from test_moe import make_stack, _route, _x, _bits, KS, TOPK  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    c = M.CHAIN
    yield
    M.CHAIN = c


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("rows", [1, 2])
def test_bit_identical(cb, wpre, fq, rows):
    kin, ninter = 384, 640                                      # 3 and 5 blocks of 128
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    m = M.MoeGather(g, u, d, TOPK, wpre=wpre, fq=fq)
    ids, w = _route(rows, 7)
    for seed in range(3):
        x = _x(rows, kin, 10 + seed)
        M.CHAIN = False
        ref = np.array(m(x, ids, w, fq8=fq))
        M.CHAIN = True
        outs = [np.array(m(x, ids, w, fq8=fq)) for _ in range(3)]     # repeated: the arrival counters are zero again after each launch
        for o in outs:
            assert np.array_equal(_bits(o), _bits(ref))
        assert not np.any(np.array(m._chain_ctr(rows)[0])) and not np.any(np.array(m._chain_ctr(rows)[1]))


def test_launch_count_and_flag_off():
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, "mul1", 1)
    u, _ = make_stack(kin, ninter, KS, "mul1", 2)
    d, _ = make_stack(ninter, kin, KS, "mul1", 3)
    m = M.MoeGather(g, u, d, TOPK)
    M.CHAIN = False
    assert not m.chain_ok()
    M.CHAIN = True
    assert m.chain_ok()
    m.union = True
    assert not m.chain_ok()                                     # the union path keeps its own launches


def test_sources_reuse_mid_and_final_statements():
    gu = S.chain_gu_source((2, 3), "mul1", True, False, 3, 10.0, True, True, 1.0, 0.0)
    assert "threadgroups_per_grid.x" in gu and "5u" not in gu.split("chain_blocks")[1][:80]       # the grid size is read from the launch, never baked
    assert "LDF4(part" in gu and "xh2 + ulong(p) * N + col" in gu and "(2u * chain_tgk)" in gu
    dn = S.chain_dn_source((2, 3), "mul1", True, False, 2, 6, False)
    assert "ys[j][lane] = y4;" in dn and "(6u * chain_tgk)" in dn and "return;" in dn


@pytest.mark.parametrize("sfq", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_sumoe_bit_identical(sfq, rows):
    from kyojin_mlx import moe_sumoe as SU
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, "mul1", 1); u, _ = make_stack(kin, ninter, KS, "mul1", 2); d, _ = make_stack(ninter, kin, KS, "mul1", 3)
    sp = [make_stack(kin, ninter, [3], "mul1", 11)[1], make_stack(kin, ninter, [3], "mul1", 12)[1], make_stack(ninter, kin, [3], "mul1", 13)[1]]
    ext = SU.with_shared(g, u, d, [(mx.array(p[0][0]), mx.array(p[0][1]), mx.array(p[0][2])) for p in sp])
    sm = SU.MoeSharedFused(*ext, TOPK, shared_fq=sfq, limit=10.0, wpre=True, fq=True)
    ids, w = _route(rows, 4)
    x = _x(rows, kin, 5)
    M.CHAIN = False
    ref = np.array(sm(x, ids, w, fq8=True))
    M.CHAIN = True
    got = np.array(sm(x, ids, w, fq8=True))
    assert np.array_equal(_bits(got), _bits(ref))


def _zero_ctr(m, rows):
    c = m._chain_ctr(rows)
    return not np.any(np.array(c[0])) and not np.any(np.array(c[1]))


def _build(kin, ninter, cb="mul1", rows=1, **kw):
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return M.MoeGather(g, u, d, TOPK, **kw)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_shapes_interleaved_counters_zero(order):
    """Regression: kernels built for one shape must not serve another (the block count used to be baked into the cached source, so two stacks of different
    width shared counter slots and a stale 1 was left behind). Several shapes, built and run in every order, twice in a row each: counters all zero after each call."""
    shapes = [(384, 640), (256, 384), (512, 256)]
    shapes = shapes[order:] + shapes[:order]
    ms = [(sh, _build(*sh)) for sh in shapes]
    for rows in (1, 2):
        ids, w = _route(rows, 7)
        for rep in range(2):
            for (kin, ninter), m in ms[::-1] if rep else ms:
                x = _x(rows, kin, 3)
                M.CHAIN = False
                ref = np.array(m(x, ids, w))
                M.CHAIN = True
                for _ in range(2):
                    got = np.array(m(x, ids, w))
                    assert np.array_equal(_bits(got), _bits(ref))
                    assert _zero_ctr(m, rows)


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("rows", [1, 2])
def test_every_variant_twice_counters_zero(cb, rows):
    """Every chain variant (plain, wpre/fq, pinned rows, forced tgk, shared slot) twice in a row: counter buffers zero after each call."""
    kin, ninter = 384, 640
    ids, w = _route(rows, 7)
    for kw in (dict(), dict(wpre=True, fq=True), dict(tgk_gu=2, tgk_dn=2), dict(tgk_gu=4, tgk_dn=1)):
        tg = {k: kw.pop(k) for k in ("tgk_gu", "tgk_dn") if k in kw}
        m = _build(kin, ninter, cb, **kw)
        for k, v in tg.items():
            setattr(m, k, v)
        x = _x(rows, kin, 5)
        M.CHAIN = False
        ref = np.array(m(x, ids, w, fq8=kw.get("fq", False)))
        M.CHAIN = True
        for _ in range(2):
            assert np.array_equal(_bits(np.array(m(x, ids, w, fq8=kw.get("fq", False)))), _bits(ref))
            assert _zero_ctr(m, rows)


def test_refuses_unverified_configs():
    m = _build(256, 384)
    M.CHAIN = True
    assert m.chain_ok()
    old = M.KSPLIT
    try:
        M.KSPLIT = 1
        assert not m.chain_ok()                                 # k-split GEMV has an early return before the arrival
    finally:
        M.KSPLIT = old
    with pytest.raises(AssertionError):
        S._chain_check("if (x) return;", "")                    # a GEMV template with an early return is refused

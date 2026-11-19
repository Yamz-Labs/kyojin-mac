"""BP-MOE2: router rowdot + select in one launch (glm_moe_rs.rowsel) against rowdot + glm_moe.select: logits, indices, weights bit-identical."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import glm_moe as GM  # noqa: E402
from kyojin_mlx import glm_moe_rs as RS  # noqa: E402
from kyojin_mlx.rowinv import rowdot  # noqa: E402


def ref(x2, w, bias, T, E, K, norm, scale):
    lg = rowdot(x2, w, out_dtype=mx.float32)
    inds, wts = GM.select(lg, bias, T, E, K, norm, scale)
    return lg, inds, wts


def mk(T, E, H, seed, wdt=mx.float16, ties=False):
    rng = np.random.default_rng(seed)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.5).astype(np.float16))
    w = (rng.standard_normal((E, H)) * 0.02).astype(np.float32)
    bias = (rng.standard_normal((E,)) * 0.1).astype(np.float32)
    if ties:
        w[:] = 0.0
        bias = np.round(bias * 4) / 4
    return x2, mx.array(w).astype(wdt), mx.array(bias)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("E,K", [(288, 8), (256, 8), (192, 6), (1000, 12), (200, 1)])
@pytest.mark.parametrize("norm", [True, False])
def test_bit_identical(T, E, K, norm):
    H = 2048
    assert RS.eligible(T, E, K, H)
    for seed in range(4):                                    # consecutive launches: the counter is reset by the last arriver
        x2, w, bias = mk(T, E, H, seed)
        lg0, i0, w0 = ref(x2, w, bias, T, E, K, norm, 2.5)
        lg1, i1, w1 = RS.rowsel(x2, w, bias, T, E, K, norm, 2.5)
        assert mx.array_equal(lg0, lg1) and mx.array_equal(i0, i1) and mx.array_equal(w0.view(mx.uint32), w1.view(mx.uint32))


@pytest.mark.parametrize("wdt", [mx.float16, mx.bfloat16, mx.float32])
def test_dtypes_and_ties(wdt):
    for ties in (False, True):
        x2, w, bias = mk(2, 288, 4096, 5, wdt, ties)
        a, b = ref(x2, w, bias, 2, 288, 8, True, 2.5), RS.rowsel(x2, w, bias, 2, 288, 8, True, 2.5)
        assert all(mx.array_equal(p.view(mx.uint32) if p.dtype == mx.float32 else p, q.view(mx.uint32) if q.dtype == mx.float32 else q) for p, q in zip(a, b))


def test_nan_row():
    x2, w, bias = mk(1, 288, 4096, 9)
    x2 = x2 * mx.array(float("nan"))
    a, b = ref(x2, w, bias, 1, 288, 8, True, 2.5), RS.rowsel(x2, w, bias, 1, 288, 8, True, 2.5)
    assert mx.array_equal(a[1], b[1])
    assert int(mx.max(b[1]).item()) < 288


def test_chain_many_launches():
    """200 dependent launches (the served pattern: each router input depends on the previous output) all equal the two-launch reference."""
    x2, w, bias = mk(1, 288, 4096, 11)
    outs0, outs1 = [], []
    a = x2
    b = x2
    for i in range(200):
        _, i0, w0 = ref(a, w, bias, 1, 288, 8, True, 2.5)
        _, i1, w1 = RS.rowsel(b, w, bias, 1, 288, 8, True, 2.5)
        outs0.append((i0, w0))
        outs1.append((i1, w1))
        a = mx.depends((a * 0.99).astype(mx.float16), w0)
        b = mx.depends((b * 0.99).astype(mx.float16), w1)
    mx.eval(outs0, outs1)
    for (i0, w0), (i1, w1) in zip(outs0, outs1):
        assert mx.array_equal(i0, i1) and mx.array_equal(w0.view(mx.uint32), w1.view(mx.uint32))


@pytest.mark.parametrize("T", [1, 3, 8])
@pytest.mark.parametrize("E,K", [(288, 8), (64, 4), (256, 8), (1000, 12)])
@pytest.mark.parametrize("ties", [False, True])
def test_select_par_bit_identical(T, E, K, ties):
    """select_par (one threadgroup of E threads per token, from BP-MOEPIPE) = select, bit for bit; ties resolve to the lowest index."""
    rng = np.random.default_rng(T * 7 + E)
    lg = (rng.standard_normal((T, E)) * 2).astype(np.float32)
    if ties:
        lg = np.round(lg)
    bias = np.round((rng.standard_normal((E,)) * 0.1) * 8) / 8 if ties else rng.standard_normal((E,)) * 0.1
    lg, bias = mx.array(lg), mx.array(bias.astype(np.float32))
    for norm in (True, False):
        a, b = GM.select(lg, bias, T, E, K, norm, 2.5), GM.select_par(lg, bias, T, E, K, norm, 2.5)
        assert mx.array_equal(a[0], b[0]) and mx.array_equal(a[1].view(mx.uint32), b[1].view(mx.uint32))

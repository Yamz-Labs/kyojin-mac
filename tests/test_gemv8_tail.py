"""8-bit GEMV / gather with a partial last 512-chunk (k % 512 != 0, k % 32 == 0: Qwen3-30B-A3B experts 768, GLM-4.5 1408): same result as dequant + matmul."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, foreign_pack as FP

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU")
N = 256


def _pack(fmt, n, k, rng):
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    if fmt == "q8_0":
        return FP.Pack("q8_0", n, k, q.view(np.int8).view(np.uint8), rng.uniform(1e-3, 3e-3, (n, k // 32)).astype(np.float16), np.ones(n, np.float32))
    q[(q & 127) == 127] = 0x38                                     # no NaN codes
    if fmt == "fp8":
        return FP.Pack("fp8", n, k, q, None, rng.uniform(5e-5, 2e-4, n).astype(np.float32))
    g = rng.uniform(64.0, 127.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
    return FP.Pack("fp8b", n, k, q, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -20))


def _close(y, ref):
    y, ref = np.array(y, np.float32), np.array(ref, np.float32)
    assert np.isfinite(y).all()
    assert np.abs(y - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-3, np.abs(y - ref).max()


@needs_gpu
@pytest.mark.parametrize("fmt", ["fp8", "fp8b", "q8_0"])
@pytest.mark.parametrize("k", [768, 1408, 2560, 1024])
def test_gemv8_partial_chunk(fmt, k):
    rng = np.random.default_rng(k)
    d = FL.DevPack(_pack(fmt, N, k, rng))
    assert d.gemv_ok()
    w = d.dequant().astype(mx.float32)
    for R in (1, 3):
        x = mx.array(rng.normal(0, 1, (R, k)).astype(np.float16))
        _close(d.apply(x), x.astype(mx.float32) @ w.T)


@needs_gpu
@pytest.mark.parametrize("fmt", ["fp8", "fp8b", "q8_0"])
def test_gather8_partial_chunk(fmt):
    """Stacked experts with k = 768 (the down projection of Qwen3-30B-A3B), gather and the gate|up pair."""
    rng = np.random.default_rng(7)
    E, k = 4, 768
    pg, pu = [_pack(fmt, N, k, rng) for _ in range(E)], [_pack(fmt, N, k, rng) for _ in range(E)]
    g, u = FL.DevPack(pg), FL.DevPack(pu)
    x = mx.array(rng.normal(0, 1, (2, k)).astype(np.float16))
    xrows, eids = mx.array([0, 0, 1, 1], mx.uint32), mx.array([3, 1, 0, 2], mx.uint32)
    ref = lambda ps: mx.stack([x[r].astype(mx.float32) @ FL.DevPack(ps[e]).dequant().astype(mx.float32).T for r, e in zip(xrows.tolist(), eids.tolist())])
    _close(g.gather(x, xrows, eids), ref(pg))
    if g.pair_ok(u):
        a, b = g.gather_pair(u, x, xrows, eids)
        _close(a, ref(pg)); _close(b, ref(pu))

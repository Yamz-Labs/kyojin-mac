"""Fast exact top-k (kyojin_mlx.idxfast) vs the current kernels and vs the stock mx chain.

Bit-identical selection (same index set, same tie order, same -1 tail; same candidate mask on t < lens) for random small dims and DS4.1
dims, ties (quantised scores, -inf, +-0), R = 1..4 rows, several widths incl. chunk boundaries, fp16 / bf16 (+ fp32 vs the stock chain)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

from kyojin_mlx import dsv4_core as C
from kyojin_mlx import idxfast as F


def stock_topk(scores, lens, k):
    sc = mx.where(mx.arange(scores.shape[1])[None, :] < lens[:, None], scores, -mx.inf)
    ids = mx.sort(mx.argsort(-sc, axis=-1)[..., :k], axis=-1).astype(mx.int32)
    return mx.where(ids < lens[:, None], ids, -1)


def stock_mask(scores, lens, blocks, bs):
    scores = mx.where(mx.arange(scores.shape[1])[None, :] < lens[:, None], scores, -mx.inf)
    n = scores.shape[1]
    pad = (-n) % bs
    sc = mx.concatenate([scores, mx.full((scores.shape[0], pad), -mx.inf)], axis=1) if pad else scores
    bm = sc.reshape(scores.shape[0], -1, bs).max(axis=-1)
    nb = bm.shape[1]
    last = (mx.maximum(lens, 1) - 1) // bs
    match = mx.arange(nb)[None, :] == last[:, None]
    pinned = mx.where(match, mx.array(mx.inf), bm)
    order = mx.argsort(-pinned, axis=-1)
    rank = mx.argsort(order, axis=-1)
    keep = (rank < min(blocks, nb)) & (bm > -mx.inf)
    return mx.repeat(keep, bs, axis=-1)[:, :n]


def noisy(rng, N, Tm, step=2.0):
    raw = np.round(rng.standard_normal((N, Tm)) * step) / 2
    raw[:, 2::5] = -np.inf
    raw[:, 3::11] = 0.0
    raw[:, 5::11] = -0.0
    return raw


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,Tm,k,lens", [
    (2, 2048, 512, [1500, 2048]),
    (1, 512, 512, [512]),
    (4, 17408, 512, [17000, 16384, 2500, 99]),
    (4, 131072, 512, [131072, 131000, 65536, 32768]),
    (3, 2050, 24, [2048, 2049, 2050]),
    (2, 300, 24, [5, 300]),
    (2, 300, 24, [0, 1]),
    (1, 40, 16, [9]),
    (1, 40, 1, [40]),
    (4, 4096, 4096, [4096, 4096, 4096, 4096]),
    (2, 8192, 16, [8191, 8192]),
])
def test_topk_mt_matches_current_and_stock(dtype, N, Tm, k, lens):
    rng = np.random.default_rng(1000 * Tm + N)
    sc = mx.array(noisy(rng, N, Tm).astype(np.float32), dtype=dtype)
    la = mx.array(lens, dtype=mx.int32)
    got = np.array(F.topk_mt(sc, la, k))
    cur = np.array(C.index_topk_rt(sc, la, k))
    ref = np.array(stock_topk(sc, la, k))
    assert np.array_equal(got, cur)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,Tm,k,blocks,bs,lens", [
    (2, 2048, 512, 2048, 8, [1500, 2048]),
    (1, 512, 24, 12, 4, [300]),
    (4, 17408, 512, 2048, 8, [17000, 16384, 2500, 99]),
    (2, 300, 24, 12, 4, [5, 300]),
    (1, 40, 16, 3, 8, [9]),
    (2, 2056, 64, 40, 8, [2055, 2056]),
    (2, 100, 8, 12, 1, [99, 100]),
    (3, 65, 64, 9, 8, [0, 1, 65]),
])
def test_select_fused_matches_current_and_stock(dtype, N, Tm, k, blocks, bs, lens):
    rng = np.random.default_rng(77 * Tm + N)
    sc = mx.array(noisy(rng, N, Tm).astype(np.float32), dtype=dtype)
    la = mx.array(lens, dtype=mx.int32)
    gi, gm = F.select_fused(sc, la, k, blocks, bs)
    mx.eval(gi, gm)
    ci, cm = C.index_topk_rt(sc, la, k), C.candidate_mask_rt(sc, la, blocks, bs)
    mx.eval(ci, cm)
    assert np.array_equal(np.array(gi), np.array(ci))
    assert np.array_equal(np.array(gi), np.array(stock_topk(sc, la, k)))
    for u in range(N):
        assert np.array_equal(np.array(gm)[u, :lens[u]], np.array(cm)[u, :lens[u]])
        assert np.array_equal(np.array(gm)[u, :lens[u]], np.array(stock_mask(sc, la, blocks, bs))[u, :lens[u]])


def test_topk_fp32_matches_stock():
    rng = np.random.default_rng(9)
    N, Tm, k = 2, 6000, 64
    raw = (rng.standard_normal((N, Tm)) * 4).round() / 4
    raw[:, ::9] = -np.inf
    sc = mx.array(raw.astype(np.float32))
    la = mx.array([Tm, 5999], dtype=mx.int32)
    got = np.array(F.topk_mt(sc, la, k))
    assert np.array_equal(got, np.array(stock_topk(sc, la, k)))


def test_capacity_invariance():
    """Wider capacity buffer with garbage past lens: same selection (runtime-width contract for verify replay)."""
    rng = np.random.default_rng(31)
    N, Tm, k = 2, 4096, 128
    sc = mx.array(noisy(rng, N, Tm).astype(np.float32), dtype=mx.float16)
    la = mx.array([3000, 4096], dtype=mx.int32)
    a = np.array(F.topk_mt(sc, la, k))
    wide = mx.concatenate([sc, mx.full((N, 2048), 7.0, dtype=mx.float16)], axis=1)
    b = np.array(F.topk_mt(wide, la, k))
    assert np.array_equal(a, b)


def test_width_is_runtime_not_template(monkeypatch):
    """A new score width must reuse the compiled pipelines: the template lists (the pipeline keys) are the same for two widths."""
    from kyojin_mlx import idxfast as IXF
    seen = []
    real = IXF._kernel

    def spy(*a, **kw):
        k = real(*a, **kw)
        return lambda **c: (seen.append(tuple(c["template"])), k(**c))[1]

    monkeypatch.setattr(IXF, "_kernel", spy)
    for w in (8192, 8448):
        sc = mx.random.normal((1, w)).astype(mx.float16)
        ref = np.argsort(-np.array(sc[0, : w - 3].astype(mx.float32)), kind="stable")[:64]
        got = IXF.topk_mt(sc, mx.array([w - 3], mx.int32), 64)
        assert sorted(np.array(got[0]).tolist()) == sorted(ref.tolist())
    half = len(seen) // 2
    assert seen[:half] == seen[half:]

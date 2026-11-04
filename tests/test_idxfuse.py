"""idxfuse (scores + select + candidate mask in one launch) vs the stock rt chain: bit-identical idx, mask on t < lens, scores on t < lens."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

from kyojin_mlx import dsv4_core as C
from kyojin_mlx import idxfuse as IF

NH, DI = 32, 128


def make(rng, N, Tm, dtype, quant):
    q = rng.standard_normal((N, NH, DI)).astype(np.float32)
    k = rng.standard_normal((N, Tm, DI)).astype(np.float32) * 0.5
    w = rng.standard_normal((N, NH)).astype(np.float32) * 0.05
    if quant:                      # coarse values: many equal scores (ties), exact zeros
        q = np.round(q * 2) / 2
        k = np.round(k * 2) / 2
        k[:, 3::7] = 0.0
        w = np.round(w * 20) / 20
    return mx.array(q).astype(dtype), mx.array(k).astype(dtype), mx.array(w).astype(dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("N,Tm,k,lens,kb", [
    (1, 1024, 512, [1024], 0),
    (1, 1024, 512, [1024], 128),
    (2, 4096, 512, [4000, 4096], 512),
    (4, 4096, 512, [4096, 3000, 513, 17], 512),
    (4, 8192, 512, [8192, 1, 0, 600], 0),
    (3, 2050, 24, [2048, 2049, 2050], 40),
    (2, 300, 24, [5, 300], 0),
    (2, 300, 24, [0, 1], 16),
    (1, 40, 16, [9], 4),
    (1, 40, 1, [40], 0),
    (2, 6000, 512, [2500, 6000], 512),     # capacity above the count
    (1, 32768, 512, [32768], 512),
])
def test_fused_matches_rt_chain(dtype, quant, N, Tm, k, lens, kb):
    rng = np.random.default_rng(7 * Tm + N + k)
    q, keys, w = make(rng, N, Tm, dtype, quant)
    ln = mx.array(lens, dtype=mx.int32)
    for tb in (None, 1 << 15):
        idx, mask, sc = IF.fused(q, keys, w, ln, Tm, k, topk_blocks=kb, block_size=8, lmax=tb)
        ref_sc = C.index_scores(q, keys, w, ln, Tm, rt=True)
        ref_idx = C.index_topk_rt(ref_sc, ln, k)
        mx.eval(idx, sc, ref_idx, ref_sc)
        assert np.array_equal(np.array(idx), np.array(ref_idx))
        for u, L in enumerate(lens):
            assert np.array_equal(np.array(sc[u, :L].astype(mx.float32)).view(np.uint32), np.array(ref_sc[u, :L].astype(mx.float32)).view(np.uint32))
        if kb:
            ref_m = C.candidate_mask_rt(ref_sc, ln, kb, 8)
            mx.eval(mask, ref_m)
            for u, L in enumerate(lens):
                assert np.array_equal(np.array(mask[u, :L]), np.array(ref_m[u, :L])), (u, L)


def test_cand_input_and_sessions():
    rng = np.random.default_rng(3)
    N, Tm, k = 3, 4096, 512
    q, keys, w = make(rng, N, Tm, mx.float16, True)
    keys2 = mx.concatenate([keys, keys[:1] * 0.5], axis=0)           # S = 4 key buffers, rows read sess
    sess = mx.array([3, 1, 0], dtype=mx.uint32)
    cand = mx.array(rng.random((N, Tm)) > 0.4)
    ln = mx.array([4096, 2000, 777], dtype=mx.int32)
    idx, mask, sc = IF.fused(q, keys2, w, ln, Tm, k, cand=cand, sess=sess, topk_blocks=64)
    ref_sc = C.index_scores(q, keys2, w, ln, Tm, cand, sess, rt=True)
    assert np.array_equal(np.array(idx), np.array(C.index_topk_rt(ref_sc, ln, k)))
    rm = C.candidate_mask_rt(ref_sc, ln, 64, 8)
    for u, L in enumerate([4096, 2000, 777]):
        assert np.array_equal(np.array(mask[u, :L]), np.array(rm[u, :L]))


def test_repeat_calls_reset_counter():
    rng = np.random.default_rng(5)
    q, keys, w = make(rng, 2, 2048, mx.float16, False)
    ln = mx.array([2048, 1500], dtype=mx.int32)
    ref = C.index_topk_rt(C.index_scores(q, keys, w, ln, 2048, rt=True), ln, 512)
    for _ in range(40):
        idx, _, _ = IF.fused(q, keys, w, ln, 2048, 512)
        assert np.array_equal(np.array(idx), np.array(ref))

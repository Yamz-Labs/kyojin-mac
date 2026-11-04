"""Candidate-block exact top-k (kyojin_mlx.idxblk) vs idxfast.topk_mt and the stock sort chain: same index set, order and -1 tail.
Adversarial: heavy ties, all-equal rows, one huge score per block, +-0, -inf, lens shorter than k, partial last block."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

from kyojin_mlx import idxblk as B
from kyojin_mlx import idxfast as F

B.MIN_ELEMS = 0                                                   # small synthetic shapes: force the candidate path


def stock_topk(scores, lens, k):
    sc = mx.where(mx.arange(scores.shape[1])[None, :] < lens[:, None], scores, -mx.inf)
    ids = mx.sort(mx.argsort(-sc, axis=-1)[..., :k], axis=-1).astype(mx.int32)
    return mx.where(ids < lens[:, None], ids, -1)


def check(raw, lens, k, dt):
    s = mx.array(raw.astype(np.float32)).astype(dt)
    ln = mx.array(lens, dtype=mx.int32)
    assert B.applies(s.shape[0], s.shape[1], k)
    got = B.topk_blocks(s, ln, k)
    ref = F._run(s, ln, k, 2, 8, False)[0]
    st = stock_topk(s, ln, k)
    mx.eval(got, ref, st)
    assert got.dtype == mx.int32 and got.shape == (s.shape[0], k)
    assert np.array_equal(np.array(got), np.array(ref))
    assert np.array_equal(np.array(got), np.array(st))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("Tm,k", [(256, 4), (1000, 16), (4099, 64), (8192, 100)])
def test_random_and_ties(dt, Tm, k):
    rng = np.random.default_rng(Tm + k)
    for step in (4.0, 0.5, 0.0):                                  # step 0: every key equal
        raw = np.round(rng.standard_normal((3, Tm)) * step) / 2
        raw[:, 2::5] = -np.inf
        raw[:, 3::11] = 0.0
        raw[:, 7::13] = -0.0
        lens = [Tm, Tm - 5, max(Tm // 3, 1)]
        check(raw, lens, k, dt)


@pytest.mark.parametrize("dt", [mx.float16, mx.float32])
def test_one_huge_score_per_block(dt):
    rng = np.random.default_rng(1)
    Tm, k = 4096, 32
    raw = np.round(rng.standard_normal((2, Tm))) / 2
    raw[:, ::8] = 50.0                                            # every block has the same huge max; ties go to the lowest index
    check(raw, [Tm, Tm - 3], k, dt)
    raw = np.zeros((2, Tm))
    raw[:, 5 * 8 + 3] = 100.0                                     # one block holds many of the best elements
    raw[:, 5 * 8 : 5 * 8 + 8] += 90.0
    raw[:, 5 * 8 + 3] = 100.0
    raw[:, 777] = 95.0
    check(raw, [Tm, Tm], k, dt)


def test_best_elements_in_few_blocks():
    Tm, k = 4096, 24
    raw = np.zeros((1, Tm))
    raw[0, 8 * 10 : 8 * 10 + 8] = 10.0                            # 8 equal elements in one block
    raw[0, 8 * 20 : 8 * 20 + 8] = 10.0
    raw[0, 8 * 30 : 8 * 30 + 8] = 10.0
    raw[0, 8 * 40 + 1] = 9.0                                      # the tail of the top-k comes from blocks with a single large key
    raw[0, 8 * 41 + 1] = 9.0
    check(raw, [Tm], k, mx.float16)


def test_lens_shorter_than_k_gives_minus_one_tail():
    Tm, k = 4096, 64
    raw = np.random.default_rng(3).standard_normal((2, Tm))
    check(raw, [10, 37], k, mx.float16)
    check(raw, [0 + 1, 64], k, mx.float32)


def test_flag_routes_topk_mt(monkeypatch):
    rng = np.random.default_rng(5)
    s = mx.array(rng.standard_normal((2, 4096)).astype(np.float32)).astype(mx.float16)
    ln = mx.array([4096, 3000], dtype=mx.int32)
    monkeypatch.setattr(F, "DEC_TOPK_BLOCKS", False)
    a = F.topk_mt(s, ln, 32)
    monkeypatch.setattr(F, "DEC_TOPK_BLOCKS", True)
    b = F.topk_mt(s, ln, 32)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))

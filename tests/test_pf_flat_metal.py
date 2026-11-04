"""pf_flat Metal kernels on small synthetic shapes (skips without a GPU with neural accelerators): the block-max kernel and the fused gathered rescoring are bitwise the
fscore values, and the plain / source / consumer paths give the stock indices. tools/bench_pf_flat.py runs the same gate at 128K-1M keys."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import ds_pf_idx as PI   # noqa: E402
from kyojin_mlx import idx_topk as IT    # noqa: E402
from kyojin_mlx import pf_flat as PF     # noqa: E402

pytestmark = pytest.mark.skipif(not (mx.metal.is_available() and PI.NX.nax_available()), reason="needs Metal with neural accelerators")
NH, D, BS = 32, 128, 8


def make(T, Tc, seed):
    rng = np.random.default_rng(seed)
    return (mx.array(rng.standard_normal((1, T, NH, D)).astype(np.float16)), mx.array(rng.standard_normal((1, Tc, D)).astype(np.float16)),
            mx.array((rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16)))


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    monkeypatch.setattr(PF, "BMAX", True)
    monkeypatch.setattr(PF, "BMAX_MIN", 0)


@pytest.mark.parametrize("T,Tc", [(64, 20000), (40, 9001), (33, 4096)])
def test_bmax_kernel_is_the_block_max_of_fscore(T, Tc):
    q, k, w = make(T, Tc, 0)
    start, ratio = 4 * (Tc - T), 4
    s = PI.scores_range(q, k, w, start, ratio, 0, T)
    nblk = -(-Tc // BS)
    ref = mx.pad(s, [(0, 0), (0, nblk * BS - Tc)], constant_values=-mx.inf).reshape(T, nblk, BS).max(-1)
    assert mx.array_equal(PF.bmax_scores(q, k, w, start, ratio, 0, T), ref).item()
    s5 = PI.scores_range(q, k, w, start, ratio, 5, T - 5)                  # rows from t0 > 0 use the absolute causal length of that row
    ref5 = mx.pad(s5, [(0, 0), (0, nblk * BS - Tc)], constant_values=-mx.inf).reshape(T - 5, nblk, BS).max(-1)
    assert mx.array_equal(PF.bmax_scores(q, k, w, start, ratio, 5, T - 5), ref5).item()


@pytest.mark.parametrize("gs", [False, True])
def test_gathered_rescoring_is_bitwise_fscore(gs, monkeypatch):
    monkeypatch.setattr(PF, "GSCORE", gs)
    T, Tc = 48, 12001
    q, k, w = make(T, Tc, 1)
    start, ratio = 4 * (Tc - T), 4
    s = PI.scores_range(q, k, w, start, ratio, 0, T)
    rng = np.random.default_rng(2)
    nblk = -(-Tc // BS)
    ids = np.sort(np.stack([rng.permutation(nblk)[:37] for _ in range(T)]), axis=1).astype(np.int32)
    ids[::3, -5:] = -1
    ids = mx.array(ids)
    got = PF.gathered_scores(q, k, PF.key_blocks(k), w, ids, start, ratio, 0)
    pos = PF._positions(ids)
    ref = mx.where(pos < Tc, mx.take_along_axis(s, mx.minimum(pos, Tc - 1).astype(mx.int32), axis=1), -mx.inf)
    assert mx.array_equal(got, ref).item()
    if gs:
        assert PF.HITS["gscore"] > 0 and not PF.HITS["gscore_fallback"]


@pytest.mark.parametrize("gs", [False, True])
def test_paths_equal_stock(gs, monkeypatch):
    monkeypatch.setattr(PF, "GSCORE", gs)
    T, Tc, topk, K = 64, 20000, 64, 96
    q, k, w = make(T, Tc, 3)
    q2, k2, w2 = make(T, Tc, 4)
    start, ratio = 4 * (Tc - T), 4
    s = PI.scores_range(q, k, w, start, ratio, 0, T)
    ref = PI.topk_block(s, topk)
    assert mx.array_equal(PF.plain_topk(q, k, w, start, ratio, topk, T), ref).item()
    idxs, ids = PF.source_topk(q, k, w, start, ratio, topk, T, K)
    assert mx.array_equal(idxs, ref).item()
    mask = PF.mask_from_ids(ids, -(-Tc // BS), Tc)
    s2 = PI.scores_range(q2, k2, w2, start, ratio, 0, T)
    ref2 = PI.topk_block(mx.where(mask, s2, -mx.inf), topk)
    assert mx.array_equal(PF.consumer_topk(q2, k2, w2, start, ratio, topk, T, ids), ref2).item()

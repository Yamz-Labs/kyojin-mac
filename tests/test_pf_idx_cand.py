"""E1 (candidate-subset scoring) and E2 (key-tiled score loop) of the DS4.1 prefill indexer: same selection and same scores as the full-width path.
Metal kernels, small synthetic shapes (skips without a GPU)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags            # noqa: E402
from kyojin_mlx import ds_pf_idx as PI  # noqa: E402
from kyojin_mlx import idx_topk as IT   # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
NH, D, BS = 32, 128, 8


def make(T, Tc, seed=0, quant=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, NH, D)).astype(np.float16)
    k = rng.standard_normal((1, Tc, D)).astype(np.float16)
    w = (rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16)
    if quant:                                                          # few distinct values: many exact ties
        q, k = (np.round(q * 2) / 2).astype(np.float16), (np.round(k * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


def make_cand(T, Tc, nb, start, ratio, seed=1):
    """Block-granular candidate mask like select_candidate_blocks: <= nb blocks per row among the causally visible ones, the last visible block always kept."""
    rng = np.random.default_rng(seed)
    nblk = -(-Tc // BS)
    m = np.zeros((T, nblk), dtype=bool)
    for t in range(T):
        clen = (start + t + 1) // ratio
        vis = max(1, -(-clen // BS))
        m[t, rng.permutation(vis - 1)[:nb - 1]] = True
        m[t, vis - 1] = True
    return mx.array(np.repeat(m, BS, axis=1)[:, :Tc][None])


def old_path(q, k, w, start, ratio, cand, topk):
    s = PI.scores_range(q, k, w, start, ratio, 0, q.shape[1])
    s = mx.where(cand[0], s, -mx.inf)
    idx = IT.topk(s, topk)
    return np.array(idx), np.array(mx.take_along_axis(s, idx, axis=-1))


@pytest.mark.parametrize("T,Tc,nb,topk,quant", [(64, 4096, 128, 512, False), (64, 9000, 256, 512, False), (64, 20000, 512, 512, False),
                                                 (48, 12000, 200, 256, True), (64, 20000, 640, 512, True)])
def test_cand_topk_matches_full_width(T, Tc, nb, topk, quant):
    q, k, w = make(T, Tc, quant=quant)
    start, ratio = 4 * (Tc - T), 4                                     # the chunk's last rows see (about) all Tc columns
    cand = make_cand(T, Tc, nb, start, ratio)
    ref_i, ref_v = old_path(q, k, w, start, ratio, cand, topk)
    i, v = PI.cand_topk(q, k, w, start, ratio, 0, T, cand, BS, nb, topk, with_scores=True)
    assert np.array_equal(np.array(i), ref_i)
    d = np.abs(np.array(v) - ref_v)
    assert np.all((np.array(v) == ref_v) | (np.isnan(d))) or d.max() == 0.0, f"max abs score diff {d.max()}"


def test_cand_topk_row_blocks_and_micro_blocks():
    T, Tc, nb, topk = 40, 9000, 160, 300
    q, k, w = make(T, Tc, seed=3)
    start, ratio = 4 * (Tc - T), 4
    cand = make_cand(T, Tc, nb, start, ratio)
    ref_i, _ = old_path(q, k, w, start, ratio, cand, topk)
    a = np.array(PI.cand_topk(q, k, w, start, ratio, 0, 17, cand, BS, nb, topk, max_elems=NH * nb * BS * 3))      # 3-row micro-blocks
    b = np.array(PI.cand_topk(q, k, w, start, ratio, 17, T - 17, cand, BS, nb, topk))
    assert np.array_equal(np.concatenate([a, b]), ref_i)


@pytest.mark.parametrize("n,Tc", [(64, 9000), (64, 4097), (32, 20000)])
def test_key_tile_scores_bitwise(n, Tc, monkeypatch):
    q, k, w = make(n, Tc, seed=5)
    start, ratio = 4 * (Tc - n), 4
    small = 8 * NH * Tc                                                # 8-row micro-blocks on the untiled path (the matmul bits of 1-3 row blocks differ, see below)
    monkeypatch.setenv(PI.KTILE_ENV, "0")
    ref = np.array(PI.scores_range(q, k, w, start, ratio, 0, n, small))
    one = np.array(PI.scores_range(q, k, w, start, ratio, 0, n, 1 << 40))     # one block: what the vendor tiles give
    monkeypatch.setenv(PI.KTILE_ENV, "1")
    for me in (small, 2 * NH * Tc, 1 << 40):                           # tiled: same bits whatever the row split, including the 2-row blocks of the untiled path at 1M
        got = np.array(PI.scores_range(q, k, w, start, ratio, 0, n, me))
        assert got.shape == ref.shape and np.array_equal(got, ref) and np.array_equal(got, one)


def test_flag_defaults_match_module_literals():
    for name, env in (("pf_idx_cand", PI.CAND_ENV), ("pf_idx_ktile", PI.KTILE_ENV)):
        assert flags.FLAGS[name].env == env and flags.FLAGS[name].default == "1"
    import inspect
    src = inspect.getsource(PI)
    assert src.count('os.environ.get(KTILE_ENV, "1")') == 1 and src.count('os.environ.get(CAND_ENV, "1")') == 1

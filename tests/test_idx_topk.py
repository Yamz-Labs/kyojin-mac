"""Blocked indexer top-k (kyojin_mlx.idx_topk + the prefill indexer of ds_pf_idx) against the single-pass form: same indices, same order, same candidate mask.

The blocked path must be invisible in the output: it only decides how many query rows are reduced at a time, and every row is reduced on its own. The tests use random
scores with exact ties (the tie rule is "lower index first", i.e. the stable argsort) and chunk lengths that are not a multiple of the block."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import idx_topk as IT                                            # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def scores(T, Tc, seed, ties=False):
    rng = np.random.default_rng(seed)
    if ties:                                            # a small integer range: many rows hold repeated scores
        return mx.array(rng.integers(-3, 4, size=(T, Tc)).astype(np.float32))
    return mx.array(rng.standard_normal((T, Tc)).astype(np.float32))


@pytest.mark.parametrize("T", [1, 5, 32, 100, 129])
@pytest.mark.parametrize("Tc", [3, 16, 64, 200])
@pytest.mark.parametrize("k", [1, 7, 16, 200])
@pytest.mark.parametrize("q", [0, 1, 8, 32, 64])
@pytest.mark.parametrize("ties", [False, True])
def test_blocked_topk_is_the_stock_chain(T, Tc, k, q, ties):
    """`sort(argsort(-s)[..., :k])` per row, concatenated over blocks, is the whole-chunk result, values and order."""
    sc = scores(T, Tc, T * 7 + Tc, ties)
    k = min(k, Tc)
    ref = np.array(mx.sort(mx.argsort(-sc, axis=-1)[..., :k], axis=-1).astype(mx.int32))
    got = np.array(IT.scan(lambda t0, n: sc[t0:t0 + n], T, q, lambda t0, s: IT.topk(s, k)))
    assert got.shape == ref.shape == (T, k)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("T", [5, 100, 129])
@pytest.mark.parametrize("Tc", [16, 200])
@pytest.mark.parametrize("k", [7, 16])
@pytest.mark.parametrize("q", [0, 8, 32])
def test_blocked_partition_is_the_stock_selection(T, Tc, k, q):
    """deepseek_v32 / GLM selection: `argpartition(-k)` per block gives the same rows in the same order as the whole chunk."""
    sc = scores(T, Tc, T + 3 * Tc, ties=True)
    k = min(k, Tc)
    ref = np.array(mx.argpartition(sc, kth=-k, axis=-1)[..., -k:])
    got = np.array(IT.scan(lambda t0, n: sc[t0:t0 + n], T, q, lambda t0, s: IT.partition(s, k)))
    assert np.array_equal(got, ref)


def test_plan_and_qblock():
    assert IT.plan(10, 0) == [(0, 10)] and IT.plan(10, 10) == [(0, 10)] and IT.plan(10, 32) == [(0, 10)]
    assert IT.plan(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert IT.qblock(100, 32) == 32 and IT.qblock(32, 32) == 0 and IT.qblock(100, 0) == 0


def test_scan_runs_one_block_at_a_time_and_joins_tuples():
    seen = []

    def make(t0, n):
        seen.append((t0, n))
        return mx.zeros((n, 4))

    a, b = IT.scan(make, 10, 4, lambda t0, s: (s, s + t0))
    assert seen == [(0, 4), (4, 4), (8, 2)]
    assert a.shape == (10, 4) and b.shape == (10, 4)
    assert np.array_equal(np.array(b), np.concatenate([np.full((4, 4), 0.0), np.full((4, 4), 4.0), np.full((2, 4), 8.0)]))


def test_scan_evaluates_each_block_before_the_next(monkeypatch):
    """The point of the block loop is that a block's scores are dead before the next block is built: every block is evaluated."""
    evals = []
    real = mx.eval

    def spy(*a):
        evals.append(a[0].shape)
        return real(*a)
    monkeypatch.setattr(IT.mx, "eval", spy)
    IT.scan(lambda t0, n: mx.zeros((n, 8)), 10, 4, lambda t0, s: s)
    assert evals == [(4, 8), (4, 8), (2, 8)]


# ---------------------------------------------------------------------------------------- the prefill indexer of DS4.1
L = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from mlx.utils import tree_map                                               # noqa: E402
from mlx_vlm.models.deepseek_v41.config import ModelConfig                   # noqa: E402
from types import SimpleNamespace                                            # noqa: E402

from kyojin_mlx import ds_pf_idx as I                                        # noqa: E402

TOPK = 16


def make_indexer(ratio, cand_layer):
    cfg = ModelConfig()
    cfg.compress_ratios = [ratio, ratio]
    cfg.kv_source_layer_ids = []                                             # no key publishing: the test sets cache.index_k
    cfg.candidate_source_layer_id = cand_layer
    cfg.index_topk = TOPK
    cfg.candidate_topk_blocks = 6
    cfg.candidate_block_size = 4
    cfg.hidden_size = 256
    cfg.q_lora_rank = 128
    cfg.index_n_heads = 32
    cfg.index_head_dim = 128
    cfg.qk_rope_head_dim = 64
    mx.random.seed(5)
    idx = L.Indexer(cfg, 1)
    idx.update(tree_map(lambda p: (p * 8).astype(mx.float16), idx.parameters()))
    mx.eval(idx.parameters())
    return idx, cfg


@pytest.mark.parametrize("T", [33, 64, 100, 129])
@pytest.mark.parametrize("start", [0, 40])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cand_layer", [-1, 0, 1], ids=["plain", "consumer", "source"])
@pytest.mark.parametrize("q", [0, 16, 64, 512])
def test_prefill_indexer_blocked_bit_identical(T, start, ratio, cand_layer, q, monkeypatch):
    """ds_pf_idx blocked vs the same indexer with one block: indices and candidate mask equal to the bit, for a chunk that is not a multiple of the block."""
    idx, cfg = make_indexer(ratio, cand_layer)
    fused = I._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(T + start + ratio * 100 + cand_layer)
    end = start + T
    n = end // ratio
    x = mx.array(rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)).astype(mx.float16)
    qr = mx.array(rng.standard_normal((1, T, cfg.q_lora_rank)).astype(np.float32)).astype(mx.float16)
    keys = mx.array(rng.standard_normal((1, 400, cfg.index_head_dim)).astype(np.float32)).astype(mx.float16)
    cand = mx.array(rng.random((1, T, n)) < 0.7) if cand_layer == 0 else None
    out = {}
    blocks = []
    real_scan = I.IT.scan
    for name, qb in (("whole", 0), ("blocked", q)):
        monkeypatch.setenv(IT.QBLOCK_ENV, str(qb))
        monkeypatch.setattr(I.IT, "scan", lambda mk, n, b, fn: (blocks.append(b), real_scan(mk, n, b, fn))[1])
        cache = SimpleNamespace(index_k=keys, candidates=cand)
        r = fused(idx, x, qr, None, start, 0, cache)
        mx.eval(r)
        out[name] = (np.array(r), None if cache.candidates is None else np.array(cache.candidates))
    monkeypatch.setattr(I.IT, "scan", real_scan)
    assert blocks == [0, IT.qblock(T, q)]                         # effective block: q >= T is one block, i.e. the whole-chunk pass
    assert out["whole"][0].shape == out["blocked"][0].shape == (1, T, min(TOPK, n))
    assert np.array_equal(out["whole"][0], out["blocked"][0]), (T, start, ratio, cand_layer, q)
    if cand_layer == 1:
        assert np.array_equal(out["whole"][1], out["blocked"][1]), (T, start, ratio, q)
        assert out["blocked"][1].shape == (1, T, n)


def test_score_blocks_equal_the_whole_score_matrix(monkeypatch):
    """Splitting the queries changes no score: the matmul product is per row and the post kernel is per element."""
    rng = np.random.default_rng(1)
    T, NH = 100, 32
    q = mx.array(rng.standard_normal((1, T, NH, 128)).astype(np.float16))
    keys = mx.array(rng.standard_normal((1, 150, 128)).astype(np.float16))
    w = mx.array((rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16))
    whole = np.array(I.scores_mm(q, keys, w, 20, 2, max_elems=1 << 20))
    for qb in (0, 16, 33):
        got = np.array(mx.concatenate([s for _, _, s in I.score_blocks(q, keys, w, 20, 2, qblock=qb, max_elems=1 << 20)], axis=0))
        assert np.array_equal(got, whole), qb
    rows = np.array(I.scores_range(q, keys, w, 20, 2, 40, 17))
    assert np.array_equal(rows, whole[40:57])

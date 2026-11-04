"""KYOJIN_ISCORE_BLOCKS: a consumer layer scored over the kept blocks only, against the masked full-width path it stands in for.

Pins (1) the compaction (ascending positions, nothing past a row's count, a spare column for dropped blocks), (2) the gathered scores and the
selection bit-identical to the masked full-width call, (3) the drift against the thread-per-key chain the core served before (same mask, same
top-64 order up to the head-sum rounding), (4) the gate: consumers only, never the candidate source, never below iscore_blocks_min.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.language")

from types import SimpleNamespace                                                  # noqa: E402

from kyojin_mlx import dsa_decode as D                                             # noqa: E402
from kyojin_mlx import dsv4_core as C                                             # noqa: E402
from kyojin_mlx import idxblocks as B                                             # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
NH, DI, BS, NBK, K = 32, 128, 8, 512, 32          # NBK = the mask's own budget: every kept block fits the table


def world(rows=3, cap=2048, sessions=2, dtype=mx.float16, seed=5, keep=0.25, garbage_past_lens=True):
    """Keys, queries, weights, per-row counts / sessions and a block mask: `keep` of the blocks below the row's count, random blocks past it."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((rows, NH, DI)).astype(np.float32)
    w = rng.standard_normal((rows, NH)).astype(np.float32)
    k = rng.standard_normal((sessions, cap, DI)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    lens = np.array([cap, cap - 3, 3 * cap // 4, cap // 2, cap][:rows], dtype=np.int32)
    sess = np.array([i % sessions for i in range(rows)], dtype=np.int32)
    nb = cap // BS
    m = rng.random((rows, nb)) < keep
    m[:, 0] = True                                                             # a kept block at position 0
    mask = np.repeat(m, BS, axis=1)
    for u in range(rows):
        mask[u, lens[u]:] = rng.random(cap - lens[u]) < 0.5 if garbage_past_lens else False   # undefined by contract: must not be read
    a = (mx.array(q).astype(dtype), mx.array(w).astype(dtype), mx.array(k).astype(dtype), mx.array(lens), mx.array(sess), mx.array(mask))
    mx.eval(*a)
    return a


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("keep", [0.25, 1.0, 0.02])
def test_selection_equals_the_masked_full_width_path(dtype, keep):
    """The gathered selection is the masked path's selection: same indices, same -1 tail, for every row."""
    q, w, keys, lens, sess, mask = world(dtype=dtype, keep=keep)
    t_max = keys.shape[1]
    ref_s = D.scores_rows(q, w, keys, lens, t_max, mask, sess)
    ref = C.index_topk_rt(ref_s, lens, K)
    got = B.select(q, w, keys, lens, mask, sess, t_max, K, BS, NBK)
    mx.eval(ref, got)
    np.testing.assert_array_equal(np.array(got), np.array(ref))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gathered_scores_are_bit_identical(dtype):
    """Every kept key holds the score the masked full-width kernel gives it: the matrix kernel is per key, so compaction changes no value."""
    q, w, keys, lens, sess, mask = world(dtype=dtype)
    t_max = keys.shape[1]
    ref = np.array(D.scores_rows(q, w, keys, lens, t_max, mask, sess).astype(mx.float32))
    pos, real = B.kept(mask, lens, t_max, BS, NBK)
    W = pos.shape[1]
    gk = keys[sess[:, None], pos]
    got = D.scores_rows(q, w, gk, real.sum(axis=-1).astype(mx.int32), W, None, mx.arange(q.shape[0], dtype=mx.int32), nax=False)
    mx.eval(got)
    got = np.array(got.astype(mx.float32))
    real = np.array(real)
    for u in range(q.shape[0]):
        np.testing.assert_array_equal(got[u][real[u]], ref[u][np.array(pos[u])[real[u]]])   # slots past the count are not written: never read


def test_compaction_positions():
    """Ascending global positions of exactly the kept keys below the row's count; nothing past `lens`, nothing of a block the mask dropped."""
    q, w, keys, lens, sess, mask = world(rows=2, cap=4096, keep=0.1, seed=13)
    t_max = keys.shape[1]
    pos, real = B.kept(mask, lens, t_max, BS, NBK)
    pos, real = np.array(pos), np.array(real)
    for u in range(2):
        got = pos[u][real[u]]
        want = np.flatnonzero(np.array(mask[u])[: int(lens[u])])
        assert np.array_equal(np.sort(got), want) and len(got) == len(want)
        assert np.all(np.diff(got) > 0)                                        # ascending: the tie rule of the select stays "lower index first"
        assert got.max() < int(lens[u])


def test_against_the_thread_per_key_chain():
    """The path this replaces (the fp32 chain): the same masked keys, the top-64 order inside the head-sum rounding."""
    q, w, keys, lens, sess, mask = world(rows=4, cap=4096, keep=0.25, seed=17)
    t_max = keys.shape[1]
    ref = np.array(C.index_scores(q, keys, w, lens, t_max, mask, sess, rt=True).astype(mx.float32))
    got = B.select(q, w, keys, lens, mask, sess, t_max, K, BS, NBK)
    mx.eval(got)
    got = np.array(got)
    for u in range(4):
        n_u = int(lens[u])
        mk = np.array(mask[u])[:n_u].astype(bool)
        sel = np.array(got[u])[np.array(got[u]) >= 0]
        assert len(sel) == min(K, int(mk.sum()))                               # one index per kept key, up to k
        assert np.all(mk[sel])                                                 # every selected key is a kept key
        kp = np.flatnonzero(mk)
        top = kp[np.argsort(-ref[u][kp], kind="stable")[: len(sel)]]           # the chain's top-k of the same kept keys
        assert len(set(top.tolist()) & set(sel.tolist())) / max(1, len(sel)) >= 0.98


def test_gate():
    """Enabled by its own flag, consumers only, and only from iscore_blocks_min keys."""
    src = SimpleNamespace(n_heads=NH, index_head_dim=DI, index_topk=K, candidate_block_size=BS, candidate_topk_blocks=NBK,
                          is_candidate_source=True, uses_candidates=False)
    con = SimpleNamespace(**{**vars(src), "is_candidate_source": False, "uses_candidates": True})
    q, w, keys, lens, sess, mask = world(rows=2, cap=4096)
    t_max = keys.shape[1]
    B.MIN_KEYS, was, ENABLED = 1024, B.MIN_KEYS, B.ENABLED
    try:
        B.ENABLED = False
        assert not B.applies(con, keys, mask, q, t_max)                        # default off
        B.ENABLED = True
        assert B.applies(con, keys, mask, q, t_max)
        assert not B.applies(src, keys, mask, q, t_max)                        # the source decides the mask
        assert not B.applies(con, keys, None, q, t_max)                        # a plain layer has no mask
        assert not B.applies(con, keys, mask, q, 1023)                         # below iscore_blocks_min
        assert not B.applies(con, keys, mask, q.astype(mx.float32), t_max)     # geometry the row kernel does not serve
    finally:
        B.MIN_KEYS, B.ENABLED = was, ENABLED


def test_layer_returns_none_when_the_gate_is_shut():
    """The core keeps the stock chain when the layer does not take the gathered path."""
    q, w, keys, lens, sess, mask = world(rows=2, cap=4096)
    con = SimpleNamespace(n_heads=NH, index_head_dim=DI, index_topk=K, candidate_block_size=BS, candidate_topk_blocks=NBK,
                          is_candidate_source=False, uses_candidates=True)
    em = SimpleNamespace(lens_arr=lens, tmax=int(lens.max()))
    was = B.ENABLED
    try:
        B.ENABLED = False
        assert B.layer(con, keys, mask, q, w, em, keys.shape[1], sess) is None
    finally:
        B.ENABLED = was

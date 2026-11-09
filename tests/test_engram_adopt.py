"""Flag engram_adopt: the first chunk's row jobs submitted at admission and adopted by the chunk (synthetic tables, no model; the device hash is mlx-vlm's)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kyojin_mlx import engram_mlx as EM, engram_rows as ER, flags
from kyojin_mlx.serve import engine as EN
import test_engram_rows as TE

mx = pytest.importorskip("mlx.core")


@pytest.fixture
def tables(tmp_path):
    sp, rws = TE.spec_with_rows()
    TE.make_shards(tmp_path, sp, rws)
    t = ER.EngramTables(str(tmp_path), sp)
    t.ahead = EM.EngramAhead(t, TE.token_map())
    yield t
    t.close()


def device_hashes(spec, ids, a, b):
    """mlx-vlm's NgramHashState on the cached prefix then the chunk [a, b): what the chunk's first Engram layer gets (history carried by the state)."""
    eng = pytest.importorskip("mlx_vlm.models.deepseek_v41.engram")
    vcfg = SimpleNamespace(engram_layer_ids=list(spec.layer_ids), engram_max_ngram_size=4, engram_n_heads=8, engram_vocab_size=spec.vocab_size,
                           engram_num_embeddings=list(spec.num_embeddings), engram_head_dim=256, engram_pad_token_id=2, engram_compressed_vocab_size=300)
    hasher = eng.NgramHashState(vcfg, eng.EngramLayout.from_config(vcfg), token_map=[int(x) for x in TE.token_map()])
    state = SimpleNamespace(engram=None)
    if a:
        hasher(mx.array(np.asarray(ids[:a])[None]), 0, state)
    return hasher(mx.array(np.asarray(ids[a:b])[None]), a, state)


def test_flag_registered_default_off():
    f = flags.FLAGS["engram_adopt"]
    assert f.env == "KYOJIN_ENGRAM_ADOPT" and f.default == "0"


@pytest.mark.parametrize("a,b", [(0, 300), (1, 200), (2, 130), (3, 140), (4, 400), (63, 200), (1000, 1100), (1500, 1800)])
def test_host_rows_equal_device_hashes_bit_for_bit(tables, a, b):
    ids = list(np.random.default_rng(a + b).integers(0, 2000, 2000))
    ah = tables.ahead
    assert ah.adopt_first(7, ids, a, b)
    h, jobs = ah.adopt_pending[7].fut.result(10)
    want = np.asarray(device_hashes(tables.spec, ids, a, b))
    assert h.dtype == want.dtype == np.int64 and np.array_equal(h[None], want)
    for p in jobs:
        jobs[p][1].result(10)


def _adopted_run(tables, ids, a, b, hashes):
    ah = tables.ahead
    before = ER.STATS["calls"]
    assert ah.adopt_first(1, ids, a, b)
    ah.adopt_pending[1].fut.result(10)
    ah.adopt_select(1, a, b)
    ah.start(hashes)
    return ah, before


def test_chunk_adopts_the_admission_jobs_no_second_read(tables):
    ids = list(np.random.default_rng(1).integers(0, 2000, 900))
    a, b = 130, 700
    hashes = device_hashes(tables.spec, ids, a, b)
    ah, before = _adopted_run(tables, ids, a, b, hashes)
    assert ah.adopt_hit == 1 and ah.adopt_miss == 0 and ah.last_adopted
    for p in range(len(tables.spec.layer_ids)):
        shape, uniq, inv, w, s = ah.take(p, tuple(hashes.shape[:2]) + (hashes.shape[3],)).result(10)
        hp = np.asarray(hashes)[:, :, p, :]
        wu, inv_want = np.unique(hp.reshape(-1), return_inverse=True)
        w_want, s_want = tables.read_rows(p, wu)
        assert tuple(shape) == hp.shape and np.array_equal(uniq, wu) and np.array_equal(inv, inv_want)
        assert np.array_equal(w, w_want) and np.array_equal(s, s_want)
    # one read per layer for the chunk's rows, none for the adopted twin (the checks above add len(layers) reads of their own)
    assert ER.STATS["calls"] - before == 2 * len(tables.spec.layer_ids)


def test_adopted_and_cold_chunk_give_the_same_embedding_rows(tables):
    ids = list(np.random.default_rng(2).integers(0, 2000, 600))
    a, b = 70, 400
    hashes = device_hashes(tables.spec, ids, a, b)
    ah, _ = _adopted_run(tables, ids, a, b, hashes)
    emb = EM.ParallelRowEmbedding(tables, 1)
    got = np.array(emb(hashes[:, :, 1, :]).astype(mx.float32))
    ah.jobs = {}
    ah.adopt_cur = None
    cold = np.array(EM.ParallelRowEmbedding(tables, 1)(hashes[:, :, 1, :]).astype(mx.float32))
    assert np.array_equal(got, cold)


def test_debug_mode_asserts_equal_rows(tables):
    ids = list(np.random.default_rng(3).integers(0, 2000, 500))
    tables.ahead.adopt_debug = True
    ah, _ = _adopted_run(tables, ids, 10, 300, device_hashes(tables.spec, ids, 10, 300))
    assert ah.adopt_hit == 1


def test_other_hashes_fall_back_to_the_plain_path(tables):
    ids = list(np.random.default_rng(4).integers(0, 2000, 500))
    other = list(ids)
    other[150] = (other[150] + 1) % 2000                               # a different token inside the chunk (e.g. another cache hit's history)
    ah = tables.ahead
    assert ah.adopt_first(1, ids, 100, 400)
    ah.adopt_pending[1].fut.result(10)
    ah.adopt_select(1, 100, 400)
    ah.start(device_hashes(tables.spec, other, 100, 400))
    assert ah.adopt_hit == 0 and ah.adopt_miss == 1 and not ah.last_adopted
    assert set(ah.jobs) == {0, 1} and ah.started == 1                    # start() submitted its own jobs


def test_history_across_the_cut_is_checked(tables):
    ids = list(np.random.default_rng(5).integers(0, 2000, 500))
    other = list(ids)
    other[99] = (other[99] + 1) % 2000                                 # the last cached token differs: only the history of the first positions changes
    ah = tables.ahead
    assert ah.adopt_first(1, ids, 100, 400)
    ah.adopt_pending[1].fut.result(10)
    ah.adopt_select(1, 100, 400)
    ah.start(device_hashes(tables.spec, other, 100, 400))
    assert ah.adopt_hit == 0 and ah.adopt_miss == 1


def test_other_chunk_bounds_or_unfinished_worker_or_small_chunk(tables):
    ids = list(np.random.default_rng(6).integers(0, 2000, 500))
    ah = tables.ahead
    assert ah.adopt_first(1, ids, 100, 400)
    ah.adopt_select(1, 100, 300)                                       # the real first chunk ends elsewhere
    assert ah.adopt_cur is None and ah.adopt_miss == 1 and not ah.adopt_pending
    assert not ah.adopt_first(2, ids, 0, 10)                           # decode-sized
    ah.token_map = None
    assert not ah.adopt_first(3, ids, 0, 300)
    ah.token_map = TE.token_map()
    ah.adopt_select(99, 0, 10)                                         # unknown session: nothing armed
    assert ah.adopt_cur is None


def test_pending_queue_is_bounded_and_finish_frees(tables):
    ids = list(np.random.default_rng(7).integers(0, 2000, 500))
    ah = tables.ahead
    for sid in range(6):
        ah.adopt_first(sid, ids, 0, 200)
    assert len(ah.adopt_pending) == 4 and ah.adopt_miss == 2
    ah.adopt_drop(5)
    assert 5 not in ah.adopt_pending and ah.adopt_miss == 3


def test_prefix_peek_matches_lookup_without_side_effects():
    class BE:
        def restore(self, snap): return ("cache", snap)
        def snapshot(self, c): return c
        def nbytes(self, s): return 10
        def shared_parts(self, s): return {}
    pc = EN.PrefixCache(BE(), 16, 10_000)
    ids = list(range(100))
    pc.entries[EN.prefix_digests(ids, [32])[0]] = ("snap32", 10, None)
    pc.entries[EN.prefix_digests(ids, [64])[0]] = ("snap64", 10, None)
    order = list(pc.entries)
    assert pc.peek(ids) == 64 and list(pc.entries) == order and pc.hits == 0
    assert pc.peek(ids[:64]) == 32                                     # a proper prefix only
    assert pc.peek(ids, grid=lambda b, sig: b != 64) == 32
    assert pc.peek([5] + ids[1:]) == 0
    assert pc.lookup(ids)[0] == 64

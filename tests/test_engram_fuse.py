"""Engram hash in one launch (kyojin_mlx.engram_fuse) against the stock NgramHashState: hash ids and the history on the state equal bit for bit, over
decode steps, chained rows, masks, rollbacks (a history longer than the call), batches, and the fallbacks; plus the launch count."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

mx = pytest.importorskip("mlx.core")
eng = pytest.importorskip("mlx_vlm.models.deepseek_v41.engram")
from test_engram_rows import spec_with_rows, token_map
from kyojin_mlx import engram_fuse as EF


def hasher(seed=0):
    sp, rows = spec_with_rows()
    tm = token_map(seed=seed)
    vcfg = SimpleNamespace(engram_layer_ids=list(sp.layer_ids), engram_max_ngram_size=4, engram_n_heads=8, engram_vocab_size=sp.vocab_size,
                           engram_num_embeddings=list(rows), engram_head_dim=256, engram_pad_token_id=2, engram_compressed_vocab_size=300)
    return eng.NgramHashState(vcfg, eng.EngramLayout.from_config(vcfg), token_map=[int(x) for x in tm])


@pytest.fixture(scope="module")
def h():
    EF.install()
    return hasher()


def both(h, ids, start, st_a, st_b, mask=None):
    a = h._stock_call(ids, start, st_a, token_mask=mask)
    b = h(ids, start, st_b, token_mask=mask)
    mx.eval(a, b, st_a.engram, st_b.engram)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(np.array(a), np.array(b))
    assert st_a.engram.shape == st_b.engram.shape and st_a.engram.dtype == st_b.engram.dtype
    assert np.array_equal(np.array(st_a.engram), np.array(st_b.engram))


def run(h, ids_all, chunks, batch=1, masks=None):
    sa, sb = SimpleNamespace(engram=None), SimpleNamespace(engram=None)
    pos = 0
    for n in chunks:
        ids = mx.array(ids_all[:batch, pos:pos + n])
        both(h, ids, pos, sa, sb, None if masks is None else mx.array(masks[:batch, pos:pos + n]))
        pos += n
    return sa, sb, pos


def test_decode_and_rows(h):
    ids = np.random.default_rng(1).integers(0, 2000, (1, 120))
    run(h, ids, [57] + [1] * 20 + [3, 1, 5, 2, 4, 8, 1, 1] + [9] + [1] * 5)          # prefill, decode, chained rows (1..8), a 9-row call (stock path)


def test_batch_and_mask(h):
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 2000, (3, 80))
    masks = rng.random((3, 80)) > 0.2
    run(h, ids, [30, 1, 1, 2, 1, 4, 1], batch=3, masks=masks)


def test_rollback_history_longer_than_call(h):
    ids = np.random.default_rng(3).integers(0, 2000, (1, 90))
    sa, sb, pos = run(h, ids, [50, 4])
    both(h, mx.array(ids[:, 50:51]), 50, sa, sb)                                          # re-decode inside the history: stock keeps the tail
    both(h, mx.array(ids[:, 51:53]), 51, sa, sb)
    both(h, mx.array(ids[:, 54:55]), 54, sa, sb)


def test_batch_change_falls_back(h):
    sa, sb = SimpleNamespace(engram=None), SimpleNamespace(engram=None)
    ids = np.random.default_rng(4).integers(0, 2000, (2, 40))
    both(h, mx.array(ids[:1, :20]), 0, sa, sb)
    both(h, mx.array(ids[:2, 20:21]), 20, sa, sb)


def test_dead_and_start(h):
    ids = np.random.default_rng(5).integers(0, 2000, (1, 12))
    masks = np.ones((1, 12), dtype=bool); masks[0, [2, 7]] = False
    run(h, ids, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], masks=masks)                         # starts at position 0, dead tokens inside the n-gram window


def test_one_launch(h):
    from count_launches import graph_census
    sa, sb = SimpleNamespace(engram=None), SimpleNamespace(engram=None)
    ids = np.random.default_rng(6).integers(0, 2000, (1, 70))
    both(h, mx.array(ids[:, :60]), 0, sa, sb)
    o = h(mx.array(ids[:, 60:61]), 60, sb)
    c = graph_census([o, sb.engram])
    assert c["dispatches"] <= 2, c["by_type"]                                             # the custom kernel (+ the scalar position array)
    o2 = h._stock_call(mx.array(ids[:, 60:61]), 60, sa)
    assert graph_census([o2, sa.engram])["dispatches"] > 40

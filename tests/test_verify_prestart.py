"""Flag verify_prestart on stubs (no model, no Metal): a lazy-draft verify pass defers its Engram hash read to the first Engram layer. Same rows, same
embedding; the order of the host and the GPU is what changes (hash kernels submitted at once, pre-Engram layers submitted before the read)."""
import os
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from kyojin_mlx import engram_mlx as EM, engram_rows as ER, flags as FL, rows as RW
import test_engram_rows as TE


@pytest.fixture
def tables(tmp_path):
    sp, rws = TE.spec_with_rows()
    TE.make_shards(tmp_path, sp, rws)
    t = ER.EngramTables(str(tmp_path), sp)
    t.ahead = EM.EngramAhead(t)
    yield t, rws
    t.close()


def hashes(rws, n=6, seed=0):
    return mx.array(np.stack([np.random.default_rng(seed + p).integers(0, rws[p], (n, 1, 24)) for p in range(2)], axis=2).reshape(n, 1, 2, 24))


def test_flag_registered_off_by_default_and_switchable():
    f = FL.FLAGS["verify_prestart"]
    assert f.default == "0" and f.runtime and f.attrs[0][:2] == ("engram_mlx", "PRESTART")
    assert EM.PRESTART is False


def test_defer_reads_nothing_until_release_and_submits_the_hash_kernels_first(tables, monkeypatch):
    t, rws = tables
    h = hashes(rws)
    log = []
    monkeypatch.setattr(mx, "async_eval", lambda *a: log.append(("async_eval", [id(x) for x in a])))
    ahead = t.ahead
    ahead.defer(h)
    assert ahead.late is h and ahead.jobs == {} and ahead.started == 0       # no host read of the ids yet
    assert log == [("async_eval", [id(h)])]
    ahead.release()
    assert ahead.late is None and ahead.started == 1 and set(ahead.jobs) == {0, 1}
    ahead.release()                                                          # idempotent
    assert ahead.started == 1


class FakeEngram:
    """The Engram layer module the rows-ahead wrapper subclasses: embeds its hash column and records what the host did, in order."""
    layer_hash_index = 0

    def __init__(self, tables, log):
        self.emb, self.log = EM.ParallelRowEmbedding(tables, 0), log

    def __call__(self, x, hash_ids, token_mask=None):
        self.log.append("embed")
        return self.emb(hash_ids[:, :, 0, :]) if hash_ids.ndim == 4 else self.emb(hash_ids)


def test_first_engram_layer_submits_then_reads_and_the_embedding_is_unchanged(tables, monkeypatch):
    t, rws = tables
    h = hashes(rws)
    log = []
    ref = FakeEngram(t, [])(None, h)                                         # the old order: ids read at the entry, no rows ahead
    cls = EM._ahead_engram(FakeEngram, t.ahead)
    layer = cls(t, log)
    x = mx.ones((6, 4))
    seen = []
    monkeypatch.setattr(mx, "async_eval", lambda *a: seen.append(("async_eval", len(t.ahead.jobs), t.ahead.late is not None)))
    t.ahead.jobs = {}
    t.ahead.defer(h)
    out = layer(x, h)
    # defer submitted the hash kernels (late pending, no jobs); the layer submitted x BEFORE the read (late still pending), then the read started
    assert seen[0] == ("async_eval", 0, True) and seen[1] == ("async_eval", 0, True)
    assert t.ahead.late is None and t.ahead.started >= 1
    assert mx.array_equal(out, ref).item()


def test_a_layer_without_a_pending_read_is_untouched(tables, monkeypatch):
    t, rws = tables
    calls = []
    monkeypatch.setattr(mx, "async_eval", lambda *a: calls.append(a))
    layer = EM._ahead_engram(FakeEngram, t.ahead)(t, [])
    layer(mx.ones((2, 2)), hashes(rws, 2))
    assert calls == [] and t.ahead.late is None                              # no wait, no submit: the prefill and plain-step paths are as before


class _Hash:
    def __init__(self, ahead):
        self._kyojin_ahead = ahead

    def __call__(self, ids, offset, row, token_mask=None):
        return (ids[:, :, None, None] * 7 + mx.arange(48).reshape(1, 1, 2, 24)) % 50


def _pass(tables_, token_lists, monkeypatch, on):
    monkeypatch.setattr(EM, "PRESTART", on)
    lm = SimpleNamespace(engram_hash=_Hash(tables_.ahead), _ensure_engram_hash=lambda: None)
    entry = SimpleNamespace(rows=[SimpleNamespace(offset=0, engram=None) for _ in token_lists])
    return RW._engram_pass(lm, entry, token_lists, read_ahead=True)


def test_engram_pass_defers_only_a_lazy_draft_with_the_flag_on(tables, monkeypatch):
    t, _ = tables
    monkeypatch.setattr(mx, "async_eval", lambda *a: None)
    lazy = [mx.array([3, 4, 5], dtype=mx.int32)]
    eh, _h = _pass(t, lazy, monkeypatch, True)
    assert t.ahead.late is eh and t.ahead.jobs == {}
    eh, _h = _pass(t, lazy, monkeypatch, False)                              # flag off: read at the entry as today
    assert t.ahead.late is None and t.ahead.started == 1
    t.ahead.started = 0
    eh, _h = _pass(t, [[3, 4, 5]], monkeypatch, True)                        # host token lists: nothing lazy to wait for, read at the entry
    assert t.ahead.late is None and t.ahead.started == 1


def test_a_pending_read_never_survives_to_the_next_pass(tables, monkeypatch):
    t, _ = tables
    monkeypatch.setattr(mx, "async_eval", lambda *a: None)
    lazy = [mx.array([3, 4, 5], dtype=mx.int32)]
    _pass(t, lazy, monkeypatch, True)
    assert t.ahead.late is not None
    _pass(t, [[1, 2, 3]], monkeypatch, True)                                 # the next pass resets it before it starts its own jobs
    assert t.ahead.late is None

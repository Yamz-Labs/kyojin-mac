"""kvmem: the levers of this branch composed. kvseg seals pool / key rows into shared segments; stackpack stacks the same layers as codes for a batched step and
releases the session entries while stacked; kv_handoff_free drops the dead cross-layer fields after a prefill chunk. This file checks the crossings on CPU: the
sealed lists come back from a stack, the bill does not lose them, and freeing the handoff leaves every stored row in place."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import fp4pack, fp4pool as FP, kvledger as KL, stackpack as SP  # noqa: E402
from kyojin_mlx.fp4pool import KeySlots, PoolSlots  # noqa: E402

SEG = 32


@pytest.fixture(autouse=True)
def cpu_and_flags(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(FP, "SEG", SEG)
    for k in ("KYOJIN_KV_HANDOFF_FREE", "KYOJIN_KV_SEG_ONCE", "KYOJIN_STACKPACK"):
        monkeypatch.delenv(k, raising=False)
    yield
    mx.set_default_device(dev)


def grid(shape, block, e4m3, seed=0, dtype=mx.float16):
    x = mx.random.normal(shape, key=mx.random.key(seed)) * 3.0
    c, s = fp4pack.pack(x.astype(dtype), block, e4m3)
    return fp4pack.unpack(c, s, block, e4m3, dtype)


def sealed_slots(cls, rows, seed=0, D=512):
    """A live slots object whose first `rows` rows are sealed into segments (what kvseg does at a snapshot)."""
    s = cls([None])
    s.append_rows(0, grid((1, rows, D), cls.BLOCK, cls.E4M3, seed))
    mx.eval(s.raw(0).codes, s.raw(0).scales)
    s.segment_snapshot()                                   # copies the full segments into the live entry, returns the shared snapshot
    mx.eval(*s.raw_arrays())
    return s


@pytest.mark.parametrize("cls,D", [(PoolSlots, 512), (KeySlots, 128)])
def test_sealed_segments_survive_a_stack_and_the_writeback(cls, D):
    a = sealed_slots(cls, 3 * SEG, 1, D)
    b = sealed_slots(cls, 2 * SEG, 2, D)
    held = [a.raw(0), b.raw(0)]
    before = [(e.segs, e.sealed, e.sr, e.sb) for e in held]
    assert all(sg[3] > 0 for sg in before)

    ps, cnt = SP.stack([a, b], 0, 8)
    assert ps is not None and cnt == [3 * SEG, 2 * SEG]
    assert ps.segs[0] == before[0] and ps.segs[1] == before[1]        # the same segment arrays, shared by identity
    SP.release([a, b], 0)
    assert a.raw(0) is None and b.raw(0) is None                      # the batch is the only holder while stacked

    for u, e in enumerate(held):
        back = SP.session_entry(ps, u, e.n)
        assert back.segs is e.segs and back.sealed == e.sealed and back.sr == e.sr and back.sb == e.sb
        assert bool(mx.array_equal(back.codes, e.codes[:, :e.n]).item())
        assert bool(mx.array_equal(back.scales, e.scales[:, :e.n]).item())


def test_writeback_drops_a_stale_sealed_list_shorter_than_it():
    e = sealed_slots(PoolSlots, 2 * SEG).raw(0)
    ps, _ = SP.stack([sealed_slots(PoolSlots, 2 * SEG)], 0, 8)
    back = SP.session_entry(ps, 0, SEG // 2)                          # a truncation below the sealed rows
    assert back.segs is None and back.n == SEG // 2


def test_ledger_counts_a_sealed_segment_once_across_the_store():
    """kvseg: the prefix ledger bills a segment once; kv_seg_once removes the live layer's copy of the same segment from Engine.kv_bytes (accounting only)."""
    s = sealed_slots(PoolSlots, 2 * SEG)
    e = s.raw(0)
    counted = sum(a.nbytes for a in s.raw_arrays())
    ret = FP.retained_walk([s], billed={id(sg): sg[0].nbytes + sg[1].nbytes for sg in e.segs})
    assert ret["seg"] == e.sb and ret["seg"] > 0 and ret["seg_overlap"] == e.sb
    assert ret["counted"] == counted and counted == e.n * (e.codes.shape[2] + e.scales.shape[2])    # the contiguous buffer through n; the segments are extra resident bytes


def test_handoff_free_keeps_the_sealed_rows(tmp_path):
    mx.random.seed(0)
    s = sealed_slots(PoolSlots, 2 * SEG)
    segs = s.raw(0).segs
    n = s.raw(0).n
    # a tiny cache object with the four handoff fields and a slot list, as a layer cache holds them
    from types import SimpleNamespace

    class C:
        def __init__(self):
            self.compress, self.keys = s, KeySlots([None])
            self.index_k = mx.zeros((1, n, 128), dtype=mx.float16)
            self.candidates = mx.zeros((1, n), dtype=mx.int32)
            self.compress_kv = mx.zeros((1, n, 512), dtype=mx.float16)
            self.topk_idxs = mx.zeros((1, 1, 8), dtype=mx.int32)

        def reset_handoff(self):
            self.index_k = self.candidates = self.compress_kv = self.topk_idxs = None

    c = C()
    hand = KL.handoff_bytes(c)
    assert hand > 0 and KL.free_handoff(c) == hand and KL.handoff_bytes(c) == 0
    assert c.compress is s and c.compress.raw(0).segs is segs and c.compress.raw(0).n == n
    rows = KL.itemize(c)
    assert not [r for r in rows if r["role"] == "handoff"]


def test_an_evicted_entry_does_not_pin_its_draft_snapshot():
    """Retention: PrefixCache.last_draft pointed at the last hit's drafter ring and survived eviction / shrink(0) (the admin flag switch drops the store)."""
    from kyojin_mlx.serve.engine import PrefixCache
    from kyojin_mlx.serve.backends import Backend

    class B(Backend):
        def snapshot(self, cache):
            return cache

        def fork(self, cache):
            return cache

        def nbytes(self, cache):
            return 1

        def draft_nbytes(self, d):
            return 1

        def draft_snapshot(self, d):
            return d

        def shared_parts(self, snap):
            return {}

    pc = PrefixCache(B(), 16, 1 << 20)
    d1, d2 = {"ring": 1}, {"ring": 2}
    pc.store((16, b"a"), "A", draft=d1)
    pc.store((32, b"b"), "B", draft=d2)
    pc.last_draft = pc.entries[(32, b"b")][2]
    pc.shrink(2)                                            # drops the older entry only: the newer draft stays
    assert pc.last_draft is d2 and (16, b"a") not in pc.entries
    pc.shrink(0)
    assert pc.last_draft is None and not pc.entries


def test_flags_are_registered_and_carry_the_served_defaults():
    from kyojin_mlx import flags
    want = {"kvseg": ("KYOJIN_KVSEG", "8192"), "stackpack": ("KYOJIN_STACKPACK", "0"), "cdecpack": ("KYOJIN_CDECPACK", "0"),
            "kv_handoff_free": ("KYOJIN_KV_HANDOFF_FREE", "1"), "kv_seg_once": ("KYOJIN_KV_SEG_ONCE", "1")}
    for name, (env, dflt) in want.items():
        f = flags.FLAGS[name]
        assert f.env == env and f.default == dflt

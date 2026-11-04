"""pf_grow (KYOJIN_PF_GROW): a pool / index-key layer restored from a prefix snapshot or a disk checkpoint carries room for its next appends, so the first
append after a restore writes its rows where they go instead of growing the layer (`mx.zeros` of the whole next capacity step + a copy of every row, the 985 MB
of zeros + 0.83 GiB copy the served lane pays per turn at 983040 rows).

CPU only, small tensors (4096 pool rows: 1088 KiB of codes, 136 KiB of scales). Bytes are counted with the MLX allocator, never timed. `DF.ENABLED` is off so
the bytes of an append are the codes' (the dense mirror has its own lever), and `defer_dense` keeps `append_rows` from reading the layer back."""
import gc
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import depthflat as DF, fp4pack, fp4pool, longctx as LC, rows as R  # noqa: E402
from kyojin_mlx.fp4pool import KeySlots, PoolSlots, make_packed_class  # noqa: E402

N, D = 4096, 512                      # rows of the fixture pool layer
KB = 1024.0


@pytest.fixture(autouse=True)
def cpu_flat(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(DF, "ENABLED", False)                      # no mirror: the append's bytes are the code buffers'
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)
    monkeypatch.setattr(PoolSlots, "defer_dense", True)            # append_rows does not read the layer back dense
    monkeypatch.setattr(KeySlots, "defer_dense", True)
    for k in fp4pool.HITS:
        fp4pool.HITS[k] = 0
    yield
    mx.set_default_device(dev)


def grid(n, D=512, seed=0, block=16, e4=True):
    """`n` rows on the codec's grid: packing them changes no value."""
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def bits(a):
    a = np.array(a)
    return a.view(np.uint8) if a.dtype == np.float16 else a


def packed_class():
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    return make_packed_class(DeepseekV41Cache)


def cache(n=N, seed=1):
    """One layer of a packed vendor cache with a pool and an index-key layer of `n` rows (the shape the served cache has)."""
    c = packed_class()(1)
    c.append_compressed(0, grid(n, 512, seed))
    c.append_index_keys(0, grid(n, 128, seed + 1, 32, False))
    mx.eval(*c.compress.raw_arrays(), *c.keys.raw_arrays())
    return c


def exact(cache_):
    """`cache_` with its code buffers cut to exactly their rows: what a compacted prefix snapshot and a disk checkpoint hold."""
    for slots in (cache_.compress, cache_.keys):
        for i, e in enumerate(list.__iter__(slots)):
            if e is not None:
                slots.set_raw(i, fp4pool._Pk(fp4pool.own(e.codes[:, :e.n]), fp4pool.own(e.scales[:, :e.n]), e.n, e.dtype))
        mx.eval(*slots.raw_arrays())
    return cache_


def restored(c, how, monkeypatch, room):
    """`c` after a restore of the kind the lane takes: 'disk' = the checkpoint path (`state` setter, `DsAdapter.restore`), 'hit' = a prefix snapshot hit
    (`PrefixCache._lookup_memory` -> `rows.restore`)."""
    monkeypatch.setattr(fp4pool, "RESTORE_ROOM", room)
    if how == "disk":
        exact(c)
        c.state = c.state
        return c
    snap = R.snapshot([c])
    return R.restore(snap)[0]


def peak(f):
    """(peak KiB over the active bytes at the start, result) of `f`, which must evaluate what it builds."""
    gc.collect()
    mx.eval(mx.zeros(1))
    b0 = mx.get_active_memory()
    mx.reset_peak_memory()
    out = f()
    mx.eval(mx.zeros(1)) if out is None else mx.eval(out)
    return (mx.get_peak_memory() - b0) / KB, out


# ------------------------------------------------------------------------------------------------ the room
def test_the_room_is_a_budget_the_restored_kv_can_pay():
    """`longctx.room` at the lane's depth: the restored KV may grow by 10 % (the brief), the rule spends 6.25 % of the codes' rows."""
    for n in (8192, 65536, 491520, 983040):
        cap = LC.room(n, PoolSlots.STEP)
        assert cap % PoolSlots.STEP == 0 and cap > n and cap - n >= PoolSlots.STEP
        assert cap / n - 1 <= 0.10
    assert LC.room(983040, 256) == 1044480 and LC.room(983040, 256) / 983040 - 1 == 0.0625


@pytest.mark.parametrize("how", ["disk", "hit"])
def test_restore_with_room_reads_bit_identical(how, monkeypatch):
    """The rows of a restored layer are the rows it had: same codes, same dense read as the flag-off restore, and the capacity is only a stride."""
    off, on = cache(), cache()
    src = cache()
    r_off = restored(off, how, monkeypatch, False)
    r_on = restored(on, how, monkeypatch, True)
    e_off, e_on = r_off.compress.raw(0), r_on.compress.raw(0)
    assert e_off.codes.shape[1] == N and e_on.codes.shape[1] == LC.room(N, PoolSlots.STEP)
    for a, b in zip(r_off.compress.raw_arrays(), r_on.compress.raw_arrays()):
        assert np.array_equal(np.array(a), np.array(b))            # same codes, same rows, only the buffer is wider
    assert r_on.keys.raw(0).codes.shape[1] == LC.room(N, KeySlots.STEP)
    for a, b in zip(r_off.keys.raw_arrays(), r_on.keys.raw_arrays()):
        assert np.array_equal(np.array(a), np.array(b))
    for r in (r_off, r_on):                                        # what the attention reads: the rows it was built from
        assert np.array_equal(bits(r.compress[0]), bits(src.compress[0]))
        assert np.array_equal(bits(r.keys[0]), bits(src.keys[0]))
    assert fp4pool.HITS["room"] == 2                                 # one pool layer + one index-key layer


# ------------------------------------------------------------------------------------------------ no copy at the first append
@pytest.mark.parametrize("how", ["disk", "hit"])
def test_no_copy_at_first_append_after_restore(how, monkeypatch):
    """The append after a restore allocates nothing of the layer: one row's codes, no zero buffer, no copy of the pool. Flag off: the growth."""
    v = grid(8, 512, 2)
    mx.eval(v)
    r = restored(cache(), how, monkeypatch, True)
    cap0 = r.compress.raw(0).codes.shape[1]
    room_kb, _ = peak(lambda: (r.append_compressed(0, v), mx.eval(*r.compress.raw_arrays())))
    e = r.compress.raw(0)
    assert e.n == N + 8 and e.codes.shape[1] == cap0                 # the buffer did not move: the rows went in place
    assert room_kb < 0.5 * e.codes.nbytes / KB                       # no allocation of the buffer class: 8 rows + the append's own scratch

    d_grow = restored(cache(), how, monkeypatch, False)
    assert d_grow.compress.raw(0).codes.shape[1] == N                # exactly the rows: the next append has no room
    grow_kb, _ = peak(lambda: (d_grow.append_compressed(0, v), mx.eval(*d_grow.compress.raw_arrays())))
    e2 = d_grow.compress.raw(0)
    assert e2.codes.shape[1] == LC.capacity(N + 8, PoolSlots.STEP)   # today: the next capacity step, zeros + a copy of every row
    assert grow_kb > e.codes.nbytes / KB * 0.9                       # the growth allocates the whole buffer
    assert grow_kb > room_kb * 3                                     # the lever removes that allocation


def test_restore_off_is_the_fork_it_was(monkeypatch):
    """pf_grow off: `rows.restore` is `rows.fork`, no room is added and nothing is counted."""
    monkeypatch.setattr(fp4pool, "RESTORE_ROOM", False)
    c = cache()
    a, b = R.fork([c])[0], R.restore([c])[0]
    assert a.compress.raw(0).codes.shape[1] == b.compress.raw(0).codes.shape[1] == N
    assert fp4pool.HITS["room"] == 0
    assert np.array_equal(np.array(a.compress[0]), np.array(b.compress[0]))


# ------------------------------------------------------------------------------------------------ past the room, rollback
def test_growth_past_the_room_is_the_growth_of_today(monkeypatch):
    """Appending past the room grows by the `capacity` rule again, and the layer reads the rows it was given."""
    monkeypatch.setattr(fp4pool, "RESTORE_ROOM", True)
    c = restored(cache(), "hit", monkeypatch, True)
    room = c.compress.raw(0).codes.shape[1]
    more = room - N + 8                                             # over the room in one append
    v = grid(more, 512, 7)
    c.append_compressed(0, v)
    mx.eval(*c.compress.raw_arrays())
    e = c.compress.raw(0)
    assert e.n == room + 8 and e.codes.shape[1] == LC.capacity(e.n, PoolSlots.STEP)
    ref = mx.concatenate([cache().compress[0], v], axis=1)
    assert np.array_equal(bits(c.compress[0][:, :N]), bits(ref[:, :N]))          # the restored rows are the codes they were
    assert np.array_equal(np.array(c.compress[0]), np.array(ref))               # and the rows of the chunk are the ones appended
    assert np.array_equal(np.array(e.codes[:, :e.n]), np.array(fp4pack.pack(ref[:, :e.n], 16, True)[0]))


def test_rollback_after_restore(monkeypatch):
    """A rollback leaves the restored rows: the row count goes back to the restored n (a rejected pass), and the write-back of a stack shorter than the
    layer replaces it with exactly those rows. An append after either lands at the row it should."""
    monkeypatch.setattr(fp4pool, "RESTORE_ROOM", True)
    c = restored(cache(), "hit", monkeypatch, True)
    base = np.array(cache().compress[0])
    cap0 = c.compress.raw(0).codes.shape[1]
    v = grid(8, 512, 11)
    c.append_compressed(0, v)
    mx.eval(*c.compress.raw_arrays())
    assert c.compress.raw(0).n == N + 8
    c.compress.raw(0).n = N                                          # the rollback of a rejected pass (cdecpack.settle does this)
    assert np.array_equal(bits(c.compress[0]), bits(mx.array(base)))
    assert c.compress.raw(0).codes.shape[1] == cap0                   # the buffer, and its room, survive the rollback
    v2 = grid(3, 512, 12)
    c.append_compressed(0, v2)
    mx.eval(*c.compress.raw_arrays())
    assert np.array_equal(bits(c.compress[0][:, :N]), bits(mx.array(base)))
    assert np.array_equal(np.array(c.compress[0]), np.array(mx.concatenate([mx.array(base), v2], axis=1)))

    stack = mx.concatenate([mx.array(base), grid(8, 512, 13)], axis=1)
    c.compress.write_back(0, stack, N)                                # a shorter stack: the layer is those rows only
    mx.eval(*c.compress.raw_arrays())
    e = c.compress.raw(0)
    assert e.n == N and e.codes.shape[1] <= max(cap0, N)
    assert np.array_equal(bits(c.compress[0]), bits(mx.array(base)))
    v3 = grid(4, 512, 14)
    c.append_compressed(0, v3)
    mx.eval(*c.compress.raw_arrays())
    assert np.array_equal(bits(c.compress[0][:, :N]), bits(mx.array(base)))
    assert np.array_equal(np.array(c.compress[0]), np.array(mx.concatenate([mx.array(base), v3], axis=1)))


def test_a_thawed_segment_snapshot_is_not_copied_again(monkeypatch):
    """KYOJIN_KVSEG with pf_grow: the thaw of a segment snapshot already owns a contiguous buffer with room, so the hit takes no second copy."""
    monkeypatch.setattr(fp4pool, "SEG", 32)
    monkeypatch.setattr(fp4pool, "RESTORE_ROOM", True)
    monkeypatch.setattr(DF, "ENABLED", True)                          # the thaw's room comes from the capacity rule (1/8 slack, depthflat)
    cap = LC.capacity(N, PoolSlots.STEP)
    hit = R.restore(R.snapshot([cache()]))[0]
    e = hit.compress.raw(0)
    assert e.owned and e.codes.shape[1] == cap and cap > N             # the thaw's buffer, which has room
    assert fp4pool.HITS["room"] == 0 and fp4pool.HITS["thaw"] > 0
    v = grid(8, 512, 21)
    mx.eval(v)
    kb, _ = peak(lambda: (hit.append_compressed(0, v), mx.eval(*hit.compress.raw_arrays())))
    assert hit.compress.raw(0).codes.shape[1] == cap
    assert kb < 0.5 * e.codes.nbytes / KB                             # no copy of the pool at the first append


# ------------------------------------------------------------------------------------------------ a whole forward through a hit
def test_attention_reads_the_same_values_through_a_hit(monkeypatch):
    """A whole forward from a restored snapshot: the logits of the flag-on hit are the logits of the flag-off hit, bit for bit (the packed pool and index
    keys are what the attention and the indexer read). Tiny random DS4.1, CPU."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M
    monkeypatch.setattr(PoolSlots, "defer_dense", False)             # the model reads the entry an append returns
    monkeypatch.setattr(KeySlots, "defer_dense", False)
    lm, _ = M.make(sliding_window=16)
    ids, more = M.tokens(96, seed=5), M.tokens(24, seed=6)
    out = {}
    for room in (False, True):
        monkeypatch.setattr(fp4pool, "RESTORE_ROOM", room)
        c0 = fp4pool.wrap_cache(lm.make_cache())
        for s in range(0, 64, 32):
            lg = lm(mx.array([ids[s:s + 32]]), cache=c0).logits
        mx.eval(lg, *c0[0].state)
        hit = R.restore(c0)                                         # a prefix hit: the fork of the stored snapshot
        lg = lm(mx.array([more]), cache=hit).logits
        mx.eval(lg, *hit[0].state)
        out[room] = (np.array(lg), {n: slots.raw(2).codes.shape[1] for n, slots in
                                    (("compress", hit[0].compress), ("keys", hit[0].keys))})
    assert fp4pool.HITS["room"] > 0                                  # the hit took the lever
    assert np.array_equal(bits(out[True][0]), bits(out[False][0]))    # same logits
    for name in ("compress", "keys"):                                 # because the code buffers of the hit have room (the read did not change)
        assert out[True][1][name] > out[False][1][name]


# ------------------------------------------------------------------------------------------------ the flag
def test_flag_pf_grow_defaults_off_and_switches_the_module(monkeypatch):
    from kyojin_mlx import flags
    monkeypatch.delenv("KYOJIN_PF_GROW", raising=False)
    assert flags.FLAGS["pf_grow"].env == "KYOJIN_PF_GROW" and flags.FLAGS["pf_grow"].default == "0"
    monkeypatch.setenv("KYOJIN_PF_GROW", "1")
    flags.set("pf_grow", "1")
    assert fp4pool.RESTORE_ROOM is True
    flags.set("pf_grow", "0")
    assert fp4pool.RESTORE_ROOM is False

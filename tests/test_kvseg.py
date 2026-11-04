"""KYOJIN_KVSEG: prefix snapshots of the fp4 pool / index keys share their full segments (fp4pool). CPU only; no Metal kernel is involved.
Bytes are counted with the MLX allocator (mx.get_peak_memory), never timed."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import fp4pack, fp4pool, rows as R  # noqa: E402
from kyojin_mlx.fp4pool import PoolSlots, KeySlots  # noqa: E402

SEG = 32


@pytest.fixture(autouse=True)
def cpu_and_seg(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(fp4pool, "SEG", SEG)
    for k in fp4pool.HITS:
        fp4pool.HITS[k] = 0
    yield
    mx.set_default_device(dev)


def grid(n, D=512, seed=0, block=16, e4=True):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def stock(slots, i):
    e = slots.raw(i)
    return fp4pack.unpack(e.codes[:, :e.n], e.scales[:, :e.n], slots.BLOCK, slots.E4M3, e.dtype)


def synth(rows, cap=None, D=512, cls=PoolSlots):
    """A one-layer slots object holding `rows` rows of all-zero codes (a valid fp4 value), built without packing anything."""
    cap = cap or rows + (rows >> 3) + 256
    blk = cls.BLOCK
    s = cls([None])
    s.set_raw(0, fp4pool._Pk(mx.zeros((1, cap, D // 2), dtype=mx.uint8), mx.zeros((1, cap, D // blk), dtype=mx.uint8), rows, mx.float16))
    mx.eval(s.raw(0).codes, s.raw(0).scales)
    return s


def arrays_ids(slots):
    return {id(a): a.nbytes for a in slots.raw_arrays()}


# ------------------------------------------------------------------------------------------------ slots level
@pytest.mark.parametrize("cls,D,block,e4", [(PoolSlots, 512, 16, True), (KeySlots, 128, 32, False)])
def test_snapshot_reads_back_exact_and_is_independent(cls, D, block, e4):
    s = cls([None])
    for j, m in enumerate([100, 7, 90, 33, 1, 64]):
        s.append_rows(0, grid(m, D, j, block, e4))
    live = s[0]
    snap = s.segment_snapshot()
    assert type(snap).__name__ == "Seg" + cls.__name__ and isinstance(snap, cls)
    assert mx.array_equal(snap[0], live) and mx.array_equal(snap[0], stock(s, 0))
    s.append_rows(0, grid(9, D, 50, block, e4))                       # the live session goes on: the snapshot does not move
    assert mx.array_equal(snap[0], live) and s[0].shape[1] == live.shape[1] + 9
    hit = snap.copy()
    assert type(hit) is cls and mx.array_equal(hit[0], live)
    hit.append_rows(0, grid(5, D, 60, block, e4))
    assert mx.array_equal(hit[0][:, :live.shape[1]], live) and mx.array_equal(hit[0], stock(hit, 0))
    assert mx.array_equal(snap[0], live)
    assert fp4pool.HITS["seal"] > 0 and fp4pool.HITS["thaw"] > 0


def test_prefix_compact_keeps_a_segmented_snapshot_exact_and_shared():
    """kvseg + prefix_compact: rows.compact walks a segmented layer (crash found on the lane 01/11: SegSnapPk has no ndim). Nothing is copied, nothing lazy is left to evaluate."""
    s = PoolSlots([None])
    s.append_rows(0, grid(SEG * 3 + 5, seed=1))
    live = s[0]
    snap = s.segment_snapshot()
    out = []
    new = R.compact(snap, out)
    assert out == [] and new.raw(0) is snap.raw(0) and mx.array_equal(new[0], live)
    assert mx.array_equal(new.copy()[0], live)


@pytest.mark.parametrize("step", [SEG * 2, SEG * 2 + 3])
def test_full_segments_are_shared_by_identity_and_new_bytes_do_not_grow_with_depth(step):
    s = PoolSlots([None])
    s.append_rows(0, grid(SEG * 3 + 5, seed=1))
    prev = s.segment_snapshot()
    news, totals = [], []
    for k in range(14):
        s.append_rows(0, grid(step, seed=10 + k))                       # one "chunk"
        cur = s.segment_snapshot()
        a, b = arrays_ids(prev), arrays_ids(cur)
        news.append(sum(nb for i, nb in b.items() if i not in a))
        totals.append(sum(b.values()))
        pe, ce = prev.raw(0), cur.raw(0)
        assert all(x is y for x, y in zip(pe.segs, ce.segs))           # the segments already full are the same objects
        prev = cur
    assert totals[-1] > 4 * totals[0]                                  # the snapshot itself grows with depth ...
    e = cur.raw(0)
    per_row = (e.segs[0][0].nbytes + e.segs[0][1].nbytes) // SEG
    bound = (step // SEG + 1) * SEG * per_row + SEG * per_row           # ... what it adds: the segments completed by one chunk, the open tail (< SEG rows), one wrap
    assert max(news) <= bound
    if step % SEG == 0:
        assert len(set(news)) == 1                                      # a fixed tail: exactly the same bytes at depth 2 and depth 14
    else:
        assert max(news[7:]) - max(news[:7]) <= SEG * per_row           # a drifting tail never moves the bound with depth


def test_live_append_after_snapshot_does_not_copy_the_buffer(monkeypatch):
    """(The dense mirror is off: its lazy graph, never evaluated here, would hold the code buffer. The served path evaluates it every chunk.)
    The allocator view of the cost: with the flag off the snapshot aliases the live buffer and the next append copies the whole capacity; segmented, it writes in place."""
    from kyojin_mlx import depthflat as DF
    monkeypatch.setattr(DF, "ENABLED", False)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)
    peaks = {}
    for name, seg in (("off", 0), ("on", SEG)):
        fp4pool.SEG = seg
        s = synth(1 << 16)
        full = s.raw(0).codes.nbytes
        snap = s.segment_snapshot() if seg else s.copy()               # rows.snapshot picks one of the two by the flag
        if seg:
            s.raw(0)                                                   # (first snapshot sealed everything; the one below is the steady one)
            snap = s.segment_snapshot()
        mx.eval(*snap.raw_arrays())
        worst = 0
        for j in range(3):                                             # three snapshot + append rounds: the steady state, not just the first append
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            s.append_rows(0, grid(8, seed=3 + j))
            mx.eval(s.raw(0).codes, s.raw(0).scales)
            worst = max(worst, mx.get_peak_memory() - base)
            snap = s.segment_snapshot() if seg else s.copy()
            mx.eval(*snap.raw_arrays())
        peaks[name] = (worst, full)
        del snap
    off, full = peaks["off"]
    on, _ = peaks["on"]
    assert off >= full                                                 # flag off: the whole codes buffer was copied by the append
    assert on < full // 16                                             # flag on: in place (a few rows)


def test_off_grid_and_dense_layers_snapshot_as_before():
    s = PoolSlots([None, grid(10, seed=1)])                            # layer 1 is a dense array; layer 0 empty
    s.append_rows(0, grid(70, seed=2))
    raw = mx.array(np.random.default_rng(9).standard_normal((1, 3, 512)).astype(np.float32)).astype(mx.float16)
    s.append_rows(0, raw)                                              # off the grid: the layer goes dense
    snap = s.segment_snapshot()
    assert mx.array_equal(snap[0], s[0]) and mx.array_equal(snap[1], s[1])
    assert len(snap) == 2


def test_snapshot_survives_the_flag_flipping_to_zero_and_cow_rollback():
    s = PoolSlots([None])
    s.append_rows(0, grid(100, seed=4))
    snap = s.segment_snapshot()
    fp4pool.SEG = 0
    assert mx.array_equal(snap[0], s[0]) and mx.array_equal(snap.copy()[0], s[0])
    assert snap.raw_arrays() and sum(a.nbytes for a in snap.raw_arrays()) > 0
    d = PoolSlots([None])                                              # spec.copy_layer style rollback: set_raw(l, cow(snapshot layer))
    d.set_raw(0, d.cow(snap.raw(0)))
    base = d[0]
    new = grid(6, seed=5)
    d.append_rows(0, new)
    assert mx.array_equal(d[0][:, :100], base) and mx.array_equal(d[0][:, 100:], new) and mx.array_equal(d[0], stock(d, 0))
    assert mx.array_equal(snap[0], base)                               # the stored snapshot did not move
    fp4pool.SEG = SEG
    live = PoolSlots([None])
    live.append_rows(0, grid(100, seed=4))
    snap2 = live.segment_snapshot()
    live.set_raw(0, snap2.cow(snap2.raw(0)))                           # restore in place over a live slot
    live.append_rows(0, new)
    assert mx.array_equal(live[0], mx.concatenate([base, new], axis=1))


def test_growth_realloc_keeps_the_sealed_segments():
    s = PoolSlots([None])
    s.append_rows(0, grid(SEG * 4, seed=1))
    s.segment_snapshot()
    sealed = s.raw(0).sealed
    assert sealed == SEG * 4
    s.append_rows(0, grid(900, seed=2))                                # crosses the capacity: new code buffer
    e = s.raw(0)
    assert e.sealed == sealed and len(e.segs) == 4
    before = fp4pool.HITS["seal"]
    snap = s.segment_snapshot()
    assert fp4pool.HITS["seal"] - before == (SEG * 4 + 900) // SEG - 4  # only the segments completed since: no reseal from row 0
    assert mx.array_equal(snap[0], stock(s, 0))


def test_segment_size_change_restarts_the_seal_and_stays_exact():
    s = PoolSlots([None])
    s.append_rows(0, grid(200, seed=1))
    s.segment_snapshot()
    fp4pool.SEG = 64
    snap = s.segment_snapshot()
    assert s.raw(0).sr == 64 and fp4pool.HITS["reset"] == 1
    assert mx.array_equal(snap[0], stock(s, 0))


# ------------------------------------------------------------------------------------------------ prefix cache ledger
def test_prefix_cache_bills_a_shared_segment_once():
    from kyojin_mlx.serve.engine import PrefixCache
    from kyojin_mlx.serve.backends import Backend

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    class C:                                                            # a cache with the two slot attributes snapshot() looks at
        def __init__(self):
            self.compress, self.keys = PoolSlots([None]), KeySlots([None])

    def run(seg):
        fp4pool.SEG = seg
        pc, c = PrefixCache(B(), 16, 1 << 40), C()
        sizes, uniq = [], []
        for k in range(10):
            c.compress.append_rows(0, grid(SEG * 2 + 3, seed=k))
            c.keys.append_rows(0, grid(SEG * 2 + 3, 128, 100 + k, 32, False))
            pc.store((16 * (k + 1), bytes([k])), [c])
            sizes.append(pc.bytes)
            held = {}
            for e in pc.entries.values():
                held.update(arrays_ids(e[0][0].compress)), held.update(arrays_ids(e[0][0].keys))
            uniq.append(sum(held.values()))
        return pc, sizes, uniq

    pc_on, on, uniq_on = run(SEG)
    assert on == uniq_on                                                # the ledger equals the bytes of the distinct arrays held
    pc_off, off, _ = run(0)
    assert on[-1] < off[-1]                                             # flag off bills every alias whole: quadratic in the depth
    assert (off[-1] - off[0]) > 4 * (on[-1] - on[0]) / 2
    pc_on.shrink(0)
    assert pc_on.bytes == 0 and not pc_on.refs and not pc_on.parts and not pc_on.entries


def test_prefix_cache_eviction_keeps_shared_segments_billed_until_the_last_holder_goes():
    from kyojin_mlx.serve.engine import PrefixCache
    from kyojin_mlx.serve.backends import Backend

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    class C:
        def __init__(self):
            self.compress, self.keys = PoolSlots([None]), KeySlots([None])

    pc, c = PrefixCache(B(), 16, 1 << 40), C()
    for k in range(4):
        c.compress.append_rows(0, grid(SEG * 2 + 3, seed=k))
        pc.store((16 * (k + 1), bytes([k])), [c])
    last = list(pc.entries)[-1]
    first = list(pc.entries)[0]
    full = pc.bytes
    pc.bytes_before = full
    pc.shrink(full - 1)                                                 # drops the oldest only
    assert first not in pc.entries and last in pc.entries
    held = {}
    for e in pc.entries.values():
        held.update(arrays_ids(e[0][0].compress))
    assert pc.bytes == sum(held.values())


# ------------------------------------------------------------------------------------------------ the tiny model: cold, hit, hit again
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402

CHUNK = 64


def prefill_chunks(lm, cache, ids, a, b, snaps=None):
    s, last = a, None
    while s < b:
        e = min(b, (s // CHUNK + 1) * CHUNK)
        last = lm(mx.array([ids[s:e]]), cache=cache).logits
        mx.eval(last, *cache[0].state)
        s = e
        if snaps is not None and s % CHUNK == 0 and s < b:
            snaps[s] = R.snapshot(cache)
    return last[:, -1].astype(mx.float32)


def decode(lm, cache, tok, n=6):
    out = []
    for _ in range(n):
        lg = lm(mx.array([[tok]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg)
        out.append(lg)
        tok = int(mx.argmax(lg).item())
    return out


def new_cache(lm):
    return fp4pool.wrap_cache(lm.make_cache())


def scenario(seg, n=400, ced=False):
    """cold prefill with a snapshot per chunk, then a hit and a hit again from the deepest snapshot, a snapshot of a hit, a second hit with another suffix.
    Sets the flag and the CPU device itself and puts both back (a module-scoped fixture runs before the autouse one, and a leaked flag would reach every later file)."""
    old_seg, old_dev = fp4pool.SEG, mx.default_device()
    mx.set_default_device(mx.cpu)
    fp4pool.SEG = seg
    try:
        return _scenario(seg, n, ced)
    finally:
        fp4pool.SEG = old_seg
        mx.set_default_device(old_dev)


def _scenario(seg, n, ced):
    from kyojin_mlx import ced as C
    from kyojin_mlx.archs import DS41
    lm, _ = M.make(sliding_window=16)
    if ced:
        C.enable(lm, DS41)
    ids = M.tokens(n)
    cache = new_cache(lm)
    snaps = {}
    cold = prefill_chunks(lm, cache, ids, 0, n, snaps)
    b = max(snaps)
    res = {"cold": cold, "cold_dec": decode(lm, cache, int(mx.argmax(cold).item()))}
    for name in ("hit", "hit_again"):
        h = R.fork(snaps[b])
        lg = prefill_chunks(lm, h, ids, b, n)
        res[name] = lg
        res[name + "_dec"] = decode(lm, h, int(mx.argmax(lg).item()))
    mid = sorted(snaps)[len(snaps) // 2]                                # a hit from a middle snapshot with another suffix; its own snapshot; a hit from that
    alt = ids[:mid] + [(t + 7) % 90 for t in ids[mid:]]
    h = R.fork(snaps[mid])
    snaps2 = {}
    lg = prefill_chunks(lm, h, alt, mid, n, snaps2)
    res["alt"] = lg
    h2 = R.fork(snaps2[max(snaps2)])
    res["alt_hit"] = prefill_chunks(lm, h2, alt, max(snaps2), n)
    res["alt_dec"] = decode(lm, h2, int(mx.argmax(res["alt_hit"]).item()))
    res["snap_dense"] = [snaps[k][0].compress[2] for k in sorted(snaps)]            # the stored snapshots read back after all of this
    res["state_after"] = [mx.array(a) for a in cache[0].state]
    return res


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "ced"])
def runs(request):
    return scenario(0, ced=request.param), scenario(SEG, ced=request.param)


def eq(a, b):
    return bool(mx.array_equal(a, b).item())


def test_cold_hit_hit_again_match_and_equal_the_flag_off_path(runs):
    off, on = runs
    for k in ("cold", "hit", "hit_again", "alt", "alt_hit"):
        assert eq(off[k], on[k]), k
    assert eq(on["cold"], on["hit"]) and eq(on["hit"], on["hit_again"]) and eq(on["alt"], on["alt_hit"])
    for k in ("cold_dec", "hit_dec", "hit_again_dec", "alt_dec"):
        assert all(eq(x, y) for x, y in zip(off[k], on[k])), k
    assert all(eq(x, y) for x, y in zip(on["cold_dec"], on["hit_dec"]))
    assert all(eq(x, y) for x, y in zip(off["snap_dense"], on["snap_dense"]))
    assert all(eq(x, y) for x, y in zip(off["state_after"], on["state_after"]))


def test_the_flag_engaged_in_the_model_path(runs):
    """Not vacuous: the tiny model's pool layers are packed _Pk layers, the snapshots were segmented, a hit thawed them."""
    for k in fp4pool.HITS:
        fp4pool.HITS[k] = 0
    scenario(SEG, n=200)
    h = fp4pool.HITS
    assert h["seal"] > 0 and h["snap"] > 0 and h["thaw"] > 0


def test_snapshot_class_and_sharing_in_the_model_path():
    fp4pool.SEG = SEG
    lm, _ = M.make(sliding_window=16)
    ids = M.tokens(300)
    cache = new_cache(lm)
    snaps = {}
    prefill_chunks(lm, cache, ids, 0, 300, snaps)
    ks = sorted(snaps)
    assert type(snaps[ks[0]][0].compress).__name__ == "SegPoolSlots" and type(snaps[ks[0]][0].keys).__name__ == "SegKeySlots"
    a, b = snaps[ks[-2]][0].compress.raw(4), snaps[ks[-1]][0].compress.raw(4)
    assert isinstance(a, fp4pool.SegSnapPk) and len(b.segs) >= len(a.segs) >= 1
    assert all(x is y for x, y in zip(a.segs, b.segs))


def test_kvdisk_files_are_identical_with_the_flag_on_or_off(tmp_path):
    from kyojin_mlx import kvdisk as K

    def run(seg, sub):
        try:
            return _run(seg, sub)
        finally:
            fp4pool.SEG = SEG                                           # the autouse fixture's value; its monkeypatch puts the original back

    def _run(seg, sub):
        fp4pool.SEG = seg
        lm, _ = M.make(sliding_window=16)
        ids = M.tokens(260)
        kv = K.KvDisk(str(tmp_path / sub), {"arch": "dsv4", "ns": "a"}, lm.make_cache, block=64, cap_bytes=0)
        cache = new_cache(lm)
        s, snaps = 0, {}
        while s < 256:
            lm(mx.array([ids[s:s + 64]]), cache=cache)
            mx.eval(*cache[0].state)
            s += 64
            snaps[s] = R.snapshot(cache)
            kv.save(ids, s, cache)
        # a hit from the middle snapshot continues and checkpoints the same blocks again (they are already indexed; the file set must not change)
        h = R.fork(snaps[128])
        lm(mx.array([ids[128:192]]), cache=h)
        mx.eval(*h[0].state)
        kv.save(ids, 192, h)
        out = {}
        for d, _, fs in os.walk(tmp_path / sub):
            for f in fs:
                if f.endswith(K.SUFFIX):
                    out[f] = open(os.path.join(d, f), "rb").read()
        return dict(sorted(out.items()))

    a, b = run(0, "off"), run(SEG, "on")
    assert len(a) == len(b) == 4 and list(a) == list(b)
    assert all(a[f] == b[f] for f in a)                                 # byte for byte: no format change


def test_scenario_puts_the_flag_and_the_device_back():
    before = (fp4pool.SEG, mx.default_device())
    scenario(0, n=200)
    scenario(SEG, n=200)
    assert (fp4pool.SEG, mx.default_device()) == before


def test_session_bytes_include_the_sealed_segments_and_stay_resident_after_the_prefix_cache_lets_go(monkeypatch):
    """The ledger of the prefix cache subtracts a shared segment when its last entry goes, but the live session still holds it: the session's own byte count must say so."""
    from kyojin_mlx import depthflat as DF
    from kyojin_mlx.serve.engine import PrefixCache
    from kyojin_mlx.serve.backends import Backend
    monkeypatch.setattr(DF, "ENABLED", False)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    class C:
        def __init__(self):
            self.compress, self.keys = PoolSlots([None]), KeySlots([None])

    be, c = B(), C()
    pc = PrefixCache(be, 16, 1 << 40)
    c.compress.append_rows(0, grid(SEG * 2, seed=1))
    c.keys.append_rows(0, grid(SEG * 2, 128, 2, 32, False))
    mx.eval(c.compress.raw(0).codes, c.keys.raw(0).codes)
    plain = sum(a.nbytes for a in be.state([c]))
    assert be.nbytes([c]) == plain                                     # nothing sealed yet
    a0 = mx.get_active_memory()
    for k in range(6):
        c.compress.append_rows(0, grid(SEG * 2, seed=10 + k))
        c.keys.append_rows(0, grid(SEG * 2, 128, 20 + k, 32, False))
        mx.eval(c.compress.raw(0).codes, c.keys.raw(0).codes, c.compress.raw(0).scales, c.keys.raw(0).scales)
        pc.store((16 * (k + 1), bytes([k])), [c])
    sealed = sum(seg[0].nbytes + seg[1].nbytes for s in (c.compress, c.keys) for seg in s.raw(0).segs)
    assert sealed > 0 and be.seg_bytes([c]) == sealed
    assert be.nbytes([c]) == sum(a.nbytes for a in be.state([c])) + sealed
    held_by_prefix = pc.bytes
    resident = mx.get_active_memory()
    pc.shrink(0)                                                       # every entry evicted
    assert pc.bytes == 0 and not pc.refs
    dropped = resident - mx.get_active_memory()
    assert dropped < held_by_prefix                                    # the ledger gave back more than the allocator did: the segments are still held by the live session ...
    assert resident - a0 - dropped >= sealed                           # ... and that is exactly what be.nbytes([c]) counts for the session
    snap_segs = [e for e in list.__iter__(R.snapshot([c])[0].compress) if isinstance(e, fp4pool.SegSnapPk)]
    assert snap_segs and be.seg_bytes(R.snapshot([c])) == 0            # a snapshot lists its segments in raw_arrays: no double count inside it
    hit = R.fork(R.snapshot([c]))
    assert be.seg_bytes(hit) == sealed                                 # a hit keeps them too


# ------------------------------------------------------------------------------------------------ the snap_trim / prefix_parts bugs, against kvseg
# Ported from tests/test_mirror_floor.py (75298c4): both fixes are the same guard, a segmented snapshot layer is skipped by the exact-copy paths because
# its segments and tail are already copies of exactly their rows. Here the exact-copy paths are fp4pool.PackedSlots.compacted (prefix_compact, the snap_trim
# of this branch) and the part accounting (backends.seg_parts / kvledger), since snap_trim and kvparts.py do not exist on this branch.
def test_compacted_and_kvseg_together_stay_exact(monkeypatch):
    monkeypatch.setattr(fp4pool, "SEG", SEG)
    c = type("C", (), {})()
    c.compress, c.keys = PoolSlots([None]), KeySlots([None])
    c.compress.append_rows(0, grid(100, seed=1))
    c.keys.append_rows(0, grid(100, 128, 11, 32, False))
    c.compress.append_rows(0, grid(70, seed=9))
    c.keys.append_rows(0, grid(70, 128, 19, 32, False))
    ref = (np.array(c.compress[0]), np.array(c.keys[0]))
    snap = R.snapshot([c], compacted=True)
    assert type(snap[0].compress).__name__.startswith("Seg")
    for name, want in (("compress", ref[0]), ("keys", ref[1])):
        new = getattr(snap[0], name).compacted()                      # the exact-copy path leaves the shared layer alone
        assert new.raw(0) is getattr(snap[0], name).raw(0)
        assert mx.array_equal(new[0], mx.array(want))


def test_nested_snapshots_share_sealed_segments_and_the_ledger_bills_them_once():
    """kvseg + the prefix ledger: every nested snapshot adds the segments sealed since the previous one (by identity) and the ledger bills each one once,
    including after the entry that first billed it is evicted (what the slot_parts + kvseg fix guarantees where parts are accounted)."""
    from kyojin_mlx.serve.engine import PrefixCache
    from kyojin_mlx.serve.backends import Backend

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    class C:
        def __init__(self):
            self.compress, self.keys = PoolSlots([None]), KeySlots([None])

    pc, c = PrefixCache(B(), 16, 1 << 40), C()
    for k in range(6):                                                 # a conversation: each turn appends and stores a nested prefix
        c.compress.append_rows(0, grid(SEG * 3, seed=k))
        c.keys.append_rows(0, grid(SEG * 3, 128, 40 + k, 32, False))
        pc.store((16 * (k + 1), bytes([k])), [c])
    segs = [{id(s) for s in e[0][0].compress.raw(0).segs} for e in pc.entries.values()]
    assert all(a < b for a, b in zip(segs, segs[1:]))                  # each snapshot holds the segments of the one before, by identity
    held = {}
    for e in pc.entries.values():
        for sl in (e[0][0].compress, e[0][0].keys):
            held.update({id(a): a.nbytes for a in sl.raw_arrays()})
    assert pc.bytes == sum(held.values())                              # counted once
    first = next(iter(pc.entries))
    pc.shrink(pc.bytes - 1)                                            # evict the oldest, which was the one billed for the shared segments
    assert first not in pc.entries
    held = {}
    for e in pc.entries.values():
        for sl in (e[0][0].compress, e[0][0].keys):
            held.update({id(a): a.nbytes for a in sl.raw_arrays()})
    assert pc.bytes == sum(held.values())                              # the count did not drop with the entry that first billed them
    pc.shrink(0)
    assert pc.bytes == 0 and not pc.refs

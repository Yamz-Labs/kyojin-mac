"""Disk tier of the prefix cache inside the engine (kyojin_mlx/kvdisk.py behind serve.engine.PrefixCache), on a fake backend whose cache is a real
mlx-lm KVCache (needs mlx and mlx-lm, no model): a checkpoint written by one engine is found by another, a prompt is never prefilled twice, the
result equals the prefill, a broken disk costs a prefill and never the request."""
import mlx.core as mx
import numpy as np
import pytest

cache_mod = pytest.importorskip("mlx_lm.models.cache")
from kyojin_mlx import kvdisk as K  # noqa: E402
from kyojin_mlx.serve.engine import Engine, PrefixCache  # noqa: E402
from test_serve_engine import Fake, collect, nxt, onehot  # noqa: E402


class FakeKV(Fake):
    """Fake whose per-session cache is [KVCache] holding the token history as its keys (so the generic disk adapter writes and reads real arrays)."""
    kind = "fakekv"

    def make_cache(self):
        return [cache_mod.KVCache()]

    @staticmethod
    def hist(cache):
        c = cache[0]
        return [] if c.keys is None else [int(t) for t in c.keys[0, 0, :c.offset, 0].tolist()]

    @staticmethod
    def push(cache, toks):
        k = mx.array([float(t) for t in toks]).reshape(1, 1, -1, 1)
        cache[0].update_and_fetch(k, k)

    def snapshot(self, cache):

        return self.fork(cache)


    def fork(self, cache):
        c = cache_mod.KVCache()
        if cache[0].keys is not None:
            c.keys, c.values, c.offset = cache[0].keys[:, :, :cache[0].offset], cache[0].values[:, :, :cache[0].offset], cache[0].offset
        return [c]

    def prefill_chunk(self, cache, ids):
        self.push(cache, ids)
        return onehot([nxt(self.hist(cache))])

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            self.push(c, [int(t)])
        return onehot([nxt(self.hist(c)) for c in bc])

    def nbytes(self, cache):
        entries = [cache] if isinstance(cache[0], cache_mod.KVCache) else cache                    # one session cache, or a list of them
        return sum(0 if e[0].keys is None else 2 * e[0].keys[:, :, :e[0].offset].nbytes for e in entries)

    def state(self, cache):
        return [] if cache[0].keys is None else [cache[0].keys, cache[0].values]


def disk(tmp_path, block=16, **kw):
    return K.KvDisk(str(tmp_path), {"arch": "fakekv"}, FakeKV().make_cache, block=block, **kw)


def engine(be, d=None, **kw):
    return Engine(be, sessions=2, ctx=4096, chunk=4, block=8, prefix_bytes=kw.pop("prefix_bytes", 1 << 20), disk=d, **kw)


def run(eng, ids, **kw):
    eng.start()
    try:
        return collect(eng.submit(ids, **kw))
    finally:
        eng.stop()


PROMPT = [(i * 7 + 3) % 90 + 1 for i in range(53)]


def test_checkpoint_of_one_engine_is_found_by_the_next(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")                                 # the block grid alone (the tail anchor has its own tests below)
    ref = run(engine(FakeKV()), PROMPT + [5, 6, 7], max_new=12)[0]
    a = engine(FakeKV(), disk(tmp_path))
    out_a = run(a, PROMPT, max_new=12)[0]
    assert a.prefix.disk_saved == 3 and sorted(b for b, _ in a.prefix.disk.index) == [16, 32, 48] and a.stats["prefix_hit_tokens"] == 0
    b = engine(FakeKV(), disk(tmp_path))                                    # a restarted server: new memory tier, same directory
    assert len(b.prefix.disk.index) == 3
    out_b = run(b, PROMPT + [5, 6, 7], max_new=12)[0]
    assert out_b == ref and out_a == run(engine(FakeKV()), PROMPT, max_new=12)[0]
    st = b.status()
    assert b.stats["prefix_hit_tokens"] == 48 and st["disk_hits"] == 1 and st["disk_hit_tokens"] == 48 and b.stats["prefill_tokens"] == 8
    assert (48, K.digest(PROMPT, 48)) in b.prefix.entries                   # kept in memory under the engine's own key (same digest as the streaming hasher)


def test_disk_hit_is_the_state_of_the_prefill(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")
    be = FakeKV()
    a = engine(be, disk(tmp_path))
    run(a, PROMPT, max_new=1)
    p = PrefixCache(be, 8, 1 << 20, disk(tmp_path))
    b, cache = p.lookup(PROMPT)
    assert b == 48 and be.hist(cache) == PROMPT[:48] and cache[0].offset == 48
    cache2 = be.make_cache()
    be.prefill_chunk(cache2, PROMPT[:48])
    assert bool(mx.array_equal(cache[0].keys[:, :, :48], cache2[0].keys[:, :, :48]).item())


def test_longer_disk_checkpoint_beats_a_shorter_memory_snapshot_and_not_the_reverse(tmp_path):
    be = FakeKV()
    d = disk(tmp_path)
    full = be.make_cache()
    for s in range(0, 48, 16):
        be.prefill_chunk(full, PROMPT[s:s + 16])
        d.save(PROMPT, s + 16, full)
    p = PrefixCache(be, 8, 1 << 20, d)
    mem = be.make_cache()
    be.prefill_chunk(mem, PROMPT[:24])
    h = p.hasher()
    h.update(np.asarray(PROMPT[:24], dtype=np.int32).tobytes())
    p.store((24, h.digest()), mem)
    b, cache = p.lookup(PROMPT)
    assert b == 48 and p.disk_hits == 1                                      # disk 48 > memory 24
    p2 = PrefixCache(be, 8, 1 << 20, d)
    p2.store((24, h.digest()), mem)
    b, cache = p2.lookup(PROMPT[:30])
    assert b == 24 and p2.disk_hits == 0                                     # disk has only 16 (<24) below 30: memory wins, nothing loaded


def test_draft_ring_goes_with_the_checkpoint(tmp_path):
    class Ring(FakeKV):
        def draft_snapshot(self, d):
            return list(d["rows"])

        def draft_nbytes(self, d):
            return sum(a.nbytes for a in d["rows"] if a is not None)
    be = Ring()
    d = disk(tmp_path)
    cache = be.make_cache()
    be.prefill_chunk(cache, PROMPT[:16])
    ring = {"rows": [mx.arange(6).reshape(2, 3).astype(mx.bfloat16), None, mx.ones((4,))]}
    p = PrefixCache(be, 8, 1 << 20, d)
    p.save_disk(PROMPT, 16, cache, ring)
    p2 = PrefixCache(be, 8, 1 << 20, disk(tmp_path))
    b, _ = p2.lookup(PROMPT[:20])
    got = p2.last_draft
    assert b == 16 and got[1] is None and got[0].dtype == mx.bfloat16
    assert bool(mx.array_equal(got[0], ring["rows"][0]).item()) and bool(mx.array_equal(got[2], ring["rows"][2]).item())


def test_a_broken_disk_costs_a_prefill_never_the_request(tmp_path):
    class Broken(K.KvDisk):
        def save(self, *a, **k):
            raise OSError("disk full")

        def lookup(self, *a, **k):
            raise OSError("io error")
    d = Broken(str(tmp_path), {"arch": "fakekv"}, FakeKV().make_cache, block=16)
    msgs = []
    eng = engine(FakeKV(), d, log=msgs.append)
    out = run(eng, PROMPT, max_new=9)[0]
    assert out == run(engine(FakeKV()), PROMPT, max_new=9)[0]
    assert eng.prefix.disk_errors >= 4 and any("kv-disk" in m for m in msgs)
    assert eng.status()["disk_errors"] == eng.prefix.disk_errors


def test_disk_block_must_follow_the_chunk(tmp_path):
    with pytest.raises(AssertionError):
        Engine(FakeKV(), sessions=1, ctx=256, chunk=6, block=12, prefix_bytes=0, disk=disk(tmp_path, block=16))


def test_negative_control_wrong_digest_never_hits(tmp_path):
    a = engine(FakeKV(), disk(tmp_path))
    run(a, PROMPT, max_new=1)
    other = list(PROMPT)
    other[3] += 1                                                            # differs inside the first block
    b = engine(FakeKV(), disk(tmp_path))
    run(b, other + [1, 2, 3], max_new=2)
    assert b.stats["prefix_hit_tokens"] == 0 and b.prefix.disk_hits == 0


# ---- kvtail: a prefix shorter than a disk block, and the tail of a continued prompt
def test_short_prefix_reaches_disk_and_is_the_first_hit_after_a_restart(tmp_path):
    a = engine(FakeKV(), disk(tmp_path))
    run(a, PROMPT[:30], max_new=2, cuts=(11,))                               # a 11-token preamble: far under the 16-token disk block
    assert (11, K.digest(PROMPT, 11).hex()) in a.prefix.disk.index and a.prefix.disk_saved == 2
    b = engine(FakeKV(), disk(tmp_path))                                    # restart: empty memory tier, first new session
    other = PROMPT[:11] + [91, 92, 93, 94, 95, 96, 97]
    ref = run(engine(FakeKV()), other, max_new=6, cuts=(11,))[0]
    out = run(b, other, max_new=6, cuts=(11,))[0]
    assert out == ref and b.stats["prefix_hit_tokens"] == 11 and b.prefix.disk_hits == 1 and b.stats["prefill_tokens"] == 7


def test_short_roots_are_pinned_outside_the_cap_and_pins_json_is_left_alone(tmp_path):
    ids = list(range(1, 120))
    d = disk(tmp_path, block=64, cap_bytes=1)                               # a cap nothing fits under
    be = FakeKV()

    def put(n):
        c = be.make_cache()
        be.prefill_chunk(c, ids[:n])
        d.save(ids, n, c)
    put(64)                                                                 # a full block: no pin, evicted at once
    put(11)
    assert (11, K.digest(ids, 11).hex()) in d.index and (64, K.digest(ids, 64).hex()) not in d.index
    assert not (tmp_path / d.ns / "pins.json").exists() and (tmp_path / d.ns / "pins.auto.json").exists()
    for n in range(2, 2 + K.PIN_SHORT + 3):                                  # more short roots than the pin list holds: the oldest fall back to LRU
        put(n)
    assert len(d._read_pins(K.PINS_AUTO)) == K.PIN_SHORT
    assert len(d._pinned()) == K.PIN_SHORT and all(k in d.index for k in d._pinned())


def test_continued_prompt_starts_at_the_tail_anchor_not_the_last_block(tmp_path, monkeypatch):
    cont = PROMPT[:46] + [91, 92, 93, 94, 95, 96, 97, 98, 99]               # the same conversation one turn later: shares 46 tokens, the cut at 47 is not its cut
    ref = run(engine(FakeKV()), cont, max_new=6, cuts=(30,))[0]
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")
    off = engine(FakeKV(), disk(tmp_path / "off"))
    run(off, PROMPT[:50], max_new=2, cuts=(30, 47))
    assert sorted(b for b, _ in off.prefix.disk.index) == [16, 30, 32, 47, 48] and off.stats["kv_tail_saved"] == 0
    off2 = engine(FakeKV(), disk(tmp_path / "off"))
    run(off2, cont, max_new=6, cuts=(30,))
    assert off2.stats["prefix_hit_tokens"] == 32                             # the block below: 14 tokens paid again
    monkeypatch.setenv("KYOJIN_KV_TAIL", "1")
    on = engine(FakeKV(), disk(tmp_path / "on"))
    run(on, PROMPT[:50], max_new=2, cuts=(30, 47))
    assert sorted(b for b, _ in on.prefix.disk.index) == [16, 30, 32, 44, 47, 48] and on.stats["kv_tail_saved"] == 1   # 44: last multiple of the chunk (4) below the last cut
    assert (44, K.digest(PROMPT, 44).hex()) in on.prefix.disk.index
    on2 = engine(FakeKV(), disk(tmp_path / "on"))                           # restart
    out = run(on2, cont, max_new=6, cuts=(30,))[0]
    assert out == ref and on2.stats["prefix_hit_tokens"] == 44 and on2.stats["prefill_tokens"] == len(cont) - 44


def test_tail_anchor_in_memory_and_exact_without_cuts(tmp_path):
    eng = engine(FakeKV())
    eng.start()
    try:
        collect(eng.submit(PROMPT, max_new=3))                               # 53 tokens, chunk 4, memory block 8: snapshots 8..48, anchor 52
        assert (52, K.digest(PROMPT, 52)) in eng.prefix.entries
        cont = PROMPT + [3, 4, 5, 6, 7]
        out = collect(eng.submit(cont, max_new=6))[0]
    finally:
        eng.stop()
    assert out == run(engine(FakeKV()), cont, max_new=6)[0]
    assert eng.stats["prefix_hit_tokens"] == 52                              # not 48
    assert eng.stats["kv_tail_saved"] == 0                                   # no disk tier: nothing written

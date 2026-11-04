"""ced_skip_dec=3: a chunk skips its decoder half and the snapshot / disk checkpoint at its end is still taken, marked stale. On the tiny random DS4.1 text model
(stock mlx-vlm modules, any MLX backend), through the real Engine prefill loop and PrefixCache / KvDisk:
(a) a straight prefill, (b) restore then a long chunk, (c) restore then a short chunk (falls back to an earlier whole snapshot), (d) restore from disk,
(e) a branch inside a skipped span: final logits and every cache buffer the decode reads equal the full prefill. Old checkpoints (no mark) load as whole."""
import json
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C, kvdisk as K, rows as R  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402
from kyojin_mlx.serve.backends import Backend  # noqa: E402
from kyojin_mlx.serve.engine import Engine  # noqa: E402
from test_ced import same, slots_equal  # noqa: E402

TAIL = 16
CHUNK = 32          # the engine grid: chunks of 32 rows (> tail), blocks of 32 (a snapshot at every boundary)


class MiniBackend(Backend):
    """The Engine on the mini model: the same ced hooks as VlmBackend (no drafter, no decode; the tests read the prefill)."""
    kind = "mini"

    def __init__(self, mode):
        self.lm, _ = M.make(sliding_window=TAIL)
        C.enable(self.lm, DS41)
        self.mode = mode
        self.last = None

    def make_cache(self):
        return self.lm.make_cache()

    def snapshot(self, cache):

        return self.fork(cache)


    def fork(self, cache):
        return R.fork(cache)

    def ced_can_skip(self, cache, n):
        return self.mode if self.mode and C.rewrites_decoder(self.lm, cache, n) else 0

    def ced_rewrites(self, rows):
        return C.rewrites_rows(self.lm, rows)

    def prefill_chunk(self, cache, ids, draft=None, skip_dec=False):
        if skip_dec:
            with C.skip_decoder(self.lm):
                self.lm(mx.array(ids)[None], cache=cache)
            mx.eval(*self.state(cache))
            return None
        lg = self.lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg, *self.state(cache))
        self.last = lg
        return lg


def engine(be, disk=None, block=CHUNK, chunk=CHUNK):
    return Engine(be, sessions=1, ctx=4096, chunk=chunk, block=block, prefix_bytes=1 << 28, disk=disk)


def prefill(eng, ids, cuts=()):
    """Run the prompt's prefill through the engine loop; returns (session, last-row logits). The session stays in `ready` with its prefilled cache."""
    eng.submit(list(ids), max_new=4, cuts=cuts)
    s = eng._inbox.pop()
    eng.prefilling.append(s)
    eng.backend.last = None
    while not s.prefilled:
        eng._prefill_step()
    return s, eng.backend.last


def disk_for(be, tmp_path, block=CHUNK):
    return K.KvDisk(str(tmp_path), {"arch": "mini"}, be.make_cache, block=block)


def cache_equal(a, b, n_layers):
    return (slots_equal(a[0], b[0], range(n_layers), slots=("window", "compress", "keys", "kv_state", "score_state")) and a[0].offset == b[0].offset)


def reference(be, ids, cuts=()):
    """The full prefill (mode 0) of `ids` on the same weights."""
    ref = MiniBackend(0)
    ref.lm.update(be.lm.parameters())
    mx.eval(ref.lm.parameters())
    return prefill(engine(ref), ids, cuts)


def pair(mode3_ids, tmp_path=None, **kw):
    be = MiniBackend(3)
    return be, engine(be, **kw)


L = 4 * CHUNK + 20      # chunks 32, 32, 32, 32, 20


def test_a_straight_prefill_skips_and_takes_stale_snapshots():
    ids = M.tokens(L)
    be, eng = pair(None)
    s, lg = prefill(eng, ids)
    ref, lg_ref = reference(be, ids)
    n = be.lm.config.num_hidden_layers
    assert same(lg, lg_ref) and cache_equal(s.cache, ref.cache, n)
    # chunks 0..2 are followed by a 32-row chunk (> tail): skipped; chunk 3 is followed by 20 rows (> tail): skipped too; the last runs whole
    assert eng.stats["pf_skip_dec"] == 4 and eng.stats["pf_skip_snap"] == 4
    assert sorted(k[0] for k in eng.prefix.entries) == [32, 64, 96, 128] and {k[0] for k in eng.prefix.stale} == {32, 64, 96, 128}


def test_default_and_modes_1_2_take_no_stale_snapshot():
    ids = M.tokens(L)
    for mode in (0, 1, 2):
        be = MiniBackend(mode)
        eng = engine(be)
        prefill(eng, ids)
        assert not eng.prefix.stale and "pf_skip_snap" not in eng.stats and "pf_skip_snap_fallback" not in eng.stats
        if mode == 2:
            assert not eng.prefix.entries and eng.stats["pf_skip_dec"] == 4
        if mode == 0:
            assert sorted(k[0] for k in eng.prefix.entries) == [32, 64, 96, 128]


def test_b_restore_then_long_chunk_is_exact():
    ids = M.tokens(L)
    be, eng = pair(None)
    prefill(eng, ids)
    # same prompt again: the lookup returns the stale snapshot at 128 only if the 20 rows after it are a CED chunk (> tail): yes
    s2, lg2 = prefill(eng, ids)
    assert s2.cached == 128 and eng.stats["pf_skip_snap_fallback"] == 0
    ref, lg_ref = reference(be, ids)
    assert same(lg2, lg_ref) and cache_equal(s2.cache, ref.cache, be.lm.config.num_hidden_layers)


def test_c_restore_then_short_chunk_falls_back_to_an_earlier_snapshot():
    ids = M.tokens(L)
    be, eng = pair(None)
    prefill(eng, ids)
    # a longer prompt that ends 8 rows (<= tail) after the 128 boundary: the stale snapshot at 128 is refused; 96 is stale too and followed by a 32-row chunk: used
    short = ids[:128] + M.tokens(8, seed=5)
    s2, lg2 = prefill(eng, short)
    assert s2.cached == 96 and eng.stats["pf_skip_snap_fallback"] == 1
    ref, lg_ref = reference(be, short)
    assert same(lg2, lg_ref) and cache_equal(s2.cache, ref.cache, be.lm.config.num_hidden_layers)
    # with nothing whole before it either: a prefill from 0
    be2, eng2 = pair(None)
    ids2 = M.tokens(CHUNK + 8, seed=7)
    s0, _ = prefill(eng2, ids2)                   # 32 skipped (8 rows follow: not > tail -> whole); so build one stale at 32 by a longer prompt first
    eng3 = engine(be2)
    prefill(eng3, M.tokens(2 * CHUNK + 20, seed=7))         # snapshots at 32 (stale: next chunk 32 > tail) and 64 (stale: next 20 > tail)
    assert {k[0] for k in eng3.prefix.stale} == {32, 64}
    probe = M.tokens(2 * CHUNK + 20, seed=7)[:CHUNK] + M.tokens(8, seed=9)       # 32 + 8 rows: the snapshot at 32 is refused, nothing earlier
    s3, lg3 = prefill(eng3, probe)
    assert s3.cached == 0 and eng3.stats["pf_skip_snap_fallback"] == 1
    ref3, lg_ref3 = reference(be2, probe)
    assert same(lg3, lg_ref3) and cache_equal(s3.cache, ref3.cache, be2.lm.config.num_hidden_layers)


def test_d_restore_from_disk(tmp_path):
    ids = M.tokens(L)
    be = MiniBackend(3)
    disk = disk_for(be, tmp_path)
    eng = engine(be, disk=disk)
    prefill(eng, ids)
    assert eng.stats["pf_skip_snap_disk"] == 4 and all(e["stale"] for e in disk.index.values())
    # a fresh engine (restart): memory tier empty, the disk answers
    eng2 = engine(be, disk=disk_for(be, tmp_path))
    s2, lg2 = prefill(eng2, ids)
    ref, lg_ref = reference(be, ids)
    n = be.lm.config.num_hidden_layers
    assert s2.cached == 128 and same(lg2, lg_ref) and cache_equal(s2.cache, ref.cache, n)
    # a short chunk after the newest checkpoint: the disk falls back to the nearest earlier file whose follow-up chunk is long
    eng3 = engine(be, disk=disk_for(be, tmp_path))
    short = ids[:128] + M.tokens(8, seed=5)
    s3, lg3 = prefill(eng3, short)
    ref3, lg_ref3 = reference(be, short)
    assert s3.cached == 96 and eng3.stats["pf_skip_snap_fallback"] == 1
    assert same(lg3, lg_ref3) and cache_equal(s3.cache, ref3.cache, n)


def test_e_branch_inside_a_skipped_span():
    ids = M.tokens(L)
    be, eng = pair(None)
    prefill(eng, ids)
    n = be.lm.config.num_hidden_layers
    for split in (70, 100, 112, 125):                   # inside the skipped chunks (64..96, 96..128), shorter and longer tails
        branch = ids[:split] + M.tokens(40, seed=split)
        s, lg = prefill(eng, branch)
        ref, lg_ref = reference(be, branch)
        assert same(lg, lg_ref) and cache_equal(s.cache, ref.cache, n), split
    # the branch ends a few rows into the span (the chunk after the stale snapshot at 96 is short AND last): the snapshot must be refused
    for split, extra in ((100, 5), (97, 3), (66, 4)):
        branch = ids[:split] + M.tokens(extra, seed=split)
        s, lg = prefill(eng, branch)
        ref, lg_ref = reference(be, branch)
        assert same(lg, lg_ref) and cache_equal(s.cache, ref.cache, n), (split, extra)
    # a cut inside the span changes the grid after the snapshot: the chunk after it is short, the snapshot is refused
    branch = ids[:96] + M.tokens(40, seed=3)
    s, lg = prefill(eng, branch, cuts=(100,))
    ref, lg_ref = reference(be, branch, cuts=(100,))
    assert same(lg, lg_ref) and cache_equal(s.cache, ref.cache, n)


def test_stale_mark_survives_the_disk_format_and_old_checkpoints_load(tmp_path):
    ids = M.tokens(L)
    be = MiniBackend(3)
    disk = disk_for(be, tmp_path)
    prefill(engine(be, disk=disk), ids)
    files = [e["path"] for e in disk.index.values()]
    assert files and all(K._header(p).get("stale") for p in files)
    # an old checkpoint: no "stale" key in the header (written before mode 3): whole, handed out without any predicate
    be0 = MiniBackend(0)
    d0 = disk_for(be0, tmp_path / "old")
    prefill(engine(be0, disk=d0), ids)
    assert not any("stale" in K._header(e["path"]) for e in d0.index.values())
    d0b = disk_for(be0, tmp_path / "old")
    assert not any(e["stale"] for e in d0b.index.values())
    b, cache, _ = d0b.lookup(ids)
    assert b == 128 and not d0b.last_stale and d0b.refused == 0
    # a stale file is not handed out without a predicate, and `ok` decides
    d1 = disk_for(be, tmp_path)
    assert d1.lookup(ids)[0] == 0 and d1.refused == 4
    assert d1.lookup(ids, ok=lambda b: b <= 96)[0] == 96 and d1.last_stale


def test_a_whole_snapshot_replaces_a_stale_one():
    ids = M.tokens(L)
    be, eng = pair(None)
    prefill(eng, ids)
    key = next(iter(eng.prefix.entries))
    assert key in eng.prefix.stale
    nb = eng.prefix.bytes
    s_cache = be.make_cache()
    eng.prefix.store(key, s_cache, None, stale=True)
    assert key in eng.prefix.stale and eng.prefix.bytes == nb              # a stale store never replaces
    eng.prefix.store(key, s_cache, None)
    assert key not in eng.prefix.stale and key in eng.prefix.entries


def test_a_stale_snapshot_followed_by_a_short_last_chunk_would_be_wrong():
    """Why the lookup refuses: the same restore with the predicate forced open gives other logits."""
    ids = M.tokens(L)
    be, eng = pair(None)
    prefill(eng, ids)
    short = ids[:128] + M.tokens(8, seed=5)
    b, cache = eng.prefix.lookup(short, ok=lambda b: True)
    assert b == 128
    lg = be.lm(mx.array(short[128:])[None], cache=cache).logits[:, -1].astype(mx.float32)
    ref, lg_ref = reference(be, short)
    assert not same(lg, lg_ref)

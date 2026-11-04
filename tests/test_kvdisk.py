"""Disk KV checkpoints (kyojin_mlx/kvdisk.py) on the tiny random DS4.1 model and on mlx-lm cache objects: bit-exact reload, chain, eviction, damage."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C, kvdisk as K  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402


def snapshot(cache):
    return [mx.array(a) for a in cache[0].state]


def eq_arrays(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype and bool(mx.array_equal(x, y).item()) for x, y in zip(a, b))


def prefill_save(lm, kv, ids, chunk, upto, extras_fn=None):
    """Prefill ids[:upto] in chunks, checkpoint at every block boundary; returns (cache, {b: state snapshot})."""
    cache = lm.make_cache()
    snaps, s = {}, 0
    while s < upto:
        e = min(upto, (s // chunk + 1) * chunk)
        out = lm(mx.array([ids[s:e]]), cache=cache).logits
        mx.eval(out, *cache[0].state)
        s = e
        if kv is not None and s % kv.block == 0:
            snaps[s] = snapshot(cache)
            kv.save(ids, s, cache, extras=extras_fn(s) if extras_fn else None)
    return cache, snaps


def continue_from(lm, cache, ids, start, chunk):
    s, last = start, None
    while s < len(ids):
        e = min(len(ids), (s // chunk + 1) * chunk)
        last = lm(mx.array([ids[s:e]]), cache=cache).logits
        mx.eval(last)
        s = e
    return last[:, -1].astype(mx.float32)


def decode(lm, cache, tok, n=5):
    out = []
    for _ in range(n):
        lg = lm(mx.array([[tok]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg)
        out.append(lg)
        tok = int(mx.argmax(lg).item())
    return out


def mk(tmp_path, lm, block, ns="a", cap=0, **kw):
    return K.KvDisk(str(tmp_path), {"arch": "dsv4", "ns": ns}, lm.make_cache, block=block, cap_bytes=cap, **kw)


@pytest.mark.parametrize("ced", [False, True])
@pytest.mark.parametrize("block,chunk,n", [(32, 32, 100), (33, 33, 100), (32, 16, 70)])
def test_reload_is_bit_exact_and_continues_identically(tmp_path, ced, block, chunk, n):
    ids = M.tokens(n)
    lm, cfg = M.make(sliding_window=16)
    if ced:
        C.enable(lm, DS41)
    kv = mk(tmp_path, lm, block)
    ref_cache, snaps = prefill_save(lm, kv, ids, chunk, n)               # also continues to the end
    top = max(b for b in snaps if b < n)
    b, cache, extras = kv.lookup(ids)
    assert b == top and extras is None
    assert cache[0].offset == b
    got = snapshot(cache)
    want = snaps[b]
    # all slots + engram; the four trailing handoffs are republished by the next forward
    nstate = len(want) - len(type(cache[0]).HANDOFF)
    assert eq_arrays(got[:nstate], want[:nstate])
    # next-token logits and a few decode steps: identical to a cache that never went to disk
    c_ref, _ = prefill_save(lm, None, ids, chunk, b)
    lg_ref = continue_from(lm, c_ref, ids, b, chunk)
    lg_new = continue_from(lm, cache, ids, b, chunk)
    assert bool(mx.array_equal(lg_ref, lg_new).item())
    t = int(mx.argmax(lg_ref).item())
    for x, y in zip(decode(lm, c_ref, t), decode(lm, cache, t)):
        assert bool(mx.array_equal(x, y).item())
    assert cache[0].offset == c_ref[0].offset == n + 5


def test_chain_of_blocks_and_engram_history(tmp_path):
    ids = M.tokens(200)
    lm, cfg = M.make(sliding_window=16, engram_layer_ids=[], engram_num_embeddings=[])
    kv = mk(tmp_path, lm, 32)
    _, snaps = prefill_save(lm, kv, ids, 32, 196)
    assert sorted(snaps) == [32, 64, 96, 128, 160, 192]
    assert len(kv.index) == 6
    b, cache, _ = kv.lookup(ids)                        # 200 tokens: longest proper prefix 192
    assert b == 192
    # a shorter query of the same prefix uses a shorter chain
    b2, cache2, _ = kv.lookup(ids[:100])
    assert b2 == 96 and cache2[0].offset == 96
    assert eq_arrays(snapshot(cache2)[:-4], snaps[96][:-4])
    # a different continuation after 64 tokens shares the first two files
    other = ids[:64] + [(t + 1) % 96 for t in ids[64:130]]
    b3, _, _ = kv.lookup(other)
    assert b3 == 64


def test_block_files_hold_only_the_new_rows(tmp_path):
    ids = M.tokens(130)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 64)
    prefill_save(lm, kv, ids, 64, 128)
    first, second = sorted(kv.index.items())
    assert first[1]["rows"]["compress.2"] == 32 and second[1]["rows"]["compress.2"] == 64            # ratio 2: 64 tokens = 32 rows per block
    assert first[1]["rows"]["compress.4"] == 64 and second[1]["rows"]["compress.4"] == 128           # the ratio-1 midpoint
    t1 = mx.load(first[1]["path"])
    t2 = mx.load(second[1]["path"])
    assert t1["g.compress.2"].shape[1] == 32 and t2["g.compress.2"].shape[1] == 32
    assert t2["g.keys.4"].shape[1] == 64 and t2["g.keys.2"].shape[1] == 32
    # fixed state is the same size in every file, so a block costs the same at any depth
    assert t1["f.window.5"].shape == t2["f.window.5"].shape


def test_extras_roundtrip_with_none(tmp_path):
    ids = M.tokens(70)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    ring = [mx.random.normal((1, 8, 16)).astype(mx.bfloat16), None, mx.arange(6).reshape(2, 3)]
    prefill_save(lm, kv, ids, 32, 64, extras_fn=lambda b: ring)
    b, _, ex = kv.lookup(ids)
    assert b == 64 and len(ex) == 3 and ex[1] is None
    assert ex[0].dtype == mx.bfloat16 and bool(mx.array_equal(ex[0], ring[0]).item()) and bool(mx.array_equal(ex[2], ring[2]).item())


def test_missing_parent_is_skipped_and_wrong_boundary_refused(tmp_path):
    ids = M.tokens(100)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    cache, _ = prefill_save(lm, None, ids, 32, 64)
    assert kv.save(ids, 64, cache) == 0 and kv.stats["skipped"] == 1 and not kv.index         # no file for 32: the chain would be unloadable
    assert kv.save(ids, 48, cache) == 0 and kv.stats["skipped"] == 2                           # an off-grid cut needs the block below it on disk
    with pytest.raises(ValueError):
        kv.save(ids, 200, cache)                                                                 # past the prompt
    with pytest.raises(ValueError):
        kv.save(ids, 32, cache)                                                                  # cache holds 64 tokens
    assert kv.lookup(ids) == (0, None, None)


def test_restart_reads_the_directory_and_drops_partial_files(tmp_path):
    ids = M.tokens(100)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    _, snaps = prefill_save(lm, kv, ids, 32, 96)
    open(os.path.join(kv.dir, ".dead" + K.TMP), "w").write("junk")
    kv2 = mk(tmp_path, lm, 32)
    assert len(kv2.index) == 3 and not any(f.endswith(K.TMP) for f in os.listdir(kv2.dir))
    b, cache, _ = kv2.lookup(ids)
    assert b == 96 and eq_arrays(snapshot(cache)[:-4], snaps[96][:-4])


def test_namespace_separates_caches(tmp_path):
    ids = M.tokens(70)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32, ns="exact")
    prefill_save(lm, kv, ids, 32, 64)
    assert mk(tmp_path, lm, 32, ns="ced").lookup(ids)[0] == 0
    assert mk(tmp_path, lm, 32, ns="exact").lookup(ids)[0] == 64
    assert mk(tmp_path, lm, 16, ns="exact").lookup(ids)[0] == 0                                  # another block size is another cache


def test_damaged_file_falls_back_to_a_shorter_chain(tmp_path):
    ids = M.tokens(120)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    _, snaps = prefill_save(lm, kv, ids, 32, 96)
    top = max(kv.index)
    path = kv.index[top]["path"]
    data = open(path, "rb").read()
    open(path, "wb").write(data[: len(data) // 2])                                               # truncated write
    b, cache, _ = kv.lookup(ids)
    assert b == 64 and not os.path.exists(path) and eq_arrays(snapshot(cache)[:-4], snaps[64][:-4])
    assert kv.stats["bad"] >= 1


def test_lru_cap_evicts_leaves_first_and_keeps_chains_loadable(tmp_path):
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    A = M.tokens(100, seed=1)
    B = M.tokens(100, seed=2)
    prefill_save(lm, kv, A, 32, 96)
    one = kv.bytes() // 3
    kv.cap = int(one * 4.5)
    prefill_save(lm, kv, B, 32, 96)                       # 6 files do not fit: the oldest leaves go
    assert kv.bytes() <= kv.cap and kv.stats["evicted"] >= 2
    for k, e in kv.index.items():
        assert e["parent"] is None or e["parent"] in kv.index                                    # never a child without its parent
    ba = kv.lookup(A)[0]
    bb = kv.lookup(B)[0]
    assert bb == 96 and ba in (0, 32, 64)
    # a use refreshes the whole chain: A's chain is the most recent now, so a new prompt evicts B first
    kv.lookup(A)
    C_ = M.tokens(100, seed=3)
    prefill_save(lm, kv, C_, 32, 96)
    assert kv.bytes() <= kv.cap


def flat(c):
    from mlx.utils import tree_flatten
    if type(c).__name__ == "KVCache":
        return [mx.array(c.keys[:, :, :c.offset]), mx.array(c.values[:, :, :c.offset])]
    return [mx.array(a) for _, a in tree_flatten(c.state) if isinstance(a, mx.array)]


def test_generic_adapter_mlx_lm_caches(tmp_path):
    cache_mod = pytest.importorskip("mlx_lm.models.cache")
    caches = [cache_mod.KVCache(), cache_mod.ArraysCache(size=2), cache_mod.RotatingKVCache(max_size=8)]

    def fill(n):
        mx.random.seed(n)
        caches[0].update_and_fetch(mx.random.normal((1, 2, n, 4)).astype(mx.bfloat16), mx.random.normal((1, 2, n, 4)).astype(mx.bfloat16))
        caches[1].cache = [mx.random.normal((1, 3, 4)), mx.random.normal((1, 5))]
        caches[2].update_and_fetch(mx.random.normal((1, 2, n, 4)), mx.random.normal((1, 2, n, 4)))
        mx.eval([c.state for c in caches])

    fill(40)
    snap = [flat(c) for c in caches]
    metas = [getattr(c, "meta_state", None) for c in caches]
    ids = list(range(1, 60))
    kv = K.KvDisk(str(tmp_path), {"arch": "generic"}, lambda: [cache_mod.KVCache(), cache_mod.ArraysCache(size=2), cache_mod.RotatingKVCache(max_size=8)],
                  block=40)
    assert kv.save(ids, 40, caches) > 0
    b, got, _ = kv.lookup(ids)
    assert b == 40
    for c, s, m in zip(got, snap, metas):
        assert eq_arrays(flat(c), s)
        assert getattr(c, "meta_state", None) == m
    assert got[0].offset == 40 and (got[2].offset, got[2].keep, got[2].max_size, got[2]._idx) == (caches[2].offset, caches[2].keep, caches[2].max_size, caches[2]._idx)
    # second block grows the KVCache part by rows only
    mx.random.seed(1)
    caches[0].update_and_fetch(mx.random.normal((1, 2, 40, 4)).astype(mx.bfloat16), mx.random.normal((1, 2, 40, 4)).astype(mx.bfloat16))
    caches[1].cache = [mx.random.normal((1, 3, 4)), mx.random.normal((1, 5))]
    caches[2].update_and_fetch(mx.random.normal((1, 2, 40, 4)), mx.random.normal((1, 2, 40, 4)))
    mx.eval([c.state for c in caches])
    ids2 = list(range(1, 100))
    ids2[:40] = ids[:40]
    assert kv.save(ids2, 80, caches) > 0
    sizes = sorted(e["size"] for e in kv.index.values())
    assert sizes[1] < 2 * sizes[0]
    b, got, _ = kv.lookup(ids2)
    assert b == 80 and eq_arrays(flat(got[0]), flat(caches[0])) and got[0].offset == 80


def test_lookup_above_and_below_bound_the_prefix(tmp_path):
    ids = M.tokens(200)
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    prefill_save(lm, kv, ids, 32, 128)                                            # files at 32, 64, 96, 128
    assert [kv.lookup(ids[:n])[0] for n in (33, 65, 129, 200)] == [32, 64, 128, 128]
    assert kv.lookup(ids, above=96)[0] == 128 and kv.lookup(ids, above=128) == (0, None, None)      # nothing longer than 128 on disk
    assert kv.lookup(ids, below=128)[0] == 96 and kv.lookup(ids, below=128, above=96) == (0, None, None)
    assert kv.lookup(ids, above=64, below=128)[0] == 96


@pytest.mark.parametrize("ced", [False, True])
def test_prefix_snapshot_continues_identically(ced):
    from kyojin_mlx import fp4pool as FP, rows as R
    n, b, chunk = 400, 256, 64
    ids = M.tokens(n)
    lm, cfg = M.make(sliding_window=16)
    if ced:
        C.enable(lm, DS41)
    def run(upto, cache=None, s=0):
        cache = cache if cache is not None else FP.wrap_cache(lm.make_cache())
        last = None
        while s < upto:
            e = min(upto, (s // chunk + 1) * chunk)
            last = lm(mx.array([ids[s:e]]), cache=cache).logits
            mx.eval(last, *cache[0].state)
            s = e
        return cache, last
    c, _ = run(b)
    snap = R.snapshot(c)
    assert snap[0].compress_kv is None and snap[0].index_k is None
    live, want = run(n, c, b)                                           # the session that took the snapshot goes on, as served
    hit, got = run(n, R.fork(snap), b)
    assert bool(mx.array_equal(want[:, -1], got[:, -1]).item())
    t = int(mx.argmax(want[:, -1]).item())
    for x, y in zip(decode(lm, live, t), decode(lm, hit, t)):
        assert bool(mx.array_equal(x, y).item())


def test_pinned_chain_survives_the_cap_and_does_not_count(tmp_path):
    import json
    lm, cfg = M.make(sliding_window=16)
    kv = mk(tmp_path, lm, 32)
    A = M.tokens(100, seed=1)
    B = M.tokens(100, seed=2)
    prefill_save(lm, kv, A, 32, 96)
    one = kv.bytes() // 3
    with open(os.path.join(kv.dir, K.PINS), "w") as f:
        json.dump([[96, K.digest(A, 96).hex()]], f)
    kv.cap = int(one * 1.5)
    prefill_save(lm, kv, B, 32, 96)                       # A is pinned: only B's files compete for the cap
    assert kv.lookup(A)[0] == 96
    pinned = kv._pinned()
    assert len(pinned) == 3 and kv.bytes() - sum(kv.index[k]["size"] for k in pinned) <= kv.cap and kv.stats["evicted"] >= 1
    os.unlink(os.path.join(kv.dir, K.PINS))               # no pin: the same store falls back to plain LRU
    prefill_save(lm, kv, B, 32, 96)
    assert kv.bytes() <= kv.cap


def test_kv_disk_off_hides_the_tier_without_a_restart(monkeypatch):
    from kyojin_mlx.serve.engine import PrefixCache
    marker = object()
    p = PrefixCache(None, 32, 0, disk=marker)
    monkeypatch.setenv("KYOJIN_KV_DISK_OFF", "1")
    assert p.disk is None
    monkeypatch.setenv("KYOJIN_KV_DISK_OFF", "0")
    assert p.disk is marker

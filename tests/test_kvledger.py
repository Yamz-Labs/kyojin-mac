"""ckvseg: where the live KV bytes go (kvledger.itemize), the dead cross-layer handoff (flag kv_handoff_free), the sealed-segment bill (flag kv_seg_once) and a prefix store
that grows with new rows only (kvseg). CPU only, tiny random DS4.1 (tests/mini_ds41.py); bytes are array sizes and the MLX allocator, never timed."""
import os
import sys
import types

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import fp4pool as FP, kvledger as KL, rows as R  # noqa: E402
from kyojin_mlx.serve.backends import Backend, VlmBackend, arrays_of  # noqa: E402

CHUNK = 64


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    for k in ("KYOJIN_KV_HANDOFF_FREE", "KYOJIN_KV_SEG_ONCE"):
        monkeypatch.delenv(k, raising=False)
    yield
    mx.set_default_device(dev)


def prefill(lm, ids, cache, free=False, hook=None):
    last = None
    for s in range(0, len(ids), CHUNK):
        last = lm(mx.array([ids[s:s + CHUNK]]), cache=cache).logits
        mx.eval(last, *cache[0].state)
        if free:
            KL.free_handoff(cache)
        if hook:
            hook(s + CHUNK)
    return last


def test_flags_default_on_and_listed():
    from kyojin_mlx import flags
    for n in ("kv_handoff_free", "kv_seg_once"):
        assert flags.FLAGS[n].default == "1"                 # served defaults: the resumed-run bytes of docs/reports (kv 5.86 -> 4.67 GiB at 1M)
    assert KL.enabled()
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "cloud", "LEVERS.md")).read()
    assert "KYOJIN_KV_HANDOFF_FREE" in doc and "KYOJIN_KV_SEG_ONCE" in doc


def test_itemize_adds_up_to_the_bill_and_names_the_handoff():
    lm, _ = M.make(sliding_window=16)
    n = 1024
    cache = FP.wrap_cache(lm.make_cache())
    prefill(lm, M.tokens(n), cache)
    rows = KL.itemize(cache)
    bill = sum(a.nbytes for a in arrays_of(cache))
    assert sum(r["nbytes"] for r in rows if r["counted"]) == bill              # every counted byte is named, none twice
    s = KL.summary(cache, n)
    print("\nbytes/token by role (tiny model, fp32):", s["per_token"])
    assert s["handoff"] == KL.handoff_bytes(cache) > 0
    assert s["packed"] > 0 and s["state"] > 0
    paths = {r["path"].split(".", 1)[1] for r in rows if r["role"] == "handoff"}
    assert paths == set(KL.HANDOFF)
    assert all(r["counted"] for r in rows if r["role"] in ("packed", "handoff", "state"))
    assert all(not r["counted"] for r in rows if r["role"] in ("slack", "mirror", "memo"))


def test_free_handoff_is_exact_and_removes_exactly_the_handoff_from_the_bill():
    lm, _ = M.make(sliding_window=16)
    ids, more = M.tokens(768), M.tokens(256, seed=7)
    a, b = FP.wrap_cache(lm.make_cache()), FP.wrap_cache(lm.make_cache())
    prefill(lm, ids, a)
    prefill(lm, ids, b, free=True)
    bill_a, bill_b = sum(x.nbytes for x in arrays_of(a)), sum(x.nbytes for x in arrays_of(b))
    freed = KL.handoff_bytes(a)
    assert KL.handoff_bytes(b) == 0 and freed > 0
    assert bill_a - bill_b == freed                                           # the bill falls by the handoff and by nothing else
    print("\nlive bill %d -> %d B (%.1f%%), handoff %d B" % (bill_a, bill_b, 100 * (bill_a - bill_b) / bill_a, freed))
    for s in range(0, 256, CHUNK):                                            # continue the prefill, then decode: same bits as with the handoff kept
        x = lm(mx.array([more[s:s + CHUNK]]), cache=a).logits
        y = lm(mx.array([more[s:s + CHUNK]]), cache=b).logits
        mx.eval(x, y)
        assert bool(mx.array_equal(x, y).item())
    for t in more[:12]:
        x = lm(mx.array([[t]]), cache=a).logits
        y = lm(mx.array([[t]]), cache=b).logits
        mx.eval(x, y)
        assert bool(mx.array_equal(x, y).item())
    for ca, cb in zip(a, b):
        for k in ("window", "kv_state", "score_state"):
            for u, v in zip(getattr(ca, k), getattr(cb, k)):
                if u is not None:
                    assert bool(mx.array_equal(u, v).item())
        for k in ("compress", "keys"):
            for i in range(len(getattr(ca, k))):
                u, v = getattr(ca, k)[i], getattr(cb, k)[i]
                if u is not None:
                    assert bool(mx.array_equal(u, v).item())


def test_backend_prefill_chunk_obeys_the_flag(monkeypatch):
    from kyojin_mlx import headlast
    lm, _ = M.make(sliding_window=16)
    fake = types.SimpleNamespace(lm=lm, HL=headlast, state=lambda c: arrays_of(c), _ring=lambda d: [],
                                 _forward_kw=lambda ids, vl=None: {"input_ids": mx.array(ids)[None]})
    fake._free_handoff = lambda c: VlmBackend._free_handoff(fake, c)
    ids = M.tokens(256)
    outs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("KYOJIN_KV_HANDOFF_FREE", flag)
        cache = FP.wrap_cache(lm.make_cache())
        for s in range(0, 256, CHUNK):
            lg = VlmBackend.prefill_chunk(fake, cache, ids[s:s + CHUNK])
        outs[flag] = (lg, KL.handoff_bytes(cache))
    assert outs["0"][1] > 0 and outs["1"][1] == 0
    assert bool(mx.array_equal(outs["0"][0], outs["1"][0]).item())


def test_handoff_free_retention_on_the_allocator():
    """Retained bytes (MLX active memory), not only the bill: what only the handoff pinned goes."""
    import gc
    lm, _ = M.make(sliding_window=16)
    ids = M.tokens(1024)
    got = {}
    for free in (False, True):
        gc.collect(); mx.eval()
        b0 = mx.get_active_memory()
        cache = FP.wrap_cache(lm.make_cache())
        prefill(lm, ids, cache, free=free)
        gc.collect(); mx.eval()
        got[free] = mx.get_active_memory() - b0
        del cache
    print("\nretained on the allocator: keep %d B, free %d B" % (got[False], got[True]))
    assert got[True] <= got[False]


# ------------------------------------------------------------------------------------------------ prefix store
def _store(monkeypatch, seg=64, turns=6, per=192):
    from kyojin_mlx.serve.engine import PrefixCache
    monkeypatch.setattr(FP, "SEG", seg)

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    lm, _ = M.make(sliding_window=16)
    ids = M.tokens(per * turns)
    cache = FP.wrap_cache(lm.make_cache())
    pc = PrefixCache(B(), per, 1 << 40)
    sizes, s = [], 0
    for k in range(turns):
        while s < per * (k + 1):
            lm(mx.array([ids[s:s + CHUNK]]), cache=cache)
            mx.eval(*cache[0].state)
            s += CHUNK
        pc.store((s, bytes([k])), cache)
        sizes.append(pc.bytes)
    return pc, cache, sizes


def test_nested_entries_share_rows_and_the_store_grows_with_new_rows(monkeypatch):
    pc, cache, sizes = _store(monkeypatch)
    entries = list(pc.entries.values())
    assert len(entries) == 6
    unique = {}
    for e in entries:
        for a in arrays_of(e[0]):
            unique[id(a)] = a.nbytes
    each = [sum(a.nbytes for a in arrays_of(e[0])) for e in entries]
    print("\nstore bytes after each entry:", sizes, "sum of entry sizes:", sum(each))
    assert pc.bytes == sum(unique.values())                                   # every shared segment counted once
    assert pc.bytes < 0.75 * sum(each)                                        # and the entries really share (6 nested prefixes)
    steps = [b - a for a, b in zip(sizes, sizes[1:])]
    # each entry adds the rows sealed since the previous one plus a private part (window ring, states, open tail) that does not depend on the context length:
    # the steps are flat (they differ only by the segment grain), while the entries themselves grow with the context
    assert all(x > 0 for x in steps) and max(steps) < 1.1 * min(steps)
    assert each[-1] > 2 * each[0] and pc.bytes < 0.6 * sum(each)


def test_kv_seg_once_bills_a_shared_segment_one_time(monkeypatch):
    pc, cache, sizes = _store(monkeypatch)
    from kyojin_mlx.serve.engine import Engine
    sess = types.SimpleNamespace(cache=cache, kv_bytes=Backend().nbytes(cache), share_seed=None)
    eng = types.SimpleNamespace(prefix=pc, bc_bytes=0, bc=None, prefilling=[], ready=[sess], members=[], waiting=[])
    monkeypatch.setenv("KYOJIN_KV_SEG_ONCE", "0")                     # the lever default flipped to 1: name the off arm
    off = Engine.kv_bytes(eng)
    monkeypatch.setenv("KYOJIN_KV_SEG_ONCE", "1")
    on = Engine.kv_bytes(eng)
    sealed = sum(e.sb for sl in (cache[0].compress, cache[0].keys) for e in list.__iter__(sl) if isinstance(e, FP._Pk))
    billed = {i: r[1] for i, r in pc.refs.items()}
    overlap = sum(nb for sl in (cache[0].compress, cache[0].keys) for e in list.__iter__(sl) if isinstance(e, FP._Pk) and e.segs for sg in e.segs for nb in [billed.get(id(sg))] if nb)
    print("\nkv_bytes off %d on %d, live sealed %d, overlap %d" % (off, on, sealed, overlap))
    assert off - on == overlap > 0


def test_status_carries_kv_roles_for_the_largest_live_cache():
    from kyojin_mlx.serve.engine import Engine
    lm, _ = M.make(sliding_window=16)
    cache = FP.wrap_cache(lm.make_cache())
    prefill(lm, M.tokens(256), cache)
    sess = types.SimpleNamespace(cache=cache, kv_bytes=Backend().nbytes(cache), pos=256)
    eng = types.SimpleNamespace(prefilling=[sess], ready=[], members=[], waiting=[])
    st = Engine._kv_roles(eng)
    assert st["kv_roles"]["handoff"] > 0 and st["kv_roles"]["counted"] == sess.kv_bytes
    assert set(st["kv_roles"]["per_token"]) >= {"packed", "state", "handoff", "counted"}

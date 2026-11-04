"""TurboQuant KV through the disk KV tier: a TQKVCache saved at a block boundary and restored by a fresh KvDisk (= a restarted server) attends BIT FOR
BIT like the cache that was never saved (packed pool words, fp16 scales and the fp16 sink / ring overlay all persisted), solo and forked.
Synthetic tests need Metal only; the model test needs a Qwen3.5 EXL3 pack: KYOJIN_Q35=<dir> (e.g. models/q35-2b-exl3-2.10)."""
import hashlib
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import ArraysCache  # noqa: E402

from kyojin_mlx import kvdisk as K, kvwin as KW, multiuser as MU, tq as TQ  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
HQ, HK, D, BLOCK = 8, 2, 64, 32


def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)


def make():
    return [TQ.TQKVCache(6, 5, 4, 16), ArraysCache(1)]


def feed(cache, q, k, v, a, b):
    c = cache[0]
    kv = c.update_and_fetch(k[:, :, a:b], v[:, :, a:b])
    out = KW._base.scaled_dot_product_attention(q[:, :, a:b], kv[0], kv[1], cache=c, scale=D ** -0.5, mask=None)
    cache[1][0] = (out.mean(axis=(1, 2)) if cache[1][0] is None else cache[1][0] + out.mean(axis=(1, 2)))
    mx.eval(out, cache[1][0])
    return out


def data(L):
    r = np.random.default_rng(1)
    f = lambda *s: mx.array((r.standard_normal(s) * (1 + r.random(s) * 2)).astype(np.float16))
    return f(1, HQ, L, D), f(1, HK, L, D), f(1, HK, L, D)


def md5(cache):
    h = hashlib.md5()
    for x in cache.state:
        if x is not None:
            h.update(np.array(x.astype(mx.float32) if x.dtype == mx.float16 else x).tobytes())
    return h.hexdigest()


def test_restore_is_bitwise(tmp_path):
    KW.install()
    L, ids = 100, list(range(1000, 1100))
    q, k, v = data(L + 40)
    ref = make()
    disk = K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK)
    for a in range(0, 96, 16):
        feed(ref, q, k, v, a, a + 16)
        if (a + 16) % BLOCK == 0:
            mx.eval(*K.KvDisk._arrays(ref))
            assert disk.save(ids, a + 16, ref) > 0
    b, got, _ = K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK).lookup(ids, below=100)
    assert b == 96 and type(got[0]).__name__ == "TQKVCache"
    assert md5(got[0]) == md5(ref[0])                                   # pool words + scales + overlay: identical bytes
    for a, c in ((96, 4), (100, 1), (101, 1), (102, 7), (109, 1)):
        o1, o2 = feed(ref, q, k, v, a, a + c), feed(got, q, k, v, a, a + c)
        assert np.array_equal(bits(o1), bits(o2)), a
    assert md5(got[0]) == md5(ref[0])


def test_fork_after_restore_is_independent(tmp_path):
    KW.install()
    ids = list(range(64))
    q, k, v = data(80)
    c = make()
    disk = K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK)
    for a in (0, 32):
        feed(c, q, k, v, a, a + 32)
        mx.eval(*K.KvDisk._arrays(c))
        disk.save(ids, a + 32, c)
    _, r, _ = K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK).lookup(ids + [0], below=65)
    f = MU.fork_cache(r)
    o_fork, o_r, o_c = feed(f, q, k, v, 64, 70), feed(r, q, k, v, 64, 70), feed(c, q, k, v, 64, 70)
    assert np.array_equal(bits(o_fork), bits(o_c)) and np.array_equal(bits(o_r), bits(o_c))


Q35 = os.environ.get("KYOJIN_Q35")


@pytest.mark.skipif(not Q35, reason="set KYOJIN_Q35")
def test_q35_restore_same_logits_and_tokens(tmp_path, monkeypatch):
    """Real Q35Backend + Engine with KYOJIN_TQ on: restored cache gives the same next-token logits to the bit; a restarted engine answers from the disk tier
    with the same greedy tokens as an engine with no disk."""
    from kyojin_mlx.serve.backends import load_q35
    from kyojin_mlx.serve.engine import Engine
    monkeypatch.setenv(TQ.ENV, "6:6:4:32")
    be, _ = load_q35(Q35)
    ns = be.disk_namespace()
    assert ns["tq"] == dict(kbits=6, vbits=6, sink=4, window=32)
    assert any(type(c).__name__ == "TQKVCache" for c in be.make_cache())
    ids = [int(x) for x in np.random.default_rng(3).integers(100, 5000, 200)]
    tail = ids + [11, 12, 13, 14, 15]
    disk = K.KvDisk(str(tmp_path / "a"), ns, be.make_cache, block=64)
    cache = be.make_cache()
    for a in range(0, 128, 32):
        be.prefill_chunk(cache, ids[a:a + 32])
        if (a + 32) % 64 == 0:
            assert disk.save(ids, a + 32, cache) > 0
    b, got, _ = K.KvDisk(str(tmp_path / "a"), ns, be.make_cache, block=64).lookup(tail, below=len(tail))
    assert b == 128
    assert [md5(c) for c in cache if type(c).__name__ == "TQKVCache"] == [md5(c) for c in got if type(c).__name__ == "TQKVCache"]
    lg = {}
    for name, c in (("ref", cache), ("disk", got)):
        for a in range(128, 200, 32):
            lg[name] = be.prefill_chunk(c, tail[a:min(a + 32, len(tail))])
        bc = be.batch([c])
        steps = [be.step(bc, mx.array([7 + i], dtype=mx.int32)) for i in range(6)]
        mx.eval(*steps)
        lg[name + "s"] = steps
    assert np.array_equal(np.array(lg["ref"]).view(np.uint32), np.array(lg["disk"]).view(np.uint32))
    for x, y in zip(lg["refs"], lg["disks"]):
        assert np.array_equal(np.array(x).view(np.uint32), np.array(y).view(np.uint32))

    def eng(d):
        return Engine(be, sessions=2, ctx=4096, chunk=32, block=32, prefix_bytes=1 << 28, disk=d)

    def run(e, prompt):
        e.start()
        try:
            h = e.submit(prompt, max_new=10)
            toks = []
            while True:
                ev = h.events.get(timeout=300)
                if ev[0] != "tok":
                    return toks, e
                toks.append(ev[1])
        finally:
            e.stop()

    ref, _ = run(eng(None), tail)
    first, ea = run(eng(K.KvDisk(str(tmp_path / "b"), ns, be.make_cache, block=64)), ids)
    assert ea.prefix.disk_saved == 3
    out, eb = run(eng(K.KvDisk(str(tmp_path / "b"), ns, be.make_cache, block=64)), tail)
    assert out == ref and eb.status()["disk_hits"] == 1 and eb.stats["prefix_hit_tokens"] == 192


def test_other_pool_layout_is_refused_cleanly(tmp_path):
    """A checkpoint written for another pool layout is not loaded (lookup falls back to no hit), the file stays usable by the matching cache class."""
    KW.install()
    ids = list(range(1000, 1100))
    q, k, v = data(100)
    ref = make()
    disk = K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK)
    for a in range(0, 96, 16):
        feed(ref, q, k, v, a, a + 16)
        if (a + 16) % BLOCK == 0:
            mx.eval(*K.KvDisk._arrays(ref))
            assert disk.save(ids, a + 16, ref) > 0

    class Future(TQ.TQKVCache):
        CKPT_LAYOUT = 7

    b, got, _ = K.KvDisk(str(tmp_path), {"t": "tq"}, lambda: [Future(6, 5, 4, 16), ArraysCache(1)], block=BLOCK).lookup(ids, below=100)
    assert b == 0 and got is None
    assert K.KvDisk(str(tmp_path), {"t": "tq"}, make, block=BLOCK).lookup(ids, below=100)[0] == 96

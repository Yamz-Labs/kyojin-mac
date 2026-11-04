"""KYOJIN_KV_DIGEST_INC: the kv-disk checkpoints take their digests from the session's running hash. Same files as the one-shot hash, mode 2 finds no mismatch, a session
restored from a snapshot above the parent point falls back to the one-shot hash for that checkpoint, and `save` accepts the digests (fake backend, mlx-lm KVCache)."""
import os

import numpy as np
import pytest

pytest.importorskip("mlx_lm.models.cache")
from kyojin_mlx import kvdisk as K  # noqa: E402
from test_serve_kvdisk import FakeKV, disk, engine, run, PROMPT  # noqa: E402

LONG = [(i * 11 + 5) % 90 + 1 for i in range(90)]


def names(d):
    """Sorted (length, digest) keys of the checkpoints a KvDisk holds."""
    return sorted(d.index)


@pytest.mark.parametrize("mode", ["1", "2"])
def test_same_checkpoint_files_as_the_one_shot_hash(tmp_path, monkeypatch, mode):
    ref, inc = tmp_path / "ref", tmp_path / "inc"
    ref.mkdir(), inc.mkdir()
    monkeypatch.delenv("KYOJIN_KV_DIGEST_INC", raising=False)
    dr, di = disk(ref, block=8), disk(inc, block=8)
    run(engine(FakeKV(), dr), LONG, max_new=2)
    monkeypatch.setenv("KYOJIN_KV_DIGEST_INC", mode)
    e = engine(FakeKV(), di)
    run(e, LONG, max_new=2)
    a, b = names(dr), names(di)
    assert a and a == b                                                    # same keys (length-digest names)
    assert e.stats["kv_digest_mismatch"] == 0
    if mode == "2":
        assert e.stats["kv_digest_check"] >= len(a)                        # every checkpoint was checked against the one-shot hash


def test_save_with_given_digests_equals_save_without(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    kvs = []
    for d, kw in ((a, {}), (b, {"dg": K.digest(LONG, 16), "pdg": K.digest(LONG, 8)})):
        kv = disk(d, block=8)
        kvs.append(kv)
        c = FakeKV().make_cache()
        FakeKV.push(c, LONG[:8])
        assert kv.save(LONG, 8, c) > 0
        FakeKV.push(c, LONG[8:16])
        assert kv.save(LONG, 16, c, **kw) > 0
    assert names(kvs[0]) == names(kvs[1]) and len(names(kvs[0])) == 2


def test_restored_session_without_the_parent_point_falls_back(tmp_path, monkeypatch):
    """A second engine restores the snapshot at 16 (a grid point it never passed with its own hash is recorded by `_begin_prefill`); a hit at a non-grid length leaves no
    parent digest for the next checkpoint and the one-shot hash is used, still the right files and no mismatch."""
    monkeypatch.setenv("KYOJIN_KV_DIGEST_INC", "2")
    d = disk(tmp_path, block=8)
    run(engine(FakeKV(), d), LONG[:21], max_new=1)
    n_before = len(d.index)
    e = engine(FakeKV(), disk(tmp_path, block=8))
    run(e, LONG, max_new=1)
    assert e.stats["kv_digest_mismatch"] == 0
    assert n_before >= 2 and len(e.prefix.disk.index) > n_before

def test_mode_off_passes_no_digests(tmp_path, monkeypatch):
    monkeypatch.delenv("KYOJIN_KV_DIGEST_INC", raising=False)
    seen = []
    d = disk(tmp_path, block=8)
    orig = d.save
    d.save = lambda *a, **k: (seen.append(sorted(k)), orig(*a, **k))[1]
    run(engine(FakeKV(), d), LONG[:33], max_new=1)
    assert seen and all("dg" not in k and "pdg" not in k for k in seen)

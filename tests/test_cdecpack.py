"""KYOJIN_CDECPACK: gather the selected top set from the packed pool, no mirror. CPU only (the attention kernels are Metal); the reference is the whole-layer unpack."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import cdecpack as CD, depthflat as DF, fp4pack, fp4pool, flags as FL
from kyojin_mlx.fp4pool import PoolSlots


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)               # every layer comes back as a PackedView in lazy mode
    for k in DF.HITS:
        DF.HITS[k] = 0
    for k in CD.HITS:
        CD.HITS[k] = 0
    yield
    mx.set_default_device(dev)


def layer(n, D=512, seed=0):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32) * 3).astype(mx.float16)
    s = PoolSlots([None])
    s.lazy = True                                               # as around the decode attention core: the append hands back a view, no mirror
    s.append_rows(0, fp4pack.unpack(*fp4pack.pack(x, 16, True), 16, True, mx.float16))
    return s, s[0]


def test_compact_equals_dense_rows_bit_for_bit_with_invalid_entries():
    s, v = layer(700)
    assert isinstance(v, fp4pool.PackedView)
    dense = fp4pack.unpack(v.codes[:, :v.n], v.scales[:, :v.n], 16, True, mx.float16)
    idx = mx.array([[3, 699, 0, -1, 250, 700, 9999, 3], [5, 6, 7, 8, -1, -1, 120, 640]], dtype=mx.int32)
    rows, remap = CD.compact(v, idx)
    assert rows.shape == (1, 16, 512) and rows.dtype == mx.float16 and remap.shape == idx.shape
    r, i, d = np.array(rows[0]), np.array(idx), np.array(dense[0])
    m = np.array(remap)
    for a in range(2):
        for b in range(8):
            if 0 <= i[a, b] < 700:
                assert m[a, b] >= 0 and np.array_equal(r[m[a, b]], d[i[a, b]])
            else:
                assert m[a, b] == -1


def test_attention_reads_the_same_values_through_the_remap():
    """What the kernels do with an index: row address + the 0 <= idx < N test. A softmax-weighted read over the compact pool equals the dense one."""
    s, v = layer(512, seed=2)
    dense = np.array(fp4pack.unpack(v.codes[:, :v.n], v.scales[:, :v.n], 16, True, mx.float16)[0]).astype(np.float32)
    idx = np.array([[10, 20, 30, 500, -1, 7], [1, 2, 3, 4, 5, 6]], dtype=np.int32)
    q = np.random.default_rng(3).standard_normal((2, 512)).astype(np.float32)

    def read(pool, ix):
        out = []
        for u in range(2):
            ok = (ix[u] >= 0) & (ix[u] < pool.shape[0])
            sc = np.where(ok, pool[np.where(ok, ix[u], 0)] @ q[u], -np.inf)
            w = np.exp(sc - sc.max()); w /= w.sum()
            out.append(w @ pool[np.where(ok, ix[u], 0)])
        return np.stack(out)
    rows, remap = CD.compact(v, mx.array(idx))
    assert np.array_equal(read(dense, idx), read(np.array(rows[0]).astype(np.float32), np.array(remap)))


def test_no_mirror_is_built_and_only_the_selected_rows_are_unpacked(monkeypatch):
    s, v = layer(4000)
    unpacked = []
    orig = fp4pack.unpack
    monkeypatch.setattr(fp4pack, "unpack", lambda c, sc, *a, **k: (unpacked.append(c.shape[-2] if c.ndim > 1 else 1), orig(c, sc, *a, **k))[1])
    idx = mx.array(np.random.default_rng(1).integers(0, 4000, (6, 64)), dtype=mx.int32)
    CD.compact(v, idx)
    assert unpacked == [6 * 64]                                   # K x n rows, not the 4000 of the layer
    assert s.raw(0).mirror is None and DF.HITS["build"] == 0
    v.dense()                                                      # the old reader: builds the mirror, kept for the session
    assert s.raw(0).mirror is not None and DF.HITS["build"] == 1


def test_bytes_per_row_packed_against_mirror():
    s, v = layer(1024)
    e = s.raw(0)
    packed = e.codes[:, :1024].nbytes + e.scales[:, :1024].nbytes
    assert packed // 1024 == 288                                   # D=512: 256 B codes + 32 B scales per row
    v.dense()
    assert e.mirror[:, :1024].nbytes // 1024 >= 1024               # the mirror a dense() read keeps: 1 KiB per row (+ slack)


def test_flag_is_registered_default_off_and_switchable():
    f = FL.FLAGS["cdecpack"]
    assert f.env == "KYOJIN_CDECPACK" and f.default == "0" and f.runtime
    old = CD.ENABLED
    try:
        FL.set("cdecpack", "1")
        assert CD.ENABLED is True
        FL.set("cdecpack", "0")
        assert CD.ENABLED is False
    finally:
        CD.ENABLED = old


def test_levers_row_exists():
    import pathlib
    t = (pathlib.Path(__file__).parent.parent / "docs" / "cloud" / "LEVERS.md").read_text()
    assert "`KYOJIN_CDECPACK`" in t

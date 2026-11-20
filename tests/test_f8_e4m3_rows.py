"""GGUF type id 43 (F8_E4M3): one e4m3fn byte per element, no block scale, decoded by row through the shared reader (foreign_gguf.dequant, MappedRows).
CPU only. The DS4.1 test reads the rows of the real per_layer_token_embd.weight held in the range-read header files (skipped when they are absent)."""
import hashlib
import os

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402

DS41 = os.path.expanduser("/path/to/data")
DS41_F8 = "per_layer_token_embd.weight"


def test_type_43_is_a_row_type():
    assert GF.TYPES[43] == ("F8_E4M3", 1, 1) and 43 not in GF.UPSTREAM_ONLY
    assert "F8_E4M3" in GF.DEQUANT and GF.NAME_TO_ID["F8_E4M3"] == 43
    assert "F8_E4M3" not in GF.KERNEL_TYPES + GF.KERNEL_TYPESK + GF.KERNEL_TYPES5 + GF.KERNEL_TYPES8


def _same(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def test_all_256_codes():
    codes = np.arange(256, dtype=np.uint8)
    t = GF.dequant("F8_E4M3", codes[None], 256)[0]
    assert t.dtype == np.float32
    assert np.isnan(t[[0x7F, 0xFF]]).all() and np.isnan(t).sum() == 2 and not np.isinf(t).any()
    assert (t[0x38], t[0x7E], t[0xFE], t[0x01], t[0x08], t[0x00]) == (1.0, 448.0, -448.0, 2.0 ** -9, 2.0 ** -6, 0.0) and np.signbit(t[0x80])
    assert _same(t, np.array(mx.from_fp8(mx.array(codes), dtype=mx.float32)))        # MLX's own e4m3fn decoder


def test_all_256_codes_ml_dtypes():
    mld = pytest.importorskip("ml_dtypes")                                              # not in the lane venv: the MLX check above is the reference there
    codes = np.arange(256, dtype=np.uint8)
    t = GF.dequant("F8_E4M3", codes[None], 256)[0]
    assert _same(t, codes.view(mld.float8_e4m3fn).astype(np.float32))


def test_synthetic_gguf_rows(tmp_path):
    from kyojin_mlx.foreign_mapped_rows import MappedRows
    rows, k = 50, 96
    rng = np.random.default_rng(7)
    b = rng.integers(0, 256, (rows, k)).astype(np.uint8)
    b[(b & 127) == 127] = 0x38
    p = str(tmp_path / "f8.gguf")
    SY.write_gguf(p, {"general.architecture": "synth"}, [("tab.weight", (k, rows), "F8_E4M3", b)])
    g = GF.GgufFile(p)
    ti = g.tensors["tab.weight"]
    assert (ti.type, ti.tname, ti.shape, ti.nbytes) == (43, "F8_E4M3", (rows, k), rows * k)
    raw = g.raw("tab.weight")
    assert raw.shape == (rows, k) and np.array_equal(raw, b)                       # one byte per element: row i is bytes [i k, (i + 1) k)
    assert np.array_equal(GF.dequant("F8_E4M3", raw, k), GF.F8_E4M3_TAB[b])
    with pytest.raises(GF.GgufTypeError, match="row only"):
        g.tensor_f32("tab.weight")                                                  # no whole-tensor decode (ROW_ONLY)
    m = MappedRows(raw, "F8_E4M3", k)
    ids = np.array([3, 0, 49, 3], np.int64)
    y = np.array(m(mx.array(ids)).astype(mx.float32))
    assert y.shape == (4, k) and np.array_equal(y, GF.F8_E4M3_TAB[b[ids]].astype(np.float16).astype(np.float32))


def _qwen4exp_file(tmp_path, extra=()):
    rows, k = 64, 160
    b = np.random.default_rng(5).integers(0, 127, (rows, k)).astype(np.uint8)
    p = str(tmp_path / "q.gguf")
    SY.write_gguf(p, {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}, list(extra) + [("per_layer_token_embd.weight", (k, rows), "F8_E4M3", b)])
    return p, b


def test_mapped_table_path_takes_f8_and_a_dense_f8_tensor_fails_loudly(tmp_path, monkeypatch):
    from kyojin_mlx import foreign_linear as FL
    from kyojin_mlx import foreign_qwen4exp_gguf as Q
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    monkeypatch.setenv("KYOJIN_MAPPED_ROW_TABLE", "1")
    Q.register()
    try:
        p, b = _qwen4exp_file(tmp_path)
        src = FL._read_gguf(p, "qwen4exp")
        raw, tn, width, _ = src.mapped["per_layer_token_embd.weight"]
        assert tn == "F8_E4M3" and width == 160 and np.array_equal(raw, b) and not src.dense and not src.packs
        w = np.random.default_rng(6).integers(0, 127, (32, 160)).astype(np.uint8)
        p, _ = _qwen4exp_file(tmp_path, [("token_embd.weight", (160, 32), "F8_E4M3", w)])
        with pytest.raises(GF.GgufTypeError, match="row only"):
            FL._read_gguf(p, "qwen4exp")
    finally:
        monkeypatch.undo()
        Q.register()


def test_b32_layout_is_read_from_the_span(tmp_path):
    """id 43 written with 33-byte blocks (e8m0 scale + 32 e4m3) is F8_E4M3_B32; the same id with one byte per element stays F8_E4M3."""
    rows, k = 40, 64
    rng = np.random.default_rng(11)
    v = rng.integers(0, 256, (rows, k)).astype(np.uint8)
    v[(v & 127) == 127] = 0x38
    sc = rng.integers(117, 131, (rows, k // 32)).astype(np.uint8)
    blk = np.concatenate([sc[:, :, None], v.reshape(rows, k // 32, 32)], -1).reshape(rows, -1)
    p = str(tmp_path / "b32.gguf")
    SY.write_gguf(p, {"general.architecture": "synth"}, [("tab.weight", (k, rows), "F8_E4M3", blk), ("plain.weight", (k, rows), "F8_E4M3", v)])
    g = GF.GgufFile(p)
    t, q = g.tensors["tab.weight"], g.tensors["plain.weight"]
    assert (t.type, t.tname, t.nbytes, q.type, q.tname, q.nbytes) == (43, "F8_E4M3_B32", rows * k // 32 * 33, 43, "F8_E4M3", rows * k)
    assert g.raw("tab.weight").shape == (rows, k // 32 * 33) and np.array_equal(g.raw("plain.weight"), v)
    want = (GF.F8_E4M3_TAB[v].reshape(rows, -1, 32) * np.exp2(sc.astype(np.float32) - 127)[:, :, None]).reshape(rows, k)
    assert np.array_equal(GF.dequant("F8_E4M3_B32", g.raw("tab.weight"), k), want, equal_nan=True)
    with pytest.raises(GF.GgufTypeError, match="row only"):
        g.tensor_f32("tab.weight")


@pytest.mark.skipif(not os.path.exists(DS41), reason="DS4.1 UD-Q4_K_XL header files not present")
def test_ds41_rows_map_and_read_back():
    g = GF.GgufFile(DS41)                      # the 9 shards; the table lives in shard 8 (192 + 264 x 768022850 bytes), whose first 24 MiB of data are real bytes
    f8 = [n for n, t in g.tensors.items() if t.type == 43]
    assert f8 == [DS41_F8]
    ti = g.tensors[DS41_F8]
    assert (ti.tname, ti.ne, ti.nbytes) == ("F8_E4M3_B32", (256, 768022850), 264 * 768022850)
    src = g._owner.get(DS41_F8, g)
    assert src.data_start + ti.offset + ti.nbytes == os.path.getsize(src.path)
    raw = g.raw(DS41_F8)
    assert raw.shape == (768022850, 264)
    ids = [0, 1, 2, 90000]
    x = GF.dequant(ti.tname, raw[ids], 256)
    direct = np.stack([np.fromfile(src.path, np.uint8, count=264, offset=src.data_start + ti.offset + i * 264) for i in ids]).reshape(4, 8, 33)
    assert x.shape == (4, 256) and np.isfinite(x).all() and (direct[..., 0] >= 110).all() and (direct[..., 0] <= 135).all()      # e8m0 scales near 2^-7
    want = (np.array(mx.from_fp8(mx.array(direct[..., 1:].reshape(4, 256)), dtype=mx.float32)).reshape(4, 8, 32) * np.exp2(direct[..., :1].astype(np.float32) - 127)).reshape(4, 256)
    assert np.array_equal(x, want)

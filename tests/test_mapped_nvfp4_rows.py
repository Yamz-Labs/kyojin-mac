# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 n-gram table of the Qwen3.8-Flash-Next FP4-Local pack read by row id: rows equal the full dequant. CPU only, synthetic tables."""
import numpy as np
import pytest
import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_pack as FP
from kyojin_mlx import foreign_qwen4exp_st as Q
from kyojin_mlx.foreign_mapped_rows import MappedNvfp4Rows



from kyojin_mlx.foreign_mapped_rows import MappedNvfp4Rows

LM = "model.language_model."
PLE = LM + "layers.1.ple.ple_embedding.ngram_embedding"


def _info(rows=(5, 7, 4), k=32):
    i = {}
    for s, r in enumerate(rows):
        i[f"{PLE}.shard_{s}.weight"] = ("U8", (r, k // 2))
        i[f"{PLE}.shard_{s}.weight_scale"] = ("F8_E4M3", (r, k // 16))
    i[f"{PLE}.weight_scale_2"] = ("F32", (1,))
    return i


def _table(rows=(5, 7, 4), k=160, seed=0):
    rng = np.random.default_rng(seed)
    qs = [rng.integers(0, 256, (r, k // 2), dtype=np.uint8) for r in rows]
    sc = [rng.integers(0, 0x78, (r, k // 16), dtype=np.uint8) for r in rows]          # positive E4M3 bytes below the NaN code
    sc[1][2, 0] = 0x7F                                                                 # the NaN code scales by 0, as in the kernel
    return qs, sc


def _full(qs, sc, g):
    q, s = np.concatenate(qs), np.concatenate(sc)
    p = FP.Pack("nvfp4", q.shape[0], q.shape[1] * 2, q, np.ascontiguousarray(s).view("<u2").reshape(q.shape[0], -1), np.float32(g))
    return FP.dequant(p)


@pytest.mark.parametrize("k", [32, 160])
def test_rows_equal_full_dequant(k):
    qs, sc = _table(k=k)
    m = MappedNvfp4Rows(list(zip(qs, sc)), 0.0123, k)
    full = _full(qs, sc, 0.0123).astype(np.float16)
    ids = np.array([[[0, 4, 5, 11], [12, 15, 5, 5]]])                      # shard edges (4|5, 11|12), the last row, repeats
    out = m(mx.array(ids))
    assert out.shape == (1, 2, 4, k) and out.dtype == mx.float16
    assert np.array_equal(np.asarray(out), full[ids])
    assert m.stats["rows"] == 8 and m.stats["distinct"] == 6 and m.stats["bytes"] == 6 * (k // 2 + k // 16)     # only the distinct rows were read
    assert m.num_embeddings == 16
    with pytest.raises(IndexError):
        m(mx.array([16]))
    with pytest.raises(IndexError):
        m(mx.array([-1]))


def test_rows_decode_float32_exact():
    qs, sc = _table()
    m = MappedNvfp4Rows(list(zip(qs, sc)), 0.5, 160)
    assert np.array_equal(m.decode(np.arange(16)), _full(qs, sc, 0.5))


def test_bad_tables_refused():
    qs, sc = _table()
    with pytest.raises(ValueError):
        MappedNvfp4Rows([(qs[0], sc[0][:, :5])], 1.0)                        # scales not one per 16
    with pytest.raises(ValueError):
        MappedNvfp4Rows(list(zip(qs, sc)), 0.0)
    with pytest.raises(ValueError):
        MappedNvfp4Rows([], 1.0)


def test_build_tables_from_a_checkpoint_like_object():
    qs, sc = _table(k=32)

    class CK:
        def raw(self, n):
            i = int(n.split(".shard_")[1].split(".")[0])
            return ("U8", qs[i].shape, qs[i]) if n.endswith(".weight") else ("F8_E4M3", sc[i].shape, sc[i])

        def f32(self, n):
            return np.array([0.25], np.float32)
    m = Q.build_tables(CK(), Q.ple_tables(_info()))[PLE]
    assert np.array_equal(np.asarray(m(mx.array([3, 9]))), _full(qs, sc, 0.25).astype(np.float16)[[3, 9]])



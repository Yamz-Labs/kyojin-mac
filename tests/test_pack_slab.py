"""foreign_gguf.to_pack repacks a big tensor in row slabs (host temporaries bounded): every field equals the whole-tensor repack, for every kernel type."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

TYPES = GF.KERNEL_TYPES + GF.KERNEL_TYPES8 + GF.KERNEL_TYPESK


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("slab_rows", (1, 7, 64))
def test_slab_pack_equals_whole(t, slab_rows, monkeypatch):
    k = 512
    raw = SY.random_blocks(t, 70, k, np.random.default_rng(11))
    whole = GF._to_pack_whole(t, raw, k)
    monkeypatch.setattr(GF, "PACK_SLAB_ELEMS", slab_rows * k)
    slab = GF.to_pack(t, raw, k)
    assert (slab.fmt, slab.n, slab.k) == (whole.fmt, whole.n, whole.k)
    assert np.array_equal(slab.qs, whole.qs) and slab.qs.dtype == whole.qs.dtype
    assert (slab.meta is None) == (whole.meta is None) and (whole.meta is None or (np.array_equal(slab.meta, whole.meta) and slab.meta.dtype == whole.meta.dtype))
    assert slab.extra.keys() == whole.extra.keys() and all(np.array_equal(slab.extra[x], whole.extra[x]) and slab.extra[x].dtype == whole.extra[x].dtype for x in whole.extra)
    assert np.all(np.asarray(slab.gscale) == np.asarray(whole.gscale))

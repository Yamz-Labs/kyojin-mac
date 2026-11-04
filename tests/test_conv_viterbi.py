"""EXL3 conversion on Metal: the Viterbi quantiser and the trellis packer against fixtures made by the CUDA kernels on the 3090.
oracle_tiles.npz: in{K} float32 [48, 256] tiles, i{K} int16 idx and w{K} float32 values from exllamav3 ext.quantize_tiles (mul1, K = 1..4)."""
from pathlib import Path

import numpy as np
import pytest

from kyojin_mlx import conv_viterbi as V, ref_numpy as R

FIX = Path(__file__).resolve().parent.parent / "results_mac" / "conv" / "oracle_tiles.npz"


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_viterbi_bit_exact_vs_cuda(K):
    o = np.load(FIX)
    w, i = V.quantize_tiles(o[f"in{K}"], K)
    assert np.array_equal(np.array(i), o[f"i{K}"].astype(np.uint16))
    assert np.array_equal(np.array(w), o[f"w{K}"])


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_pack_roundtrip(K):
    import importlib.util, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    conv = importlib.import_module("conv_mlx")
    idx = np.load(FIX)[f"i{K}"].astype(np.uint16)
    tr = conv.pack_trellis(idx.reshape(3, 16, 256), K)
    assert tr.dtype == np.int16 and tr.shape == (3, 16, 16 * K)
    assert np.array_equal(R.decode_states(R.trellis_words(tr), K), idx)


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_register_kernel_equals_reference_kernel(K):
    """the register-resident kernel (impl 2) against the device-memory kernel that is bit-exact with CUDA, on random, tiny, huge and flat tiles"""
    rng = np.random.default_rng(K)
    x = rng.standard_normal((24, 256)).astype(np.float32)
    x[1] *= 1e-3; x[2] *= 40; x[3] = 0; x[4] = 0.7; x[5, ::2] = -3
    old = dict(V.CFG)
    try:
        V.CFG["impl"] = 1
        w1, i1 = V.quantize_tiles(x, K)
        V.CFG["impl"] = 2
        w2, i2 = V.quantize_tiles(x, K)
    finally:
        V.CFG.update(old)
    assert np.array_equal(np.array(i1), np.array(i2))
    assert np.array_equal(np.array(w1), np.array(w2))

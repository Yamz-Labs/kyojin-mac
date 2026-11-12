"""Block-scaled FP8 (weight_scale_inv grid) read as stored: the loader's fp8b pack against the vendor reader's float32 dequant (CPU)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_linear as FL, foreign_pack as FP, vendor_input as VI


class _Ck:
    """The two calls _fp8b_pack makes on a checkpoint: raw(weight) and scale_f32(scale)."""
    def __init__(self, q, g):
        self.q, self.g = q, g

    def raw(self, name):
        return ("F8_E4M3", list(self.q.shape), self.q.view(np.uint8)) if name == "w" else ("F32", list(self.g.shape), self.g)

    def scale_f32(self, name):
        return self.g


def _make(n, k, bn, bk, lo, hi, seed=0):
    r = np.random.default_rng(seed)
    q = r.integers(0, 256, (n, k), dtype=np.uint8)
    q[(q & 127) == 127] = 0x38
    g = r.uniform(lo, hi, (-(-n // bn), -(-k // bk))).astype(np.float32)
    ref = FP.E4M3_TAB[q] * g[np.arange(n) // bn][:, np.arange(k) // bk]
    return q, g, ref


@pytest.mark.parametrize("n,k,bn,bk,lo,hi", [(256, 512, 128, 128, 1e-4, 3e-3), (48, 5120, 128, 128, 1e-6, 1e-5), (200, 384, 128, 128, 0.5, 4.0),
                                             (64, 256, 64, 32, 1e-3, 1e-2)])
def test_fp8b_pack_is_the_block_dequant(n, k, bn, bk, lo, hi):
    FL._FP8B_CACHE.clear()
    q, g, ref = _make(n, k, bn, bk, lo, hi)
    p = FL._fp8b_pack(_Ck(q, g), {"weight": "w", "scale": "s"}, (bn, bk))
    assert p is not None and p.fmt == "fp8b" and p.meta.shape == (n, k // 32) and p.meta.dtype == np.float16
    m = p.meta.astype(np.float32)
    assert m.max() >= 64 and m.max() < 128 and m.min() > 2.0 ** -14            # no fp16 subnormal, x256 finite
    assert float(np.log2(p.gscale)).is_integer()                                  # exact power of two
    w = FP.dequant(p)
    assert np.allclose(w, ref, rtol=1e-3, atol=0) and np.abs(w - ref).max() <= 2.0 ** -11 * np.abs(ref).max()


def test_fp8b_refuses_blocks_that_straddle_units():
    FL._FP8B_CACHE.clear()
    q, g, _ = _make(64, 96, 64, 48, 0.1, 1.0)
    assert FL._fp8b_pack(_Ck(q, g), {"weight": "w", "scale": "s"}, (64, 48)) is None


def test_fp8b_infers_the_block_from_the_grid():
    FL._FP8B_CACHE.clear()
    q, g, ref = _make(256, 512, 128, 128, 0.01, 0.1)
    p = FL._fp8b_pack(_Ck(q, g), {"weight": "w", "scale": "s"}, None)
    assert p is not None and np.allclose(FP.dequant(p), ref, rtol=1e-3)

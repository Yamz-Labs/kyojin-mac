"""cpfmoe: pf_moe_nosync passes of a multi-pass call submitted with async_eval (host encodes pass k + 1 while the GPU runs pass k).

Wiring runs anywhere mlx imports. The exactness test needs Metal (Mac): flag on equals flag off bit for bit, several passes (chunk 64, rows 150 / 200), both codebooks.
Nothing here is timed."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import flags as FL, moe_prefill as MP  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")


def test_flag_wiring_default_off():
    f = FL.FLAGS["cpfmoe"]
    assert f.default == "0" and f.env == "KYOJIN_CPFMOE" and MP.CPFMOE is False
    FL.set_many({"cpfmoe": "1"})
    try:
        assert MP.CPFMOE is True
    finally:
        FL.set_many({"cpfmoe": "0"})
    assert MP.CPFMOE is False


@needs_metal
@pytest.mark.parametrize("cb", ["mul1", "mcg"])
def test_cpfmoe_bit_identical(cb, monkeypatch):
    from test_moe import make_stack, E, TOPK, KS
    from kyojin_mlx import moe as M
    g, _ = make_stack(256, 384, KS, cb, 1)
    u, _ = make_stack(256, 384, KS, cb, 2)
    d, _ = make_stack(384, 256, KS, cb, 3)
    moe = M.MoeGather(g, u, d, TOPK)
    for rows in (150, 200):
        rng = np.random.default_rng(rows)
        ids = mx.array(np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32))
        w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
        wts = mx.array(w / w.sum(-1, keepdims=True) * 1.5)
        x = mx.array(rng.standard_normal((rows, 256)).astype(np.float16) * np.float16(2.0))
        monkeypatch.setattr(MP, "MOE_NOSYNC", True)
        monkeypatch.setattr(MP, "CPFMOE", False)
        ref = np.array(MP.MoePrefill(moe, chunk=64)(x, ids, wts))
        monkeypatch.setattr(MP, "CPFMOE", True)
        got = np.array(MP.MoePrefill(moe, chunk=64)(x, ids, wts))
        assert np.array_equal(ref.view(np.uint32), got.view(np.uint32)), (cb, rows)

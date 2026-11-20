# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_ATTN_FUSE: the fused scores -> probabilities chain of MlaAttention.attend equals the stock chain (CPU, no Metal): global layer (kpool keep mask, "causal"),
windowed layer (array mask, slice flag off and on), query blocks on top. Not bit-equal by design (exp(sc - logaddexp(lse, sink)) vs exp(sc - max) / sum): rel diff <= 1e-5."""
import numpy as np
import pytest

import mlx.core as mx

from kyojin_mlx import foreign_skeleton_mla as MLA
from test_hy4_attn_slice import _chunks, _kp_model, _maxdiff, _run, _swa_model  # noqa: E402

mx.set_default_device(mx.cpu)
TOL = 1e-5


def test_global_layer_fused_equals_stock():
    rng = np.random.default_rng(21)
    m = _kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16, top_k=64)
    xs = _chunks(rng, 64, 1200, 3)
    n0 = MLA.MlaAttention.PATHS["attn_fuse"]
    ref = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "0")
    assert MLA.MlaAttention.PATHS["attn_fuse"] == n0, "default path must not fuse"
    new = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "1")
    assert MLA.MlaAttention.PATHS["attn_fuse"] > n0, "the fused path did not run"
    d = _maxdiff(ref, new)
    print(f"fuse global: max rel diff {d:.3g}")
    assert d <= TOL


@pytest.mark.parametrize("slice_flag", ["0", "1"])
def test_windowed_layer_fused_equals_stock(slice_flag, monkeypatch):
    monkeypatch.setenv("KYOJIN_HY4_SWA_SLICE", slice_flag)
    rng = np.random.default_rng(22)
    m = _swa_model(rng, 64, 4, 32, 24, 8, 16, 128)
    xs = _chunks(rng, 64, 1200, 3)
    n0 = MLA.MlaAttention.PATHS["attn_fuse"]
    ref = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "0", 128, True)
    new = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "1", 128, True)
    assert MLA.MlaAttention.PATHS["attn_fuse"] > n0, "the fused path did not run"
    d = _maxdiff(ref, new)
    print(f"fuse windowed slice={slice_flag}: max rel diff {d:.3g}")
    assert d <= TOL


def test_fused_with_query_blocks(monkeypatch):
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK", "256")
    rng = np.random.default_rng(23)
    m = _kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16, top_k=64)
    xs = [mx.array(rng.standard_normal((1, n, 64)).astype(np.float32)) for n in (700, 600)]
    ref = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "0")
    new = _run(m, xs, "KYOJIN_HY4_ATTN_FUSE", "1")
    assert _maxdiff(ref, new) <= TOL

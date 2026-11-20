# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_ATTN_QBLK: attention of a big prefill chunk in query blocks (each block over the keys up to its own last row) computes the same function as
the one-shot path (CPU, no Metal). Global layer (kpool selection that bites, "causal" mask) and windowed layer (array mask, slice flag off and on), chunks of
1200 / 1300 / 100 rows (ragged against the 256-row block), then decode rows."""
import numpy as np
import pytest

import mlx.core as mx

from kyojin_mlx import foreign_skeleton_mla as MLA
from test_hy4_attn_slice import TOL, _kp_model, _maxdiff, _run, _swa_model  # noqa: E402

mx.set_default_device(mx.cpu)


def _big_chunks(rng, d, sizes, dec):
    xs = [mx.array(rng.standard_normal((1, n, d)).astype(np.float32)) for n in sizes]
    return xs + [mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)) for _ in range(dec)]


def test_global_layer_blocks_equal_one_shot():
    rng = np.random.default_rng(11)
    m = _kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16, top_k=64)             # n_top = 15 pools of 4 keys: the selection drops most keys past 64 positions
    xs = _big_chunks(rng, 64, (1200, 1300, 100), 3)
    n0 = MLA.MlaAttention.PATHS["qblocks"]
    ref = _run(m, xs, "KYOJIN_HY4_ATTN_QBLK", "0")
    assert MLA.MlaAttention.PATHS["qblocks"] == n0, "default path must not block"
    new = _run(m, xs, "KYOJIN_HY4_ATTN_QBLK", "256")
    assert MLA.MlaAttention.PATHS["qblocks"] > n0, "the block path did not run"
    d = _maxdiff(ref, new)
    print(f"qblk global: max rel diff {d:.3g} over {len(xs)} calls")
    assert d <= TOL


@pytest.mark.parametrize("slice_flag", ["0", "1"])
def test_windowed_layer_blocks_equal_one_shot(slice_flag, monkeypatch):
    monkeypatch.setenv("KYOJIN_HY4_SWA_SLICE", slice_flag)
    rng = np.random.default_rng(12)
    m = _swa_model(rng, 64, 4, 32, 24, 8, 16, 128)
    xs = _big_chunks(rng, 64, (1200, 1300, 100), 3)
    n0 = MLA.MlaAttention.PATHS["qblocks"]
    ref = _run(m, xs, "KYOJIN_HY4_ATTN_QBLK", "0", 128, True)
    new = _run(m, xs, "KYOJIN_HY4_ATTN_QBLK", "256", 128, True)
    assert MLA.MlaAttention.PATHS["qblocks"] > n0, "the block path did not run"
    d = _maxdiff(ref, new)
    print(f"qblk windowed slice={slice_flag}: max rel diff {d:.3g} over {len(xs)} calls")
    assert d <= TOL


def test_windowed_layers_take_their_own_block_size(monkeypatch):
    """KYOJIN_HY4_ATTN_QBLK_SWA overrides the block size of windowed layers only; global layers keep KYOJIN_HY4_ATTN_QBLK."""
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK", "0")
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK_SWA", "128")
    rng = np.random.default_rng(13)
    mw = _swa_model(rng, 64, 4, 32, 24, 8, 16, 128)
    mg = _kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16, top_k=64)
    xs = _big_chunks(rng, 64, (700, 500), 2)
    n0 = MLA.MlaAttention.PATHS["qblocks"]
    _run(mg, xs, "KYOJIN_HY4_ATTN_QBLK", "0")
    assert MLA.MlaAttention.PATHS["qblocks"] == n0, "a global layer must not block with QBLK = 0"
    ref = _run(mw, xs, "KYOJIN_HY4_ATTN_QBLK", "0", 128, True)
    assert MLA.MlaAttention.PATHS["qblocks"] > n0, "a windowed layer must block with QBLK_SWA = 128"
    monkeypatch.setenv("KYOJIN_HY4_ATTN_QBLK_SWA", "0")
    one = _run(mw, xs, "KYOJIN_HY4_ATTN_QBLK", "0", 128, True)
    assert _maxdiff(ref, one) <= TOL

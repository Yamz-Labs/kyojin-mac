# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_PF_IDX_QMM: the prefill indexer score GEMM on the MXFP8 path.

`ds_pf_idx.scores_qmm` must give the same scores as the fused fp16 kernel (`fscore`): the index keys and queries of DS4.1 are
already fp4 / ue8m0 quantized (a code in {0, 0.5, 1, 1.5, 2, 3, 4, 6} times a power-of-two block scale), and every such value is
an exact e4m3 number on the same 32-value block, so `mx.quantize(mode="mxfp8")` returns the same values and the quantized GEMM's
half-rounded product, relu * head weight and head sums are the ones the fp16 path gives. Synthetic shapes only (no model).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import ds_pf_idx as PI


def _fp4_ue8m0(x: np.ndarray) -> np.ndarray:
    """The vendor `fake_quant_fp4_ue8m0` round trip (block 32, ue8m0 scale) through the repo's packer."""
    from kyojin_mlx import fp4pack
    xa = mx.array(x.astype(np.float32))
    codes, scales = fp4pack.pack(xa, 32, e4m3_scale=False)
    return np.array(fp4pack.unpack(codes, scales, 32, e4m3_scale=False, dtype=mx.float16)).astype(np.float32)


def _case(T=8, Tc=8192, seed=0):
    rng = np.random.default_rng(seed)
    q = _fp4_ue8m0(rng.standard_normal((T, 32, 128)).astype(np.float32) * 0.3).astype(np.float16)
    k = _fp4_ue8m0(rng.standard_normal((Tc, 128)).astype(np.float32) * 0.7).astype(np.float16)
    w = (rng.standard_normal((T, 32)) * 0.1).astype(np.float16)
    return mx.array(q)[None], mx.array(k)[None], mx.array(w)[None]


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the GPU")
def test_scores_qmm_matches_fp16():
    q, k, w = _case()
    ref = PI.scores_range(q, k, w, 0, 2, 0, q.shape[1])
    kq = PI.qmm_pack(k)
    got = PI.scores_range(q, k, w, 0, 2, 0, q.shape[1], kq=kq)
    a, b = np.array(ref), np.array(got)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b)), "mask differs"
    d = np.abs(a[fin] - b[fin])
    assert d.max() < 2e-3 * (np.abs(a[fin]).max() + 1e-9), f"score max-diff {d.max():.2e}"
    ka = np.sort(np.argsort(-a, axis=-1)[:, :64], axis=-1)
    kb = np.sort(np.argsort(-b, axis=-1)[:, :64], axis=-1)
    assert np.array_equal(ka, kb), "top-64 set differs"


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the GPU")
def test_topk_set_qmm():
    from kyojin_mlx import idx_topk as IT
    q, k, w = _case(T=4, Tc=8192, seed=7)
    kq = PI.qmm_pack(k)
    a = PI.scores_range(q, k, w, 0, 2, 0, q.shape[1])
    b = PI.scores_range(q, k, w, 0, 2, 0, q.shape[1], kq=kq)
    assert np.array_equal(np.array(IT.topk(a, 32)), np.array(IT.topk(b, 32)))

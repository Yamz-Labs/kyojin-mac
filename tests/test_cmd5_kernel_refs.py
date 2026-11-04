# SPDX-License-Identifier: Apache-2.0
"""cmd5b: why the Metal tests of the dec_stage glue kernels (stagek mix z, stagek2 xgemv) compare badly against an `mx` matmul reference, decided on a CPU.

Two facts, both numbers from the numpy emulations in tests/kernel_order.py:
  1. the kernels' own reduction order is within ~1e-6 of the exact float64 result (bounds below are 10x that), so a tolerance of 1e-4 / 2e-4 is not what
     the kernel needs;
  2. a reference matmul whose operands keep 10 mantissa bits (the tensor-unit fp32 product, fuse_attn.trunc10, measured by the repo) is 4e-4 .. 1.7e-3 away from exact
     whenever the weight is a full fp32 value, and exactly equal when the weight is fp16 / bf16 (10 and 7 bits are kept whole) -- the pattern of the reds
     (xgemv: the 5 float32 cases; mix z: the multi-row cases).
So on hardware with that matmul the GPU reference, not the kernel, is the inexact side. The Metal tests now compare the kernels with the float64 result."""
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_order as KO  # noqa: E402

pytest.importorskip("ml_dtypes")

XGEMV_BOUND = 1e-5          # measured kernel order: 1.05e-6 at K = 5120, 24 rows, 512 outputs
Z_BOUND = 1e-5              # measured kernel order: 7.6e-7 .. 1.4e-6


@pytest.mark.parametrize("wdt", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("R", [1, 6])
def test_xgemv_order_is_within_1e5_of_exact(R, wdt):
    rng = np.random.default_rng(R)
    x = (rng.standard_normal((R, 5120)) * 0.7).astype(np.float16)
    w = KO.weight_as((rng.standard_normal((48, 5120)) * 0.02).astype(np.float32), wdt)
    got = KO.xgemv_order(KO.f32(x), w).astype(np.float64)
    assert np.abs(got - x.astype(np.float64) @ w.astype(np.float64).T).max() < XGEMV_BOUND


@pytest.mark.parametrize("R", [1, 6])
def test_xgemv_tensor_unit_reference_is_far_off_for_fp32_weights_only(R):
    rng = np.random.default_rng(R)
    x = (rng.standard_normal((R, 5120)) * 0.7).astype(np.float16)
    w32 = (rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)
    err = {}
    for wdt in ("float32", "float16", "bfloat16"):
        w = KO.weight_as(w32, wdt)
        ref = KO.trunc10(KO.f32(x)).astype(np.float64) @ KO.trunc10(w).astype(np.float64).T
        err[wdt] = np.abs(ref - x.astype(np.float64) @ w.astype(np.float64).T).max()
    assert err["float32"] > 1e-3 > 2e-4 and err["float16"] == 0.0 and err["bfloat16"] == 0.0, err      # the old bound was 2e-4


def _inputs():
    spec = importlib.util.spec_from_file_location("tk_inputs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dec_stage_k.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.inputs


@pytest.mark.parametrize("D,R", [(512, 1), (512, 6), (4096, 1)])
def test_mix_z_order_is_within_1e5_of_exact_and_the_tensor_unit_reference_is_not(D, R):
    x, cpre, fn, scale, base, w, y = _inputs()(R, D, seed=D + R)
    xf = KO.f32(x.reshape(R, 4 * D))
    ex = KO.exact_z(xf, fn)
    assert np.abs(KO.mix_z_order(xf, fn).astype(np.float64) - ex).max() < Z_BOUND
    assert np.abs(KO.trunc10_z(xf, fn) - ex).max() > 1e-4            # the old bound of test_kernels_match_the_references (z < 1e-4) cannot hold against it


def test_trunc10_keeps_ten_mantissa_bits():
    a = np.array([1.0 + 2.0 ** -10, 1.0 + 2.0 ** -11, -1.0 - 2.0 ** -10 - 2.0 ** -12], np.float32)
    assert KO.trunc10(a).tolist() == [1.0 + 2.0 ** -10, 1.0, -1.0 - 2.0 ** -10]
    h = np.random.default_rng(0).standard_normal(1000).astype(np.float16).astype(np.float32)
    assert np.array_equal(KO.trunc10(h), h)                           # fp16 values are exact in 10 bits

# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_KQ_DENSE (foreign_kqgmm.dense): a dense Q8_0 linear as ONE group of the grouped kernel. CPU half: the kernel source builds for every RT, the gate. Metal half
(KYOJIN_METAL_TESTS=1): dense() equals the stored wide product (DevPack.apply) up to the fp32 summation order, for RT 1 / 2 / 4 and ragged row counts."""
import os

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_kqgmm as KQ, foreign_linear as FL, foreign_synth as SY

METAL = mx.metal.is_available() and os.environ.get("KYOJIN_METAL_TESTS", "0") not in ("0", "")


@pytest.mark.parametrize("rt", [1, 2, 3, 4])
def test_q8_source_builds(rt):
    assert KQ.source("q8_0", rt=rt)


def test_gate(monkeypatch):
    class P:
        E, fmt, n, k, rowgs = 0, "q8_0", 16384, 3072, False
    monkeypatch.setenv("KYOJIN_KQ_DENSE", "0")
    assert not KQ.dense_ok(P, 512)
    monkeypatch.setenv("KYOJIN_KQ_DENSE", "1")
    assert KQ.dense_ok(P, 512) and not KQ.dense_ok(P, 16)
    P.fmt = "q4_k"
    assert not KQ.dense_ok(P, 512)
    P.fmt, P.n = "q8_0", 1024                                  # small weight: the shipped product
    assert not KQ.dense_ok(P, 512)


def test_dense_rt_by_shape():
    assert KQ.dense_rt(16384, 3072, 512) == 2 and KQ.dense_rt(8192, 3072, 512) == 1
    assert KQ.dense_rt(3072, 16384, 512) == 1 and KQ.dense_rt(16384, 1024, 512) == 1


@pytest.mark.skipif(not METAL, reason="Metal half: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("n,k,R,rt", [(1024, 3072, 512, 1), (1024, 3072, 512, 2), (2048, 1024, 301, 4), (576, 3072, 640, 2)])
def test_dense_equals_wide(n, k, R, rt):
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(1)
        dp = FL.DevPack(GF.to_pack("Q8_0", SY.random_blocks("Q8_0", n, k, rng), k))
        mx.eval(dp.arrays())
        x = mx.array(rng.standard_normal((R, k)).astype(np.float16))
        ref = dp._wide(x)
        out = KQ.dense(dp, x, MT=32, RT=rt)
        mx.eval(ref, out)
        d = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()) / max(1.0, float(mx.max(mx.abs(ref.astype(mx.float32))).item()))
        print(f"kq dense q8_0 {n}x{k} R {R} RT {rt}: max rel diff vs wide {d:.3g}")
        assert d < 2e-3
    finally:
        mx.set_default_device(mx.cpu)


@pytest.mark.skipif(not METAL, reason="Metal half: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("n,k,R", [(1024, 3072, 512), (576, 3072, 301), (3072, 1536, 640)])
def test_native_q8_int8_gemm(n, k, R):
    """KYOJIN_PF_GEMM_I8_NATIVE: stored Q8_0 codes x per-(row, 32 K) int8 activations vs the fp16 decode matmul: only the activation rounding (~0.4 % per block) differs."""
    from kyojin_mlx import foreign_pfi8 as PI8
    mx.set_default_device(mx.gpu)
    try:
        os.environ["KYOJIN_PF_GEMM_I8_NATIVE"] = "1"
        rng = np.random.default_rng(2)
        dp = FL.DevPack(GF.to_pack("Q8_0", SY.random_blocks("Q8_0", n, k, rng), k))
        mx.eval(dp.arrays())
        x = mx.array(rng.standard_normal((R, k)).astype(np.float16))
        ref = x.astype(mx.float32) @ dp.dequant().astype(mx.float32).T
        out = PI8.native_linear(dp, x)
        assert out is not None
        mx.eval(ref, out)
        d = float(mx.sqrt(mx.mean((out.astype(mx.float32) - ref) ** 2)).item()) / float(mx.sqrt(mx.mean(ref ** 2)).item())
        print(f"native q8_0 x int8 {n}x{k} R {R}: rms rel err vs fp16 decode {d:.3g}")
        assert d < 1.5e-2
    finally:
        os.environ.pop("KYOJIN_PF_GEMM_I8_NATIVE", None)
        mx.set_default_device(mx.cpu)

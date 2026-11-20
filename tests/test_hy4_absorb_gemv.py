# SPDX-License-Identifier: Apache-2.0
"""Flag absorb_gemv: the per-head absorb GEMV (16-bit and Q8_0 copies of k_b / v_b) equals the fp32 matmul over the same weights; the Q8_0 rebuild returns the stored
blocks of a Q8_0 tensor. The kernel part is behind KYOJIN_METAL_TESTS=1."""
import os
import sys
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import absorb_gemv as AG  # noqa: E402
from kyojin_mlx import flags as FL  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"


@pytest.fixture(autouse=True)
def _keep_default_device():
    """Every test hands the default device back as it found it: test_hy4_attn_slice pins the CPU at import and the CPU-only files that run after this one (test_hy4_fixed_kv) rely on it; a GPU default there turns their exact comparisons into 1e-4 ones."""
    prev = mx.default_device()
    yield
    mx.set_default_device(prev)


def _q8_tensor(rng, H, N, K):
    """a tensor that really was Q8_0: fp16 d per 32, int8 q with max |q| = 127 in every block, dequantised to fp32 (what the loader hands over)"""
    q = rng.integers(-127, 128, size=(H, N, K // 32, 32)).astype(np.int8)
    q[..., 0] = np.where(rng.random(q[..., 0].shape) < 0.5, 127, -127)
    d = (rng.random((H, N, K // 32, 1)) * 0.02 + 0.001).astype(np.float16)
    return (q.astype(np.float32) * d.astype(np.float32)).reshape(H, N, K), q.reshape(H, N, K), d.reshape(H, N, K // 32)


def test_flag_registered_and_runtime():
    assert FL.FLAGS["absorb_gemv"].runtime and FL.FLAGS["absorb_gemv"].kind == "int"
    old = AG.MODE
    try:
        FL.set_many({"absorb_gemv": 2})
        assert AG.MODE == 2
        FL.set_many({"absorb_gemv": 0})
        assert AG.MODE == 0
    finally:
        AG.MODE = old


def test_q8_rebuild_is_the_stored_blocks():
    rng = np.random.default_rng(0)
    w, q, d = _q8_tensor(rng, 3, 16, 96)
    (wq, ws), err = AG.pack(mx.array(w), 2)
    assert np.array_equal(np.array(wq), q) and np.array_equal(np.array(ws), d) and err == 0.0


def test_no_gemv_off_gpu_or_flag_off():
    at = NS(__dict__={})
    m = MLA.MlaAttention(8, 2, 4, 32, 32, 8, 16)
    x = mx.zeros((1, 2, 1, 32))
    old = AG.MODE
    try:
        AG.MODE = 0
        assert AG.active(m, x, m.k_b, "k") is None
        AG.MODE = 1                                     # flag on, copies not built yet: stock
        assert AG.active(m, x, m.k_b, "k") is None
    finally:
        AG.MODE = old


@pytest.mark.skipif(not METAL, reason="Metal run: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("H,N,K", [(64, 512, 192), (64, 256, 512), (32, 512, 192), (3, 20, 96)])
@pytest.mark.parametrize("B", [1, 2, 6, 8])
@pytest.mark.parametrize("xdt", [mx.float32, mx.float16])
def test_kernel_matches_reference(mode, H, N, K, B, xdt):
    rng = np.random.default_rng(H * 7 + N + K + B)
    w, q, d = _q8_tensor(rng, H, N, K)
    x = mx.array(rng.standard_normal((H, B, K)).astype(np.float32)).astype(xdt)
    pk, _ = AG.pack(mx.array(w), mode)
    got = AG.apply(x, pk, mx.float32)
    mx.eval(got)
    wref = w.astype(np.float16).astype(np.float64) if mode == 1 else w.astype(np.float64)
    ref = np.einsum("hbk,hnk->hbn", np.array(x.astype(mx.float32)).astype(np.float64), wref)
    tol = 2e-5 * max(1.0, np.abs(ref).max())
    assert np.abs(np.array(got) - ref).max() < tol


@pytest.mark.skipif(not METAL, reason="Metal run: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("mode", [1, 2])
def test_attention_absorb_path_equals_stock(mode):
    from test_hy4_attn_slice import _fill               # importing it pins the default device to the CPU
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(5)
    m = _fill(MLA.MlaAttention(64, 4, 48, 64, 64, 8, 32, sinks=True, gate=True), rng)
    w, _, _ = _q8_tensor(rng, 4, 64, 64)
    m.k_b = mx.array(w)
    m.v_b = mx.array(_q8_tensor(rng, 4, 32, 64)[0])
    old = AG.MODE
    try:
        AG.MODE = mode
        AG.prepare(NS(layers=[NS(self_attn=m)]))
        for L in (1, 4):
            qn = mx.array(rng.standard_normal((1, 4, L, 64)).astype(np.float32))
            o = mx.array(rng.standard_normal((1, 4, L, 64)).astype(np.float32))
            n0 = MLA.MlaAttention.PATHS.get("absorb_gemv", 0)
            a, b = m._absorb(qn, m.k_b, "k"), m._absorb(o, m.v_b, "v")
            assert MLA.MlaAttention.PATHS["absorb_gemv"] == n0 + 2
            ra, rb = MLA._amm(qn, m.k_b), MLA._amm(o, m.v_b)
            # the kernel against a float64 reference over the weights it reads (16-bit rounded in mode 1); the stock fp32 matmul is only a loose cross-check: on recent GPUs it is not an exact fp32 sum
            for got, x, wt, st in ((a, qn, m.k_b, ra), (b, o, m.v_b, rb)):
                wn = np.array(wt)
                wn = wn.astype(np.float16) if mode == 1 else wn
                ref = np.einsum("bhlk,hnk->bhln", np.array(x).astype(np.float64), wn.astype(np.float64))
                assert np.abs(np.array(got) - ref).max() < 2e-5 * max(1.0, np.abs(ref).max())
                assert np.abs(np.array(got) - np.array(st)).max() < 5e-3 * max(1.0, np.abs(ref).max())
            assert a.dtype == mx.float32 and a.shape == ra.shape and b.shape == rb.shape
    finally:
        AG.MODE = old


def test_mode1_shares_the_attn_fp16_copy():
    """With the fp16 absorb copies of attn_fp16.prepare present (KYOJIN_HY4_PF_FP16), absorb_gemv mode 1 packs THOSE arrays (one resident copy, no second fp16 cast); without them it builds its own."""
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(7)
    kb, vb = mx.array(rng.standard_normal((4, 64, 64)).astype(np.float32)), mx.array(rng.standard_normal((4, 32, 64)).astype(np.float32))
    shared = NS(k_b=kb, v_b=vb, __dict__={})
    shared.__dict__["_kb16"], shared.__dict__["_vb16"] = kb.astype(mx.float16), vb.astype(mx.float16)
    alone = NS(k_b=kb, v_b=vb)
    old, rep = AG.MODE, dict(AG.REPORT)
    try:
        AG.MODE = 1
        AG.prepare(NS(layers=[NS(self_attn=shared), NS(self_attn=alone)]))
        s, a = shared.__dict__["_ag"][1], alone.__dict__["_ag"][1]
        assert s["k"][0] is shared.__dict__["_kb16"] and s["v"][0] is shared.__dict__["_vb16"]
        assert a["k"][0] is not None and a["k"][0].dtype == mx.float16 and np.array_equal(np.array(a["k"][0]), np.array(shared.__dict__["_kb16"]))
        assert AG.REPORT["bytes_f16"] - rep["bytes_f16"] == a["k"][0].nbytes + a["v"][0].nbytes     # only the layer without a shared copy adds bytes
        assert AG.REPORT.get("shared_f16_layers", 0) - rep.get("shared_f16_layers", 0) == 1
    finally:
        AG.MODE = old
        AG.REPORT.clear(); AG.REPORT.update(rep)

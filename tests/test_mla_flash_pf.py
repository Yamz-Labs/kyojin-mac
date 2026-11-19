# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_ATTN_FLASH (mla_flash_pf): the fused flash attend over the kept key list computes the same function as the masked fp16 chain.

CPU half (always): the kernel source builds (every string edit of glm_dsa_fused._SOURCE found once), the mx reference of the kernel (ids + sinks + causal test) equals
the masked chain on a global indexer layer of Hy4 shape (nh 32, 512 + 64, top_k 2048, chunks past 2048 keys), through the real MlaAttention path with the kernel call replaced by
the reference. Metal half (KYOJIN_METAL_TESTS=1): the real kernel against the same masked chain, and against the reference, ids with dead tail entries and a left-dead first chunk."""
import os

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import attn_fp16 as AF, foreign_skeleton_mla as MLA, mla_flash_pf as MF
from test_hy4_attn_slice import _chunks, _kp_model, _maxdiff  # noqa: E402

TOL16 = 3e-2


def test_source_builds():
    s = MF.source()
    assert "MSV_DSA_ROPE" in s and "HAS_SINK" in s and s.count("* RS") >= 4


def _run(m, xs, flash):
    from test_hy4_attn_slice import _cache
    from mlx_lm.models.base import create_causal_mask  # noqa: F401
    old = os.environ.get("KYOJIN_HY4_ATTN_FLASH")
    os.environ["KYOJIN_HY4_ATTN_FLASH"] = "1" if flash else "0"
    AF.ON = True
    m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    try:
        c, outs = _cache(), []
        for x in xs:
            outs.append(m(x, mask="causal" if x.shape[1] > 1 else None, cache=c))
        mx.eval(outs)
        return outs
    finally:
        AF.ON = False
        if old is None:
            os.environ.pop("KYOJIN_HY4_ATTN_FLASH", None)
        else:
            os.environ["KYOJIN_HY4_ATTN_FLASH"] = old


def _model(shape):
    rng = np.random.default_rng(5)
    if shape == "small":
        return rng, _kp_model(rng, 64, 32, 512, 24, 64, 16, 2, 16, top_k=64), 64
    return rng, _kp_model(rng, 128, 32, 512, 192, 64, 256, 16, 128), 128


@pytest.mark.parametrize("shape", ["small", "real"])
def test_flash_reference_equals_masked_chain(shape, monkeypatch):
    mx.set_default_device(mx.cpu)
    rng, m, d = _model(shape)
    xs = _chunks(rng, d, 2560 + 37 if shape == "real" else 700, 2)
    monkeypatch.setattr(MF, "attend", MF.reference)
    monkeypatch.setattr(MF, "supported", lambda *a: True)
    n0 = MLA.MlaAttention.PATHS["attn_flash"]
    ref = _run(m, xs, False)
    assert MLA.MlaAttention.PATHS["attn_flash"] == n0
    new = _run(m, xs, True)
    assert MLA.MlaAttention.PATHS["attn_flash"] > n0, "the flash path did not run"
    d16 = _maxdiff(ref, new)
    print(f"mla_flash reference {shape}: max rel diff {d16:.3g}")
    assert d16 <= TOL16


@pytest.mark.skipif(not mx.metal.is_available() or os.environ.get("KYOJIN_METAL_TESTS", "0") in ("0", ""), reason="Metal half: KYOJIN_METAL_TESTS=1")
def test_flash_kernel_equals_reference():
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(7)
        H, T, N, W = 64, 48, 900, 300                       # 64 heads: two 32-head halves; N - T = 852 earlier rows
        RS = 832
        rows = mx.array((0.7 * rng.standard_normal((1, 1, N, RS))).astype(np.float16))
        qa = mx.array((0.5 * rng.standard_normal((1, H, T, 512))).astype(np.float16))
        qpe = mx.array((0.5 * rng.standard_normal((1, H, T, 64))).astype(np.float16))
        sinks = mx.array(rng.standard_normal(H).astype(np.float32))
        ids = np.full((1, T, W), -1, np.int32)
        for t in range(T):
            n = int(rng.integers(1, W + 1)) if t % 5 else W
            q_abs = N - T + t
            ks = np.sort(rng.choice(q_abs + 1, size=min(n, q_abs + 1), replace=False)).astype(np.int32)
            ids[0, t, :len(ks)] = ks                          # ascending, dead (-1) entries last; some rows keep the whole window
        ids[0, 3, 7] = -1                                    # a dead entry in the middle of a list
        ids = mx.array(ids)
        assert MF.supported(qa, qpe, rows, ids, sinks)
        for sk in (None, sinks):
            out = MF.attend(qa, qpe, rows, ids, 0.0625, sk)
            ref = MF.reference(qa, qpe, rows, ids, 0.0625, sk)
            mx.eval(out, ref)
            d = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) / max(1.0, float(mx.max(mx.abs(ref)).item()))
            print(f"mla_flash kernel vs reference (sinks {sk is not None}): max rel diff {d:.3g}")
            assert d < 5e-3
    finally:
        mx.set_default_device(mx.cpu)


@pytest.mark.skipif(not mx.metal.is_available() or os.environ.get("KYOJIN_METAL_TESTS", "0") in ("0", ""), reason="Metal half: KYOJIN_METAL_TESTS=1")
def test_flash_path_equals_masked_chain_gpu():
    mx.set_default_device(mx.gpu)
    try:
        rng, m, d = _model("real")
        xs = _chunks(rng, d, 2560 + 37, 2)
        n0 = MLA.MlaAttention.PATHS["attn_flash"]
        ref = _run(m, xs, False)
        new = _run(m, xs, True)
        assert MLA.MlaAttention.PATHS["attn_flash"] > n0
        d16 = _maxdiff(ref, new)
        print(f"mla_flash kernel path vs masked chain: max rel diff {d16:.3g}")
        assert d16 <= TOL16
    finally:
        mx.set_default_device(mx.cpu)


def _run_swa(m, xs, flash):
    from mlx_lm.models.base import create_causal_mask
    from test_hy4_attn_slice import _cache
    old = os.environ.get("KYOJIN_HY4_ATTN_FLASH_SWA")
    os.environ["KYOJIN_HY4_ATTN_FLASH_SWA"] = "1" if flash else "0"
    AF.ON = True
    m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    try:
        c, outs = _cache(), []
        for x in xs:
            outs.append(m(x, mask=create_causal_mask(x.shape[1], c.offset, window_size=m.window), cache=c))
        mx.eval(outs)
        return outs
    finally:
        AF.ON = False
        if old is None:
            os.environ.pop("KYOJIN_HY4_ATTN_FLASH_SWA", None)
        else:
            os.environ["KYOJIN_HY4_ATTN_FLASH_SWA"] = old


def test_window_ids():
    ids = np.array(MF.window_ids(4, 10, 3))[0]                       # rows at slots 6..9, each sees the last 3 slots
    assert ids.tolist() == [[4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
    ids = np.array(MF.window_ids(2, 2, 4))[0]                        # first rows: dead entries first, ascending afterwards
    assert ids.tolist() == [[-1, -1, -1, 0], [-1, -1, 0, 1]]


def _swa_flash_model(window):
    from test_hy4_attn_slice import _swa_model
    rng = np.random.default_rng(11)
    return rng, _swa_model(rng, 64, 2, 512, 24, 64, 16, window), 64


def test_flash_swa_reference_equals_masked_chain(monkeypatch):
    mx.set_default_device(mx.cpu)
    rng, m, d = _swa_flash_model(512)
    xs = _chunks(rng, d, 1536 + 37, 0)                               # S > 1024 > window: the first rows are left-dead, later chunks fill the window
    monkeypatch.setattr(MF, "attend", MF.reference)
    monkeypatch.setattr(MF, "supported", lambda *a: True)
    n0 = MLA.MlaAttention.PATHS["attn_flash_swa"]
    ref = _run_swa(m, xs, False)
    assert MLA.MlaAttention.PATHS["attn_flash_swa"] == n0
    new = _run_swa(m, xs, True)
    assert MLA.MlaAttention.PATHS["attn_flash_swa"] > n0, "the windowed flash path did not run"
    d16 = _maxdiff(ref, new)
    print(f"mla_flash swa reference: max rel diff {d16:.3g}")
    assert d16 <= TOL16


@pytest.mark.skipif(not mx.metal.is_available() or os.environ.get("KYOJIN_METAL_TESTS", "0") in ("0", ""), reason="Metal half: KYOJIN_METAL_TESTS=1")
def test_flash_swa_kernel_equals_masked_chain_gpu():
    mx.set_default_device(mx.gpu)
    try:
        rng, m, d = _swa_flash_model(512)
        xs = _chunks(rng, d, 1536 + 37, 0)
        ref = _run_swa(m, xs, False)
        new = _run_swa(m, xs, True)
        d16 = _maxdiff(ref, new)
        print(f"mla_flash swa kernel path vs masked chain: max rel diff {d16:.3g}")
        assert d16 <= TOL16
    finally:
        mx.set_default_device(mx.cpu)

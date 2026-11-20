# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_PF_FP16 on the fixed-state KV cache (the served path): prompt chunks above the step-compile row limit run the fp16 attention (flash kernels on Metal) and equal the
concatenating-cache pf path; the fixed buffer keeps the decode dtype with fp16 values; decode / verify rows (<= 16) are bit-identical with the flag on or off."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import fixed_kv as FK  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from test_hy4_attn_slice import _fill, _kp_model, _swa_model, _chunks  # noqa: E402
from test_hy4_fixed_kv import _diff, _fixed, _ref  # noqa: E402

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1" and mx.metal.is_available()


def _model(kind, rng, d=64, nh=4, kv=32, nope=24, rope=8, v=16, top_k=32):
    if kind == "swa":
        m = _swa_model(rng, d, nh, kv, nope, rope, v, 40)
    elif kind == "kpool":
        m = _kp_model(rng, d, nh, kv, nope, rope, v, 2, 16, top_k=top_k)
        m.swa, m.window = False, 0
    else:
        m = _fill(MLA.MlaAttention(d, nh, 48, kv, nope, rope, v, sinks=True, gate=True), rng)
        m.swa, m.window = False, 0
    m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    return m


@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_pf_equals_concat_pf(kind, monkeypatch):
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(4)
    m = _model(kind, rng)
    xs = _chunks(rng, 64, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    monkeypatch.setenv("KYOJIN_HY4_PF_FP16", "1")
    ref = _ref(m, xs, m.window)
    new = _fixed(m, xs, 64, False)
    dd = _diff(ref, new)
    print(f"pf_fp16 fixed {kind}: max rel diff vs concat {dd:.3g}")
    assert dd <= 1e-4


@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_pf_keeps_fp32_buffer_and_decode_identical(kind, monkeypatch):
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(6)
    m = _model(kind, rng)
    xs = _chunks(rng, 64, 150, 0)
    xs = [xs[0][:, :90], xs[0][:, 90:]]
    dec = [mx.array(rng.standard_normal((1, r, 64)).astype(np.float32)) for r in (1, 1, 6, 1)]
    res = []
    for flag in ("1", "0"):
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", flag)
        at = m
        kp = at.kp
        c = FK.FixedKVCache(at.kv + at.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, at.window if at.swa else 0, 64)
        for x in xs:
            c.ensure(c._n + x.shape[1], x.dtype)
            mx.eval(m(x, mask=None, cache=c))
            c._n += x.shape[1]
        assert c.cache[0].dtype == mx.float32
        if flag == "1":
            n = c._n
            b = c.cache[0][..., :n, :]
            assert float(mx.max(mx.abs(b.astype(mx.float16).astype(mx.float32) - b)).item()) == 0.0, "the prompt rows must be fp16 values"
        # decode rows from a cache prefilled with the SAME flag: compare the flag state at decode time only
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", "1")
        outs = []
        c2 = c
        for x in dec:
            c2.ensure(c2._n + x.shape[1], x.dtype)
            o = m(x, mask=None, cache=c2)
            c2._n += x.shape[1]
            outs.append(o)
        mx.eval(outs)
        res.append(outs)
    # the prefill differs (fp16 vs fp32 attention), so decode outputs differ slightly; the decode-only switch is checked below
    assert _diff(res[1], res[0]) <= 5e-3


@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_decode_rows_ignore_the_flag(kind, monkeypatch):
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(8)
    m = _model(kind, rng)
    xs = _chunks(rng, 64, 150, 0)
    dec = [mx.array(rng.standard_normal((1, r, 64)).astype(np.float32)) for r in (1, 1, 6, 1)]
    outs = []
    for flag in ("1", "0"):
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", "0")
        kp = m.kp
        c = FK.FixedKVCache(m.kv + m.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, m.window if m.swa else 0, 64)
        for x in [xs[0][:, :90], xs[0][:, 90:]]:
            c.ensure(c._n + x.shape[1], x.dtype)
            mx.eval(m(x, mask=None, cache=c))
            c._n += x.shape[1]
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", flag)
        o = []
        for x in dec:
            c.ensure(c._n + x.shape[1], x.dtype)
            o.append(np.array(m(x, mask=None, cache=c)).tobytes())
            c._n += x.shape[1]
        outs.append(o)
    assert outs[0] == outs[1]


@pytest.mark.skipif(not METAL, reason="Metal half: KYOJIN_METAL_TESTS=1")
@pytest.mark.parametrize("kind", ["swa", "kpool"])
def test_fixed_pf_flash_equals_concat_pf_flash_gpu(kind, monkeypatch):
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(9)
        d = 1024
        m = _model(kind, rng, d=d, nh=32, kv=512, nope=192, rope=64, v=256, top_k=64)
        xs = _chunks(rng, d, 2 * 512 + 40, 4)
        xs = [xs[0], xs[1], xs[2]] + xs[3:]
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", "1")
        monkeypatch.setenv("KYOJIN_HY4_ATTN_FLASH", "1")
        monkeypatch.setenv("KYOJIN_HY4_ATTN_FLASH_SWA", "1")
        n_ref = dict(MLA.MlaAttention.PATHS)
        ref = _ref(m, xs, m.window)
        new = _fixed(m, xs, 512, False)
        key = "attn_flash_swa" if kind == "swa" else "attn_flash"
        assert MLA.MlaAttention.PATHS[key] >= n_ref[key] + 6, "the flash kernel did not run on the concat and fixed paths"
        dd = _diff(ref, new)
        print(f"pf flash fixed vs concat {kind}: max rel diff {dd:.3g}")
        assert dd <= 1e-3
    finally:
        mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fixed_pf_kv32_rows_bit_equal_to_stock(kind, monkeypatch):
    """KYOJIN_HY4_PF_FP16_KV32: the buffer rows of a prompt chunk are the stock fp32 products (bit-equal), the outputs still differ (fp16 q / scores / P @ V)."""
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(11)
    m = _model(kind, rng)
    xs = _chunks(rng, 64, 150, 0)
    xs = [xs[0][:, :90], xs[0][:, 90:]]
    bufs, outs = {}, {}
    for tag, env in (("stock", ("0", "0")), ("pf", ("1", "0")), ("kv32", ("1", "1"))):
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16", env[0])
        monkeypatch.setenv("KYOJIN_HY4_PF_FP16_KV32", env[1])
        kp = m.kp
        c = FK.FixedKVCache(m.kv + m.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, m.window if m.swa else 0, 64)
        o = []
        for x in xs:
            c.ensure(c._n + x.shape[1], x.dtype)
            o.append(m(x, mask=None, cache=c))
            c._n += x.shape[1]
        mx.eval(o, c.cache)
        bufs[tag], outs[tag] = np.array(c.cache[0][..., :c._n, :]), o
    assert bufs["kv32"].tobytes() == bufs["stock"].tobytes(), "KV32 buffer rows must equal the stock fp32 rows"
    assert bufs["pf"].tobytes() != bufs["stock"].tobytes(), "plain PF rounds the rows (control)"
    assert _diff(outs["stock"], outs["kv32"]) > 0

# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_PF_FP16: the fp16 attention path for prompt chunks only. (1) prompt chunks equal the attn_fp16 path they replace (same products, same rounding); (2) the cache keeps the
decode dtype (fp32 rows holding fp16 values); (3) a decode step (L == 1) is bit-identical with the flag on or off from the same cache."""
import os

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import attn_fp16 as AF
from test_hy4_attn_slice import _cache, _chunks, _kp_model, _maxdiff  # noqa: E402


def _model():
    rng = np.random.default_rng(5)
    m = _kp_model(rng, 64, 32, 512, 24, 64, 16, 2, 16, top_k=64)
    m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    return rng, m


def _prefill(m, xs, pf, a16=False):
    old = os.environ.get("KYOJIN_HY4_PF_FP16")
    os.environ["KYOJIN_HY4_PF_FP16"] = "1" if pf else "0"
    AF.ON = a16
    try:
        c, outs = _cache(), []
        for x in xs:
            outs.append(m(x, mask="causal", cache=c))
        mx.eval(outs, c.keys)
        return c, outs
    finally:
        AF.ON = False
        if old is None:
            os.environ.pop("KYOJIN_HY4_PF_FP16", None)
        else:
            os.environ["KYOJIN_HY4_PF_FP16"] = old


def test_pf_equals_attn_fp16_and_keeps_decode_dtype():
    mx.set_default_device(mx.cpu)
    rng, m = _model()
    xs = _chunks(rng, 64, 700, 0)
    c_ref, ref = _prefill(m, xs, pf=False, a16=True)
    c_pf, new = _prefill(m, xs, pf=True)
    assert c_ref.keys.dtype == mx.float16
    assert c_pf.keys.dtype == mx.float32, "the cache must keep the decode dtype"
    assert _maxdiff(ref, new) <= 1e-6
    assert float(mx.max(mx.abs(c_pf.keys[..., :c_pf.offset, :].astype(mx.float16).astype(mx.float32) - c_pf.keys[..., :c_pf.offset, :])).item()) == 0.0   # fp16 values


def test_decode_step_identical_with_flag_on_or_off():
    mx.set_default_device(mx.cpu)
    rng, m = _model()
    xs = _chunks(rng, 64, 700, 0)
    x1 = mx.array(rng.standard_normal((1, 1, 64)).astype(np.float32))
    c_a, _ = _prefill(m, xs, pf=True)
    c_b, _ = _prefill(m, xs, pf=True)
    outs = []
    for c, flag in ((c_a, "1"), (c_b, "0")):
        old = os.environ.get("KYOJIN_HY4_PF_FP16")
        os.environ["KYOJIN_HY4_PF_FP16"] = flag
        try:
            o = m(x1, mask=None, cache=c)
            mx.eval(o)
            outs.append(np.array(o).tobytes())
        finally:
            if old is None:
                os.environ.pop("KYOJIN_HY4_PF_FP16", None)
            else:
                os.environ["KYOJIN_HY4_PF_FP16"] = old
    assert outs[0] == outs[1], "decode (L == 1) must not depend on the prefill-only flag"


@pytest.mark.skipif(not mx.metal.is_available() or os.environ.get("KYOJIN_METAL_TESTS", "0") in ("0", ""), reason="Metal half: KYOJIN_METAL_TESTS=1")
def test_pf_flash_path_equals_attn_fp16_flash_gpu():
    from test_mla_flash_pf import _model as flash_model
    mx.set_default_device(mx.gpu)
    old = os.environ.get("KYOJIN_HY4_ATTN_FLASH")
    os.environ["KYOJIN_HY4_ATTN_FLASH"] = "1"
    try:
        rng, m, d = flash_model("real")
        m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
        xs = _chunks(rng, d, 2560 + 37, 0)
        c_ref, ref = _prefill(m, xs, pf=False, a16=True)
        c_pf, new = _prefill(m, xs, pf=True)
        assert c_pf.keys.dtype == mx.float32 and c_ref.keys.dtype == mx.float16
        dd = _maxdiff(ref, new)
        print(f"pf_fp16 flash path vs attn_fp16 flash path: max rel diff {dd:.3g}")
        assert dd <= 1e-4
    finally:
        mx.set_default_device(mx.cpu)
        if old is None:
            os.environ.pop("KYOJIN_HY4_ATTN_FLASH", None)
        else:
            os.environ["KYOJIN_HY4_ATTN_FLASH"] = old


def test_kv32_cache_rows_are_the_stock_fp32_rows():
    """KYOJIN_HY4_PF_FP16_KV32: the first chunk of one layer writes exactly the stock fp32 KV rows (no fp16 rounding), the attention is still the fp16 path."""
    mx.set_default_device(mx.cpu)
    rng, m = _model()
    xs = _chunks(rng, 64, 300, 0)[:1]
    outs = {}
    for name, (pf, kv32) in {"stock": ("0", "0"), "pf": ("1", "0"), "kv32": ("1", "1")}.items():
        old = (os.environ.get("KYOJIN_HY4_PF_FP16"), os.environ.get("KYOJIN_HY4_PF_FP16_KV32"))
        os.environ["KYOJIN_HY4_PF_FP16"], os.environ["KYOJIN_HY4_PF_FP16_KV32"] = pf, kv32
        try:
            c = _cache()
            o = m(xs[0], mask="causal", cache=c)
            mx.eval(o, c.keys)
            outs[name] = (np.array(c.keys[..., :c.offset, :]), np.array(o))
        finally:
            for k, v in zip(("KYOJIN_HY4_PF_FP16", "KYOJIN_HY4_PF_FP16_KV32"), old):
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert outs["stock"][0].dtype == np.float32 and outs["kv32"][0].dtype == np.float32
    assert np.array_equal(outs["kv32"][0], outs["stock"][0]), "KV32 rows must be the stock fp32 rows"
    assert not np.array_equal(outs["pf"][0], outs["stock"][0]), "plain PF_FP16 rounds the KV rows to fp16"

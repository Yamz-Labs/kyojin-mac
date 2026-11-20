# SPDX-License-Identifier: Apache-2.0
"""attn_fp16: the absorbed-MLA attention on an fp16 fixed-state KV buffer (fp16 projections, fp16 absorbs, fp32 softmax) stays within fp16 rounding of the fp32
path, eager and compiled, plain / windowed / kpool layers; the saturating clamp holds; the prepare step builds the fp16 absorb copies once. CPU only."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import attn_fp16 as AF  # noqa: E402
from kyojin_mlx import fixed_kv as FK  # noqa: E402
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from test_hy4_attn_slice import _fill, _kp_model, _swa_model, _chunks  # noqa: E402


def _run(m, xs, dtype, compiled, parts=None):
    """parts None: stock fp32 (flag off) or the buffer dtype given (flag on, all parts: the dtype fp16); parts int: flag on with that attn_fp16_parts mask (buffer fp16 iff bit 1)"""
    if parts is None and dtype == mx.float16:
        parts = 7
    on0, p0 = AF.ON, AF.PARTS
    AF.ON, AF.PARTS = parts is not None, parts if parts is not None else p0
    try:
        return _run_(m, xs, mx.float16 if parts is not None and parts & 1 else mx.float32 if parts is not None else dtype, compiled)
    finally:
        AF.ON, AF.PARTS = on0, p0


def _run_(m, xs, dtype, compiled):
    at = m
    kp = at.kp
    c = FK.FixedKVCache(at.kv + at.r + (2 * kp["dx"] if kp else 0), kp["kp"] if kp else 1, at.window if at.swa else 0, 64)
    outs, fn = [], None
    for x in xs:
        L = x.shape[1]
        c.ensure(c._n + L, dtype)
        if compiled and L == 1:
            if fn is None:
                def fn(x_, buf, off):
                    sh = FK.FixedKVCache.__new__(FK.FixedKVCache)
                    sh.__dict__.update(c.__dict__)
                    sh.cache = [buf, off]
                    o = m(x_, mask=None, cache=sh)
                    return o, sh.cache[0], sh.cache[1]
                fn = mx.compile(fn)
            o, b, f = fn(x, *c.cache)
            c.cache[:] = [b, f]
        else:
            o = m(x, mask=None, cache=c)
        c._n += L
        outs.append(o)
    mx.eval(outs, c.cache)
    assert c.cache[0].dtype == dtype
    return outs


def _prep(m):
    a, b = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    mx.eval(a, b)
    m.__dict__["_kb16"], m.__dict__["_vb16"] = a, b


@pytest.mark.parametrize("parts", [7, 1, 2, 4, 6])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["plain", "swa", "kpool"])
def test_fp16_close_to_fp32(kind, compiled, parts):
    rng = np.random.default_rng(5)
    d = 64
    if kind == "swa":
        m = _swa_model(rng, d, 4, 32, 24, 8, 16, 40)
    elif kind == "kpool":
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=2048)      # keep every pool: a marginal-pool flip from fp16 score rounding would move the output (the NLL gate covers it)
        m.swa, m.window = False, 0
    else:
        m = _fill(MLA.MlaAttention(d, 4, 48, 32, 24, 8, 16, sinks=True, gate=True), rng)
        m.swa, m.window = False, 0
    xs = _chunks(rng, d, 150, 6)
    xs = [xs[0][:, :90], xs[0][:, 90:]] + xs[1:]
    ref = _run(m, xs, mx.float32, compiled)
    _prep(m)
    new = _run(m, xs, mx.float16, compiled, parts=parts)
    mag = max(1.0, max(float(mx.max(mx.abs(x)).item()) for x in ref))
    dd = max(float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()) for x, y in zip(ref, new)) / mag
    print(f"attn_fp16 {kind} parts={parts} compiled={compiled}: max rel diff {dd:.3g}")
    assert dd <= 6e-2           # fp16 rounding of q / latent / absorbs / scores on random weights (kpool: a selection tie may flip one key)


def test_sat_clamp():
    x = mx.array([1e6, -1e6, 3.0, float("inf")], dtype=mx.float32)
    y = AF.sat(x)
    assert y.dtype == mx.float16 and float(y[0].item()) == 65504.0 and float(y[1].item()) == -65504.0 and float(y[2].item()) == 3.0


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["plain", "kpool"])
def test_fused_attend_dispatch(kind, compiled, dtype, monkeypatch):
    """flag mla_attend_fuse x attn_fp16: the fixed-state call reaches the fused attend (PATHS["attend_fuse"]) on an fp32 and on an fp16 buffer, eager and inside mx.compile, kpool or not;
    the counters tell why a layer refuses (the CPU device refuses: attend_fuse_no_device) and the hook lives where the step compile traces it. n8 read 'never traces' from a session in which the
    arm with the flag had not run (flag restored after every arm, counters of another request kind): the counters are per session, read them in the arm's own request."""
    from kyojin_mlx import mla_attend_fused as MAF
    rng = np.random.default_rng(9)
    d = 64
    if kind == "kpool":
        m = _kp_model(rng, d, 4, 32, 24, 8, 16, 2, 16, top_k=2048)
        m.swa, m.window = False, 0
    else:
        m = _fill(MLA.MlaAttention(d, 4, 48, 32, 24, 8, 16, sinks=True, gate=True), rng)
        m.swa, m.window = False, 0
    if dtype == mx.float16:
        _prep(m)
    calls = []

    def spy(qa, q_pe, buf, off, keep, sinks, scale, win, kv, rd):
        calls.append((qa.dtype, buf.dtype, qa.shape))
        return mx.zeros(qa.shape, dtype=qa.dtype)
    monkeypatch.setattr(MAF, "ON", True)
    monkeypatch.setattr(MAF, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(MAF, "attend", spy)
    for k in ("attend_fuse", "attend_seen"):
        MLA.MlaAttention.PATHS.pop(k, None)
    xs = _chunks(rng, d, 90, 3)
    _run(m, xs, dtype, compiled)
    P = MLA.MlaAttention.PATHS
    assert P.get("attend_seen", 0) >= 1 and P.get("attend_fuse", 0) == P["attend_seen"], P
    assert calls and all(c[0] == dtype and c[1] == dtype for c in calls)
    # flag off: the stock core, no fused call
    monkeypatch.setattr(MAF, "ON", False)
    n = len(calls)
    _run(m, xs, dtype, compiled)
    assert len(calls) == n

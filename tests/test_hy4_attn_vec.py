# SPDX-License-Identifier: Apache-2.0
"""Flag mla_decode_vec (KYOJIN_HY4_ATTN_VEC): the one-launch decode attention core of an absorbed-MLA layer.
CPU part (runs anywhere): the cell-list oracle `mla_decode_vec.reference` equals the stock `MlaAttention.attend` chain (window, keep mask, key ids with -1 padding, sinks, 1..8 rows, offset below / above
the window), the flag on a CPU device falls back to the stock core with a refusal counter, the layer wiring reaches the kernel for window / keep / ids layers (spy).
Metal part (KYOJIN_METAL_TESTS=1, chain cell stage 0): the kernel equals the oracle over fp32 / fp16 buffers and queries, every row count, every cell kind; the layer call (eager and compiled) with the flag
on equals the flag off."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import foreign_skeleton_mla as MLA  # noqa: E402
from kyojin_mlx import mla_decode_vec as MV  # noqa: E402
from test_attn_fp16 import _run_  # noqa: E402
from test_hy4_attn_slice import _fill, _kp_model  # noqa: E402

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"
KV, RD = MV.KV, MV.RD


@pytest.fixture(autouse=True)
def _device():
    d = mx.default_device()
    yield
    mx.set_default_device(d)


def _inputs(rng, nh, L, C, D, dt_q=mx.float32, dt_b=mx.float32):
    qa = mx.array(rng.standard_normal((1, nh, L, KV)).astype(np.float32) * 0.15).astype(dt_q)
    qp = mx.array(rng.standard_normal((1, nh, L, RD)).astype(np.float32) * 0.3).astype(dt_q)
    buf = mx.array(rng.standard_normal((1, 1, C, D)).astype(np.float32)).astype(dt_b)
    sinks = mx.array(rng.standard_normal((nh,)).astype(np.float32))
    return qa, qp, buf, sinks


def _extra(rng, kind, L, C, off):
    if kind == 0:
        return None
    pos = off + np.arange(L)
    if kind == 1:
        k = rng.random((L, C)) < 0.3
        k[np.arange(L), pos] = True
        return mx.array(k[None, None])
    W = 96
    ids = np.full((L, W), -1, dtype=np.int32)
    for r in range(L):
        n = int(rng.integers(5, 60))
        pick = rng.choice(pos[r] + 1, size=min(n, pos[r] + 1), replace=False)
        pick[0] = pos[r]
        ids[r, :len(pick)] = pick
        ids[r, W - 1] = min(pos[r] + 3, C - 1)           # a key from the future: must stay dead
    return mx.array(ids[None])


CASES = [(1, 0, 0), (1, 90, 40), (4, 10, 40), (8, 100, 128), (3, 150, 0)]        # (rows, off, window)


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("L,off,win", CASES)
def test_reference_equals_stock_attend(kind, L, off, win):
    """the oracle = the stock chain of MlaAttention.attend with identity absorbs (nope = kv_lora = v), on the CPU"""
    if kind and win:
        pytest.skip("windowed layers use kind 0; kpool layers have no window")
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(100 + 7 * L + off + kind)
    nh, C = 2, 192
    qa, qp, buf, sinks = _inputs(rng, nh, L, C, 576)
    m = MLA.MlaAttention(64, nh, 48, KV, KV, RD, KV, sinks=True)
    m.k_b = mx.broadcast_to(mx.eye(KV)[None], (nh, KV, KV)) * 1.0
    m.v_b = m.k_b
    m.sinks = sinks
    offa = mx.array([off], dtype=mx.int32)
    extra = _extra(rng, kind, L, C, off)
    ref = MV.reference(qa, qp, buf, offa, kind, extra, sinks, m.scale, win)
    pos = off + np.arange(L)
    vis = np.arange(C)[None, :] <= pos[:, None]
    if win:
        vis &= (pos[:, None] - np.arange(C)[None, :]) < win
    keep = None
    if kind == 1:
        keep = extra
    if kind == 2:
        ids = np.array(extra[0])
        hit = np.zeros((L, C), dtype=bool)
        for r in range(L):
            v = ids[r][(ids[r] >= 0) & (ids[r] <= pos[r])]
            hit[r, v] = True
        vis &= hit
    lat, kpe = buf[..., :KV], buf[..., KV:KV + RD]
    o = m.attend(qa, qp, lat, kpe, mx.array(vis[None, None]), keep, slice_ok=False)
    mx.eval(ref, o)
    d = float(mx.max(mx.abs(ref - o.astype(mx.float32))).item()) / max(1.0, float(mx.max(mx.abs(ref)).item()))
    assert d < 2e-5, d


def test_cpu_device_refuses():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1)
    qa, qp, buf, _ = _inputs(rng, 2, 1, 64, 576)
    why = {}
    assert not MV.eligible(qa, qp, buf, 0, None, why)
    assert why == {"attn_vec_no_device": 1}


def _model(rng, kind):
    """a Hy4-shaped (latent 512 + rope 64) small layer: windowed (kind 0) or kpool (1, 2)"""
    if kind == 0:
        m = _fill(MLA.MlaAttention(64, 4, 48, KV, 24, RD, 16, sinks=True, gate=True), rng)
        m.swa, m.window = True, 40
    else:
        m = _kp_model(rng, 64, 4, KV, 24, RD, 16, 2, 16, top_k=2048)
        m.swa, m.window = False, 0
    return m


def _xs(rng, n_pre, rows):
    xs = [mx.array(rng.standard_normal((1, n_pre, 64)).astype(np.float32))]
    xs += [mx.array(rng.standard_normal((1, r, 64)).astype(np.float32)) for r in rows]
    return xs


@pytest.mark.parametrize("kind", [0, 1])
def test_wiring_reaches_kernel(kind, monkeypatch):
    """flag on: the windowed / kpool layer calls the one-launch core (spy), flag 1 leaves kpool layers on the stock core; the CPU device without the spy falls back and counts the refusal"""
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(3)
    m = _model(rng, kind)
    calls = []

    def spy(qa, q_pe, buf, off, kd, extra, sinks, scale, window):
        calls.append((kd, window, qa.shape[2]))
        return mx.zeros(qa.shape, dtype=qa.dtype)
    monkeypatch.setattr(MV, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(MV, "attend", spy)
    xs = _xs(rng, 70, [1, 1, 4])
    for mode, expect in ((0, 0), (1, 3 if kind == 0 else 0), (2, 3)):
        monkeypatch.setattr(MV, "MODE", mode)
        calls.clear()
        _run_(m, xs, mx.float32, False)
        assert len(calls) == expect, (mode, calls)
        if expect:
            assert all(c[0] == (0 if kind == 0 else 1) and c[1] == (40 if kind == 0 else 0) for c in calls)
    monkeypatch.undo()
    monkeypatch.setattr(MV, "MODE", 2)
    why0 = MLA.MlaAttention.PATHS.get("attn_vec_no_device", 0)
    a = _run_(m, xs, mx.float32, False)
    monkeypatch.setattr(MV, "MODE", 0)
    b = _run_(m, xs, mx.float32, False)
    assert MLA.MlaAttention.PATHS["attn_vec_no_device"] > why0
    assert all(float(mx.max(mx.abs(x - y)).item()) == 0.0 for x, y in zip(a, b))


# ----------------------------------------------------------------------------------------------- Metal (chain cell stage 0)
needs_metal = pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (Metal)")


@needs_metal
@pytest.mark.parametrize("dts", [(mx.float32, mx.float32), (mx.float32, mx.float16), (mx.float16, mx.float16)], ids=["f32_f32", "f32_f16", "f16_f16"])
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("L,off,win", CASES)
@pytest.mark.parametrize("sink", [True, False])
def test_kernel_equals_reference(sink, L, off, win, kind, dts):
    if kind and win:
        pytest.skip("windowed layers use kind 0")
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7 * L + off + kind)
    nh, C, D = 4, 192, (576 if kind == 0 else 640)
    qa, qp, buf, sinks = _inputs(rng, nh, L, C, D, *dts)
    sk = sinks if sink else None
    offa = mx.array([off], dtype=mx.int32)
    extra = _extra(rng, kind, L, C, off)
    why = {}
    assert MV.eligible(qa, qp, buf, kind, extra, why), why
    out = MV.attend(qa, qp, buf, offa, kind, extra, sk, 0.0625, win)
    ref = MV.reference(qa, qp, buf, offa, kind, extra, sk, 0.0625, win)
    mx.eval(out, ref)
    assert out.shape == (1, nh, L, KV) and out.dtype == dts[0]
    d = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) / max(1.0, float(mx.max(mx.abs(ref)).item()))
    print(f"kernel vs reference kind={kind} rows={L} off={off} win={win} dts={dts}: max rel diff {d:.3g}")
    assert d < (2e-3 if dts[0] == mx.float16 else 3e-5), d


@needs_metal
def test_kernel_long_window_and_compile():
    """Hy4 shapes: 64 heads, window 128 at slot 5000, 8 rows; the same call inside mx.compile with the offset as a device array"""
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(11)
    nh, L, C = 64, 8, 5200
    qa, qp, buf, sinks = _inputs(rng, nh, L, C, 576, mx.float32, mx.float32)
    offa = mx.array([5000], dtype=mx.int32)
    f = mx.compile(lambda a, b, c, o: MV.attend(a, b, c, o, 0, None, sinks, 0.0625, 128))
    out = f(qa, qp, buf, offa)
    ref = MV.reference(qa, qp, buf, offa, 0, None, sinks, 0.0625, 128)
    mx.eval(out, ref)
    d = float(mx.max(mx.abs(out - ref)).item()) / max(1.0, float(mx.max(mx.abs(ref)).item()))
    assert d < 3e-5, d
    out2 = f(qa, qp, buf, mx.array([200], dtype=mx.int32))                     # the offset is data, not a trace constant
    ref2 = MV.reference(qa, qp, buf, mx.array([200], dtype=mx.int32), 0, None, sinks, 0.0625, 128)
    mx.eval(out2, ref2)
    assert float(mx.max(mx.abs(out2 - ref2)).item()) < 3e-5 * max(1.0, float(mx.max(mx.abs(ref2)).item()))


@needs_metal
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", [0, 1])
def test_layer_flag_on_equals_off(kind, compiled, monkeypatch):
    """the fixed-state layer call (prefill, then 1-row decode and a 4-row verify block) with the flag on equals the flag off, and the one-launch core really ran"""
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(5)
    m = _model(rng, kind)
    xs = _xs(rng, 70, [1, 1, 4, 1])
    monkeypatch.setattr(MV, "MODE", 0)
    ref = _run_(m, xs, mx.float32, compiled)
    monkeypatch.setattr(MV, "MODE", 2)
    n0 = MLA.MlaAttention.PATHS["attn_vec"]
    new = _run_(m, xs, mx.float32, compiled)
    assert MLA.MlaAttention.PATHS["attn_vec"] > n0
    mag = max(1.0, max(float(mx.max(mx.abs(x)).item()) for x in ref))
    d = max(float(mx.max(mx.abs(x - y)).item()) for x, y in zip(ref, new)) / mag
    print(f"layer kind={kind} compiled={compiled}: max rel diff {d:.3g}")
    assert d < 1e-4, d

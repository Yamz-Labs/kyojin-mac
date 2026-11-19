"""mla_attend_fused: the two-launch decode attention core against the stock formula of MlaAttention.attend (scores, causal / window mask, keep, sinks, precise softmax, P @ latent)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import mla_attend_fused as MAF  # noqa: E402

KV, RD = 512, 64


@pytest.fixture(autouse=True, params=[True, False], ids=["share", "perhead"])
def _share(request, monkeypatch):
    """n9: heads of a threadgroup share each key tile (SHARE) vs n7's one simdgroup per head: both equal the stock formula"""
    monkeypatch.setattr(MAF, "SHARE", request.param)
    MAF._k.clear()


def ref(qa, q_pe, buf, off, keep, sinks, scale, window):
    _, nh, L, _ = qa.shape
    C = buf.shape[2]
    lat, k_pe = buf[..., :KV], buf[..., KV:KV + RD]
    sc = (qa @ lat.swapaxes(-1, -2)).astype(mx.float32) * scale
    sc = sc + ((q_pe * scale) @ k_pe.swapaxes(-1, -2)).astype(mx.float32)
    kpos = mx.arange(C)
    qpos = off.reshape(-1) + mx.arange(L)
    vis = kpos[None, :] <= qpos[:, None]
    if window:
        vis = vis & (qpos[:, None] - kpos[None, :] < window)
    sc = sc + mx.where(vis[None, None], 0.0, -mx.inf)
    if keep is not None:
        sc = mx.where(keep, sc, -mx.inf)
    if sinks is not None:
        sc = mx.concatenate([sc, mx.broadcast_to(sinks.astype(mx.float32)[None, :, None, None], (1, nh, L, 1))], axis=-1)
    p = mx.softmax(sc, axis=-1, precise=True)[..., :C]
    return p.astype(mx.float16) @ lat


def mk(nh, L, C, D, off, seed, keep_on, sink_on, dt=mx.float16):
    rng = np.random.default_rng(seed)
    qa = mx.array((rng.standard_normal((1, nh, L, KV)) * 0.3).astype(np.float16))
    q_pe = mx.array((rng.standard_normal((1, nh, L, RD)) * 0.5).astype(np.float16))
    buf = np.zeros((1, 1, C, D), np.float16)
    buf[0, 0, :off + L] = (rng.standard_normal((off + L, D)) * 0.4).astype(np.float16)
    keep = None
    if keep_on:
        k = rng.random((1, 1, L, C)) < 0.4
        k[..., :1] = True                                               # slot 0 is visible to every row, so no row is fully masked
        keep = mx.array(k)
    sinks = mx.array(rng.standard_normal((nh,)).astype(np.float32)) if sink_on else None
    return qa.astype(dt), q_pe.astype(dt), mx.array(buf).astype(dt), mx.array([off], dtype=mx.int32), keep, sinks


@pytest.mark.parametrize("nh,L,C,D,off,window,keep_on,sink_on", [
    (32, 1, 1024, 576, 830, 0, False, True),
    (64, 1, 1024, 576, 300, 0, False, True),
    (32, 6, 1024, 576, 830, 0, False, True),
    (32, 1, 1024, 576, 830, 512, False, True),
    (32, 6, 1024, 576, 830, 512, False, True),
    (32, 5, 1024, 576, 20, 512, False, False),
    (64, 1, 2048, 832, 1500, 0, True, True),
    (32, 6, 2048, 832, 1500, 0, True, True),
    (32, 1, 256, 576, 0, 0, False, True),
])
def test_matches_stock(nh, L, C, D, off, window, keep_on, sink_on):
    qa, q_pe, buf, offa, keep, sinks = mk(nh, L, C, D, off, nh + L + off, keep_on, sink_on)
    assert MAF.eligible(qa, q_pe, buf, KV, RD, keep)
    scale = 256 ** -0.5
    a = MAF.attend(qa, q_pe, buf, offa, keep, sinks, scale, window, KV, RD)
    b = ref(qa, q_pe, buf, offa, keep, sinks, scale, window)
    mx.eval(a, b)
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert a.shape == b.shape
    err = np.abs(a - b).max()
    assert err < 4e-3 + 4e-3 * np.abs(b).max(), err


def test_repeat_is_deterministic():
    qa, q_pe, buf, offa, keep, sinks = mk(32, 6, 1024, 576, 700, 3, False, True)
    r = [np.array(MAF.attend(qa, q_pe, buf, offa, keep, sinks, 0.0625, 0, KV, RD).astype(mx.float32)) for _ in range(3)]
    assert (r[0] == r[1]).all() and (r[1] == r[2]).all()


def test_ineligible():
    qa, q_pe, buf, offa, keep, sinks = mk(32, 1, 256, 576, 10, 1, False, True)
    why = {}
    assert not MAF.eligible(qa.astype(mx.float32), q_pe, buf, KV, RD, None, why) and why   # mixed dtypes
    assert not MAF.eligible(qa, q_pe, buf, KV, RD, mx.ones((1, 1, 1, 7), dtype=mx.bool_))


@pytest.mark.parametrize("dt,tol", [(mx.bfloat16, 2e-2), (mx.float32, 2e-3)])
def test_dtypes(dt, tol):
    qa, q_pe, buf, offa, keep, sinks = mk(32, 3, 1024, 576, 700, 9, False, True, dt)
    why = {}
    assert MAF.eligible(qa, q_pe, buf, KV, RD, None, why) and not why
    a = MAF.attend(qa, q_pe, buf, offa, None, sinks, 0.0625, 0, KV, RD)
    assert a.dtype == dt
    b = ref(qa, q_pe, buf, offa, None, sinks, 0.0625, 0)
    err = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
    assert err < tol, err

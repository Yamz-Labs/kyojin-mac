"""tq_step (one-dispatch TurboQuant decode step): the fused norm / rope / rotation / quantize / in-place pool + ring write / decode / gated merge against the stock
chain (q / k norm, rope, update_and_fetch, attend, sigmoid gate), over GQA shape classes. Needs a Mac (Metal)."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn                                                          # noqa: E402
from kyojin_mlx import kvwin as KW, tq as TQ, tq_step as TS                  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
CLASSES = [(16, 2, 256, 64, 6, 6), (8, 8, 128, 128, 6, 6), (32, 4, 64, 32, 5, 6), (4, 1, 128, 64, 6, 5), (64, 8, 128, 64, 4, 4), (12, 4, 256, 256, 6, 6)]


@pytest.fixture(autouse=True)
def _hook():
    KW.install()


def _mod(hq, hk, d, rd, rng):
    m = types.SimpleNamespace(num_attention_heads=hq, num_key_value_heads=hk, scale=d ** -0.5, rope=nn.RoPE(rd, traditional=False, base=1e6),
                              q_norm=nn.RMSNorm(d, eps=1e-6), k_norm=nn.RMSNorm(d, eps=1e-6))
    m.q_norm.weight = mx.array(rng.uniform(0.5, 1.5, d).astype(np.float16))
    m.k_norm.weight = mx.array(rng.uniform(0.5, 1.5, d).astype(np.float16))
    return m


def _stock(c, m, qp, kp, vp, gated=True):
    hq, hk = TS._heads(m)
    q, gate = mx.split(qp.reshape(1, 1, hq, -1), 2, axis=-1) if gated else (qp.reshape(1, 1, hq, -1), None)
    d = q.shape[-1]
    k = m.k_norm(kp.reshape(1, 1, hk, d)).transpose(0, 2, 1, 3)
    v = vp.reshape(1, 1, hk, d).transpose(0, 2, 1, 3)
    q = m.q_norm(q).transpose(0, 2, 1, 3)
    q, k = m.rope(q, offset=c.offset), m.rope(k, offset=c.offset)
    c.update_and_fetch(k, v)
    out = c.attend(q, m.scale)
    out = out.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    return out * mx.sigmoid(gate.reshape(1, 1, -1)) if gated else out


@pytest.mark.parametrize("gated", [True, False])                  # gated q (Qwen3.5 / 3.6) and plain q with mlx-lm head names (Qwen3)
@pytest.mark.parametrize("hq,hk,d,rd,kb,vb", CLASSES)
def test_step_matches_stock(hq, hk, d, rd, kb, vb, gated):
    rng = np.random.default_rng(0)
    m = _mod(hq, hk, d, rd, rng)
    if not gated:
        m.n_heads, m.n_kv_heads = m.num_attention_heads, m.num_key_value_heads
        del m.num_attention_heads, m.num_key_value_heads
    a, b = (TQ.TQKVCache(kb, vb, sink=4, window=48, margin=16) for _ in range(2))
    dt = mx.float16
    errs = []
    for t in range(420):                                  # crosses the sink, the ring filling, the full ring and one pool growth (step 256)
        qp = mx.array(rng.standard_normal((1, 1, hq * (2 if gated else 1) * d)).astype(np.float16))
        kp = mx.array(rng.standard_normal((1, 1, hk * d)).astype(np.float16))
        vp = mx.array(rng.standard_normal((1, 1, hk * d)).astype(np.float16))
        ref = _stock(a, m, qp, kp, vp, gated)
        got = TS.step(b, m, qp, kp, vp, m.scale, gated=gated)
        if got is None:                                   # sink rows run the stock chain
            got = _stock(b, m, qp, kp, vp, gated)
        else:
            assert t >= 4
        mx.eval(ref, got)
        errs.append(float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))) / (mx.max(mx.abs(ref.astype(mx.float32))) + 1e-6)))
    assert b.offset == a.offset == 420 and b.rk.shape == a.rk.shape
    assert max(errs) < 2e-2, max(errs)
    n = a.offset
    np.testing.assert_allclose(np.array(b.rk.astype(mx.float32)), np.array(a.rk.astype(mx.float32)), atol=2e-2)
    # pool words: rounding of the rotation may flip a rare code; scales agree
    ws = np.array(a.kq[1][:, :, :n].astype(mx.float32)); wb = np.array(b.kq[1][:, :, :n].astype(mx.float32))
    np.testing.assert_allclose(wb, ws, rtol=2e-2, atol=1e-3)
    diff = np.mean(np.array(a.vq[0][:, :, :n]) != np.array(b.vq[0][:, :, :n]))
    assert diff < 0.02, diff


def test_restored_pool_is_copied_not_shared():
    rng = np.random.default_rng(1)
    hq, hk, d = 8, 2, 128
    m = _mod(hq, hk, d, 64, rng)
    a = TQ.TQKVCache(6, 6, sink=4, window=32, margin=16)
    for t in range(40):
        _stock(a, m, *(mx.array(rng.standard_normal(s).astype(np.float16)) for s in ((1, 1, hq * 2 * d), (1, 1, hk * d), (1, 1, hk * d))))
    f = a.fork()
    snap = np.array(f.vq[0]).copy()
    mx.eval(f.vq[0])
    got = TS.step(a, m, *(mx.array(rng.standard_normal(s).astype(np.float16)) for s in ((1, 1, hq * 2 * d), (1, 1, hk * d), (1, 1, hk * d))), m.scale)
    mx.eval(got)
    assert (np.array(f.vq[0]) == snap).all()               # the fork never sees the original's in-place row

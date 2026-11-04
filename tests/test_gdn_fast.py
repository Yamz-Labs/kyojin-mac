import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_5 as Q5
from mlx_lm.models.gated_delta import gated_delta_update as stock, normalize_qk

from kyojin_mlx import gdn_fast as GF


def inputs(T, B=1, Hk=4, Hv=8, D=128, seed=0):
    mx.random.seed(seed + T)
    q = mx.random.normal((B, T, Hk, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, D)).astype(mx.bfloat16)
    q, k = normalize_qk(q, k, inv_scale=D ** -0.5, eps=1e-6)
    v = mx.random.normal((B, T, Hv, D)).astype(mx.bfloat16)
    a = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, T, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32) * 0.3
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    return q, k, v, a, b, A_log, dt_bias


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setenv("KYOJIN_GDN_FAST", "1")
    GF.install()
    return Q5.gated_delta_update


def close(x, y, tol):
    return mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item() <= tol


@pytest.mark.parametrize("T", [64, 100, 512])
def test_matches_stock(fast, T):
    args = inputs(T)
    y0, s0 = stock(*args)
    y1, s1 = fast(*args)
    assert y1.dtype == y0.dtype and s1.dtype == mx.float32 and s1.shape == s0.shape
    assert close(y0, y1, 0.03) and close(s0, s1, 0.02)


def test_state_carried_across_calls(fast):
    """prefill in two pieces with a carried state equals one call (save/restore of the recurrent state is the same array)."""
    args = inputs(256)
    yw, sw = fast(*args)
    cut = lambda sl: tuple(t[:, sl] if t.ndim >= 3 and t.shape[1] == 256 else t for t in args)
    y1, s1 = fast(*cut(slice(0, 128)))
    y2, s2 = fast(*cut(slice(128, 256)), s1)
    assert close(yw[:, :128], y1, 0.03) and close(yw[:, 128:], y2, 0.03) and close(sw, s2, 0.02)
    # stock state fed to the fast path and back
    _, st = stock(*cut(slice(0, 128)))
    y3, s3 = fast(*cut(slice(128, 256)), st)
    y4, s4 = stock(*cut(slice(128, 256)), st)
    assert close(y3, y4, 0.03) and close(s3, s4, 0.02)


def test_decode_and_mask_go_to_stock(fast, monkeypatch):
    calls = []
    real = mx.fast.gated_delta_update
    monkeypatch.setattr(mx.fast, "gated_delta_update", lambda *a, **k: calls.append(1) or real(*a, **k))
    fast(*inputs(1))
    fast(*inputs(512), None, mx.ones((1, 512), dtype=mx.bool_))
    assert not calls
    fast(*inputs(512))
    assert calls


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("KYOJIN_GDN_FAST", raising=False)
    assert GF.min_t() == 0
    monkeypatch.setenv("KYOJIN_GDN_FAST", "0")
    assert GF.min_t() == 0

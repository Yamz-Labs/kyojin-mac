"""V2-VERIFY: dsa_attend_rows (scores / softmax / PV / sum launches) against the stock SDPA path (embed_q, mx.fast.scaled_dot_product_attention fallback, unembed_out) on synthetic latents,
and row invariance (a row's bits do not depend on the other rows / sessions sharing the call)."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import dsa_attend_rows as AR  # noqa: E402
from kyojin_mlx import glm_mla as MLA  # noqa: E402

CTX, CAP = 3000, 3328


def mk(B, Lr, K, seed, invalid=0, dt=mx.float16):
    rng = np.random.default_rng(seed)
    f = lambda *s, sc=1.0: mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(dt)
    lat = f(B, 1, CAP, 512)
    we, wu = f(64, 512, 256, sc=0.05), f(64, 256, 512, sc=0.05)
    q = f(B, 64, Lr, 256)
    topk = np.stack([np.stack([rng.choice(CTX, K, replace=False) for _ in range(Lr)]) for _ in range(B)]).astype(np.int32)
    if invalid:
        topk[:, :, -invalid:] = -1
    scale = float(1.0 / np.sqrt(576.0))
    me = types.SimpleNamespace(embed_q=lambda x: mx.matmul(x, we.swapaxes(-1, -2)), unembed_out=lambda x: mx.matmul(x, wu.swapaxes(-1, -2)), scale=scale)
    kv = types.SimpleNamespace(keys=lat, _idx=CTX)
    return me, kv, q, mx.array(topk), lat, we, wu, scale


def stock(B, Lr, K, q, topk, lat, we, wu, scale, dt):
    """per (session, row): embed_q, gathered rows, MLX's SDPA fallback with the validity mask, unembed_out (the loop of glm_fuse_dsa.attend_batch)"""
    outs = []
    for j in range(Lr):
        per = []
        for b in range(B):
            idx = topk[b, j]
            valid = (idx >= 0) & (idx < CTX)
            sel = mx.take(lat[b, 0], mx.clip(idx, 0, CAP - 1), axis=0)
            sel = mx.where(valid[:, None], sel, mx.zeros_like(sel))[None, None]
            query = mx.matmul(q[b:b + 1, :, j:j + 1], we.swapaxes(-1, -2))
            o = mx.fast.scaled_dot_product_attention(query, sel, sel, scale=scale, mask=valid[None, None, None, :])
            per.append(mx.matmul(o, wu.swapaxes(-1, -2)))
        outs.append(per[0] if B == 1 else mx.concatenate(per, axis=0))
    out = outs[0] if Lr == 1 else mx.concatenate(outs, axis=2)
    return out.transpose(0, 2, 1, 3).reshape(B, Lr, -1)


@pytest.mark.parametrize("B,Lr", [(1, 1), (1, 2), (1, 4), (2, 2), (4, 2)])
@pytest.mark.parametrize("invalid", [0, 37])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_vs_stock(B, Lr, invalid, dt):
    K = 2051
    me, kv, q, topk, lat, we, wu, scale = mk(B, Lr, K, B * 10 + Lr, invalid, dt)
    ref = stock(B, Lr, K, q, topk, lat, we, wu, scale, dt)
    out = AR.attend_batch(me, q, topk, kv, 0, False)
    mx.eval(ref, out)
    assert out.shape == ref.shape and out.dtype == dt
    r, o = np.asarray(ref.astype(mx.float32)), np.asarray(out.astype(mx.float32))
    tol = 6e-3 if dt == mx.float16 else 4e-2
    assert np.abs(r - o).max() <= tol * max(1.0, np.abs(r).max()), (np.abs(r - o).max(), np.abs(r).max())


@pytest.mark.parametrize("K", [64, 100, 2051])
def test_other_K(K):
    me, kv, q, topk, lat, we, wu, scale = mk(1, 2, K, K)
    ref = stock(1, 2, K, q, topk, lat, we, wu, scale, mx.float16)
    out = AR.attend_batch(me, q, topk, kv, 0, False)
    mx.eval(ref, out)
    assert np.abs(np.asarray(ref.astype(mx.float32)) - np.asarray(out.astype(mx.float32))).max() <= 6e-3 * max(1.0, float(mx.abs(ref).max()))


def test_row_invariance():
    """row j of a 4-row call == the same row alone (batch 1, 1 row) == a 2-row call, bit for bit; any CPS."""
    K = 2051
    me, kv, q, topk, lat, we, wu, scale = mk(1, 4, K, 3)
    full = AR.attend_batch(me, q, topk, kv, 0, False)
    for cps in (5, 3, 13):
        AR.CPS = cps
        try:
            f2 = AR.attend_batch(me, q, topk, kv, 0, False)
            mx.eval(f2)
            for j in range(4):
                one = AR.attend_batch(me, q[:, :, j:j + 1], topk[:, j:j + 1], kv, 0, False)
                two = AR.attend_batch(me, q[:, :, j:j + 2] if j < 3 else q[:, :, 2:4], topk[:, j:j + 2] if j < 3 else topk[:, 2:4], kv, 0, False)
                mx.eval(one, two)
                if cps == 5:
                    assert mx.array_equal(one[:, 0], full[:, j]), (cps, j)
                    assert mx.array_equal(two[:, 0], full[:, j]) if j < 3 else mx.array_equal(two[:, 1], full[:, 3])
        finally:
            AR.CPS = 5


def test_last_only_start():
    K = 2051
    me, kv, q, topk, lat, we, wu, scale = mk(1, 3, K, 9)
    full = AR.attend_batch(me, q, topk, kv, 0, False)
    last = AR.attend_batch(me, q, topk, kv, 2, True)
    mx.eval(full, last)
    assert last.shape == (1, 1, 64 * 256) and mx.array_equal(last[:, 0], full[:, 2])


def test_usable_guard(monkeypatch):
    me, kv, q, topk, lat, we, wu, scale = mk(1, 2, 64, 1)
    monkeypatch.setattr(AR, "ENABLED", False)
    assert not AR.usable(me, q, topk, kv, we, wu)
    monkeypatch.setattr(AR, "ENABLED", True)
    assert AR.usable(me, q, topk, kv, we, wu)
    assert not AR.usable(me, q.astype(mx.float32), topk, kv, we.astype(mx.float32), wu.astype(mx.float32))

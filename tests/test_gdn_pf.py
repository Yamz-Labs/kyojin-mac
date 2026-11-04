import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3_5 as Q5
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, normalize_qk
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

from kyojin_mlx import gdn_pf as GP


def layer(hk, hv, h=256, seed=0):
    mx.random.seed(seed)
    args = Q5.TextModelArgs(hidden_size=h, linear_num_key_heads=hk, linear_num_value_heads=hv, linear_key_head_dim=128, linear_value_head_dim=128)
    m = Q5.GatedDeltaNet(args)
    m.set_dtype(mx.bfloat16)
    m.A_log = mx.random.normal((hv,)).astype(mx.float32) * 0.3
    m.dt_bias = mx.random.normal((hv,)).astype(mx.bfloat16)
    m.eval()
    return m


def close(x, y, tol):
    return mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item() <= tol


@pytest.mark.parametrize("hk,hv,S", [(2, 4, 96), (2, 2, 200), (4, 8, 64)])
def test_pre_matches_stock_ops(hk, hv, S):
    m = layer(hk, hv)
    x = (mx.random.normal((1, S, 256)) * 0.7).astype(mx.bfloat16)
    qkv, a, b = m.in_proj_qkv(x), m.in_proj_a(x), m.in_proj_b(x)
    cs = (mx.random.normal((1, 3, m.conv_dim)) * 0.5).astype(mx.bfloat16)
    co = nn.silu(m.conv1d(mx.concatenate([cs, qkv], axis=1)))
    q, k, v = [t.reshape(1, S, h, 128) for t, h in zip(mx.split(co, [m.key_dim, 2 * m.key_dim], -1), [hk, hk, hv])]
    q, k = normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)
    g, beta = compute_g(m.A_log, a, m.dt_bias), mx.sigmoid(b)
    q2, k2, v2, cn, g2, beta2 = GP.pre(qkv, cs, m.conv1d.weight, a, b, m.A_log, m.dt_bias, hk, hv)
    assert cn.shape == (1, 3, m.conv_dim) and (cn == qkv[:, -3:]).all().item()
    for got, ref in ((q2, q), (k2, k), (v2, v), (beta2, beta)):
        assert got.dtype == ref.dtype and close(got, ref, 0.02 * max(1.0, mx.abs(ref.astype(mx.float32)).max().item()))
    assert close(g2, g, 4e-3)                            # softplus rounds to bf16 in the stock graph: one bf16 ulp of sp moves g by ~1e-3


def test_gated_norm_matches_stock():
    n = Qwen3NextRMSNormGated(128, eps=1e-6)
    n.weight = (mx.random.normal((128,)) * 0.3 + 1).astype(mx.bfloat16)
    h = (mx.random.normal((1, 50, 4, 128)) * 3).astype(mx.bfloat16)
    z = mx.random.normal((1, 50, 4, 128)).astype(mx.bfloat16)
    ref, got = n(h, z), GP.gated_norm(h, z, n.weight, n.eps)
    assert close(got, ref, 0.04 * mx.abs(ref.astype(mx.float32)).max().item())


def test_layer_matches_stock_and_carries_state(monkeypatch):
    m = layer(2, 4)
    x = (mx.random.normal((1, 160, 256)) * 0.7).astype(mx.bfloat16)
    stock = Q5.GatedDeltaNet.__call__
    monkeypatch.setattr(GP, "MIN_T", 64)
    GP.install()
    fused = Q5.GatedDeltaNet.__call__
    try:
        c0, c1 = ArraysCache(2), ArraysCache(2)
        outs = []
        for call, c in ((stock, c0), (fused, c1)):
            Q5.GatedDeltaNet.__call__ = call
            outs.append([m(x[:, :96], cache=c), m(x[:, 96:], cache=c)])   # two chunks: second one starts from a carried state
        mx.eval(outs, c0.state, c1.state)
        scale = max(1.0, mx.abs(outs[0][0].astype(mx.float32)).max().item())
        for i in range(2):
            assert close(outs[0][i], outs[1][i], 0.03 * scale)
        assert (c0[0] == c1[0]).all().item() and close(c0[1], c1[1], 0.02)
        # chunks of 9..63 rows keep the stock path
        Q5.GatedDeltaNet.__call__ = fused
        c2 = ArraysCache(2)
        assert close(m(x[:, :12], cache=c2), stock(m, x[:, :12], cache=ArraysCache(2)), 1e-6)
    finally:
        Q5.GatedDeltaNet.__call__ = stock
        Q5.GatedDeltaNet._kyojin_pf = False


def test_nax_core_matches_step_kernel():
    from mlx_lm.models.gated_delta import gated_delta_kernel
    from kyojin_mlx import gdn_nax, metal_nax
    if not metal_nax.nax_available():
        pytest.skip("no neural accelerators")
    mx.random.seed(3)
    for T, corr in ((96, 0.0), (203, 0.0), (203, 0.9)):                       # corr: keys share one direction, the chunk inverse (I + A)^-1 has large entries
        q = mx.random.normal((1, T, 2, 128)).astype(mx.bfloat16)
        k = (mx.random.normal((1, T, 2, 128)) * (1 - corr) + mx.random.normal((1, 1, 2, 128)) * 3 * corr).astype(mx.bfloat16)
        v = mx.random.normal((1, T, 4, 128)).astype(mx.bfloat16)
        q, k = [t.astype(mx.bfloat16) for t in normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)]
        g = mx.sigmoid(mx.random.normal((1, T, 4)) * 2 + 3)                 # decay close to 1: the chunk inverse sees real coupling
        beta = mx.sigmoid(mx.random.normal((1, T, 4)) + 3 * corr)
        st = mx.random.normal((1, 4, 128, 128)) * 0.1
        yr, sr = gated_delta_kernel(q, k, v, g, beta, st, None)
        y, s = gdn_nax.gated_delta_nax(q, k, v, g, beta, st)
        assert y.dtype == mx.bfloat16 and s.dtype == mx.float32
        assert close(y, yr, 0.03 * mx.abs(yr.astype(mx.float32)).max().item())
        assert close(s, sr, 0.01 * mx.abs(sr).max().item())


@pytest.mark.parametrize("rows", [1, 2, 5])
def test_decode_rows_match_stock_and_carry_state(monkeypatch, rows):
    m = layer(2, 4)
    x = (mx.random.normal((1, 80 + 6 * rows, 256)) * 0.7).astype(mx.bfloat16)
    stock = Q5.GatedDeltaNet.__call__
    monkeypatch.setattr(GP, "MIN_T", 64)
    monkeypatch.setattr(GP, "DEC", True)
    GP.install()
    fused = Q5.GatedDeltaNet.__call__
    try:
        c0, c1 = ArraysCache(2), ArraysCache(2)
        outs = []
        for call, c in ((stock, c0), (fused, c1)):
            Q5.GatedDeltaNet.__call__ = call
            o = [m(x[:, :80], cache=c)]
            for i in range(6):
                o.append(m(x[:, 80 + i * rows:80 + (i + 1) * rows], cache=c))
            outs.append(o)
        mx.eval(outs, c0.state, c1.state)
        scale = max(1.0, mx.abs(outs[0][0].astype(mx.float32)).max().item())
        for a, b in zip(*outs):
            assert close(a, b, 0.03 * scale)
        assert (c0[0] == c1[0]).all().item() and close(c0[1], c1[1], 0.02)
    finally:
        Q5.GatedDeltaNet.__call__ = stock
        Q5.GatedDeltaNet._kyojin_pf = False

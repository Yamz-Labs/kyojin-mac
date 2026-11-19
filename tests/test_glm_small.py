"""G10b: fused KDA decode body vs the stock Glm5NextLinearAttention (random weights, real GLM-5.3 shapes)."""
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import glm_small as GS

G = pytest.importorskip("mlx_vlm.models.glm5_next.language")
CFG = types.SimpleNamespace(linear_num_heads=64, linear_head_dim=128, linear_conv_kernel_dim=4, hidden_size=4096, linear_lower_bound=-5.0, rms_norm_eps=1e-5)


def make(dtype, seed=0, norm_dtype=None):
    mx.random.seed(seed)
    m = G.Glm5NextLinearAttention(CFG, 0)
    P = 64 * 128
    for n, s in (("qkv_proj", 0.02), ("fbg_a_proj", 0.05), ("f_b_proj", 0.1), ("g_b_proj", 0.1), ("o_proj", 0.02)):
        lin = getattr(m, n)
        lin.weight = (mx.random.normal(lin.weight.shape) * s).astype(dtype)
    m.qkv_conv.conv.weight = (mx.random.normal(m.qkv_conv.conv.weight.shape) * 0.5).astype(dtype)
    m.A_log = mx.random.normal((64,)) * 0.3
    m.dt_bias = mx.random.normal((P,)) * 0.3
    m.o_norm.weight = (1 + 0.1 * mx.random.normal((128,))).astype(norm_dtype or dtype)
    mx.eval(m.parameters())
    return m


def run(m, xs, fused):
    GS.ENABLED = fused
    GS.install(log=lambda *_: None)
    cache = G.ArraysCache(size=2)
    outs = []
    for x in xs:
        y = m(x, cache=cache)
        mx.eval(y, cache[0], cache[1])
        outs.append(y)
    return outs, cache


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("norm_dtype", [None, mx.float32])
def test_matches_stock(dtype, norm_dtype):
    m = make(dtype, norm_dtype=norm_dtype)
    mx.random.seed(1)
    xs = [(mx.random.normal((1, 16, 4096)) * 0.5).astype(dtype)] + [(mx.random.normal((1, L, 4096)) * 0.5).astype(dtype) for L in (1, 1, 3, 1, 8, 2)]
    ref, cr = run(m, xs, False)
    GS.STATS.update(fused=0, stock=0)
    out, cf = run(m, xs, True)
    assert GS.STATS["fused"] == 6 and GS.STATS["stock"] == 1
    for a, b in zip(ref[1:], out[1:]):
        assert a.dtype == b.dtype
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        assert d <= 0.03 * mx.abs(a.astype(mx.float32)).max().item() + 1e-3, d
    assert mx.array_equal(cr[0], cf[0]).item() or mx.abs(cr[0].astype(mx.float32) - cf[0].astype(mx.float32)).max().item() < 0.05
    rel = mx.abs(cr[1] - cf[1]).max().item() / (mx.abs(cr[1]).max().item() + 1e-9)
    assert rel < (2e-2 if dtype == mx.bfloat16 else 1e-4), rel   # bf16: 1-ulp flips of the two K = 128 projections (other summation order)


def test_batch_rows_and_row_invariance():
    """The kernel alone: a row of a B = 2 launch has the bits of the same row launched alone (the stock projections are not batch invariant, so they are fed in)."""
    m = make(mx.bfloat16, seed=3)
    mx.random.seed(5)
    P = 64 * 128
    qkv = (mx.random.normal((2, 4, 3 * P))).astype(mx.bfloat16)
    fbg = (mx.random.normal((2, 4, 320))).astype(mx.bfloat16)
    cs = (mx.random.normal((2, 3, 3 * P))).astype(mx.bfloat16)
    st = mx.random.normal((2, 64, 128, 128)) * 0.1
    consts, a_log, dtb = GS._consts(m)
    args = (m.qkv_conv.conv.weight, None, None, a_log, dtb, m.o_norm.weight, m.f_b_proj.weight, m.g_b_proj.weight, consts)
    def go(sl):
        a = list(args); a[1] = cs[sl]; a[2] = st[sl]
        return GS.kda_decode(qkv[sl], fbg[sl], *a, heads=64, out_dtype=mx.bfloat16)
    both = go(slice(0, 2)); one = go(slice(0, 1)); two = go(slice(1, 2))
    mx.eval(both, one, two)
    for k in range(3):
        assert mx.array_equal(both[k][:1], one[k]).item() and mx.array_equal(both[k][1:], two[k]).item()


def test_fallbacks():
    m = make(mx.bfloat16)
    x = (mx.random.normal((1, 1, 4096))).astype(mx.bfloat16)
    GS.ENABLED = True
    GS.STATS.update(fused=0, stock=0)
    m(x, cache=G.ArraysCache(size=2))            # empty cache -> stock
    m((mx.random.normal((1, 12, 4096))).astype(mx.bfloat16), cache=G.ArraysCache(size=2))   # > MAX_TOKENS
    assert GS.STATS == {"fused": 0, "stock": 2}


def test_row_linear_wrapper_is_eligible():
    """The served multi-session path wraps f_b / g_b in kyojin_mlx.multiuser.RowLinear (rowdot at decode): the fused body must still take over."""
    from kyojin_mlx.multiuser import RowLinear
    m = make(mx.float16)
    mx.random.seed(1)
    xs = [(mx.random.normal((1, 16, 4096)) * 0.5).astype(mx.float16)] + [(mx.random.normal((1, L, 4096)) * 0.5).astype(mx.float16) for L in (1, 1, 2)]
    ref, _ = run(m, xs, False)
    m.f_b_proj = RowLinear(m.f_b_proj)
    m.g_b_proj = RowLinear(m.g_b_proj)
    GS.STATS.update(fused=0, stock=0)
    out, _ = run(m, xs, True)
    assert GS.STATS == {"fused": 3, "stock": 1}
    for a, b in zip(ref[1:], out[1:]):
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        assert d <= 0.03 * mx.abs(a.astype(mx.float32)).max().item() + 1e-3, d


def _spec_run(m, x, T, keep, fused, spec_on, base):
    GS.ENABLED, GS.SPEC = fused, spec_on
    GS.install(log=lambda *_: None)
    cache = G.ArraysCache(size=2)
    cache[0], cache[1] = base[0], base[1]
    cache.start_speculation(T)
    GS.STATS.update(fused=0, stock=0)
    y = m(x, cache=cache)
    used = GS.STATS["fused"]
    cache.commit_speculation([keep])
    mx.eval(y, cache[0], cache[1])
    return y, cache, used


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("T", [1, 2, 3, 4, 8])
def test_speculating_cache_matches_stock(T, dtype):
    """G8d: the verify rows of a speculating cache (start_speculation) take the fused kernel and leave the same records: every kept prefix gives the stock conv window and recurrent state."""
    m = make(dtype, seed=2)
    mx.random.seed(7)
    pre = (mx.random.normal((1, 16, 4096)) * 0.5).astype(dtype)
    GS.ENABLED, GS.SPEC = False, True
    GS.install(log=lambda *_: None)
    c0 = G.ArraysCache(size=2)
    mx.eval(m(pre, cache=c0), c0[0], c0[1])
    x = (mx.random.normal((1, T, 4096)) * 0.5).astype(dtype)
    for keep in range(1, T + 1):
        ref, cr, u0 = _spec_run(m, x, T, keep, False, True, c0)
        out, cf, u1 = _spec_run(m, x, T, keep, True, True, c0)
        assert u0 == 0 and u1 == 1, (u0, u1)
        d = mx.abs(ref.astype(mx.float32) - out.astype(mx.float32)).max().item()
        assert d <= 0.03 * mx.abs(ref.astype(mx.float32)).max().item() + 1e-3, d
        assert mx.abs(cr[0].astype(mx.float32) - cf[0].astype(mx.float32)).max().item() < 0.05
        rel = mx.abs(cr[1] - cf[1]).max().item() / (mx.abs(cr[1]).max().item() + 1e-9)
        assert rel < (2e-2 if dtype == mx.bfloat16 else 1e-4), (keep, rel)
        assert cf.offset == cr.offset if hasattr(cf, "offset") else True
    # flag off for the speculating cache: the stock path, nothing recorded by the kernel
    out, cf, used = _spec_run(m, x, T, T, True, False, c0)
    assert used == 0
    GS.SPEC = True


def test_row_linear_projections_are_plain():
    """The served model swaps every nn.Linear for the row-invariant RowLinear: f_b / g_b must still take the fused kernel (G8d: they never did, the whole G10b was a no-op on the served model)."""
    from kyojin_mlx.multiuser import RowLinear
    m = make(mx.float16, seed=4)
    m.f_b_proj, m.g_b_proj = RowLinear(m.f_b_proj), RowLinear(m.g_b_proj)
    mx.random.seed(3)
    GS.ENABLED, GS.SPEC = True, True
    GS.install(log=lambda *_: None)
    cache = G.ArraysCache(size=2)
    mx.eval(m((mx.random.normal((1, 16, 4096)) * 0.5).astype(mx.float16), cache=cache), cache[0], cache[1])
    x = (mx.random.normal((1, 2, 4096)) * 0.5).astype(mx.float16)
    GS.STATS.update(fused=0, stock=0)
    mx.eval(m(x, cache=cache))
    assert GS.STATS["fused"] == 1

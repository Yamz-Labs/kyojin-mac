"""MoE gather kernels (kyojin_mlx.moe) vs the per-expert Mul1Gemv path (bit-exact stages) and vs numpy fp64 (reconstruct math).
Synthetic ragged stacks (random valid trellises, mixed K, both codebooks). Run on the Mac. Real stacked layers: set KYOJIN_REAL_MOE
to `<dir>:<prefix>:<proj names>`, see test_real_layer."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import kernels as KN, moe as M, ref_numpy as R

E, TOPK = 12, 6
KS = [2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3]


def make_stack(kin, nout, ks, cb, seed):
    """Random valid ragged stack + the per-expert numpy parts."""
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    parts, words, off = [], [], [0]
    for e, K in enumerate(ks):
        tr = rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)
        suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
        svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
        parts.append((tr, suh, svh))
        w = np.ascontiguousarray(tr).view("<u4").reshape(-1)
        words.append(w); off.append(off[-1] + len(w))
    st = M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")), mx.array(np.array(ks, "u1")),
                 mx.array(np.stack([p[1] for p in parts])), mx.array(np.stack([p[2] for p in parts])), kt, nt, cb)
    return st, parts


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384                                    # 2 and 3 blocks of 128: gate/up [256 -> 384], down [384 -> 256]
    g, gp = make_stack(kin, ninter, KS, cb, 1)
    u, up = make_stack(kin, ninter, KS, cb, 2)
    d, dp = make_stack(ninter, kin, KS, cb, 3)
    return cb, (g, u, d), (gp, up, dp)


def _route(rows, seed, spread=True):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
    return mx.array(ids), mx.array(w)


def _x(rows, kin, seed):
    return mx.array(np.random.default_rng(seed).standard_normal((rows, kin)).astype(np.float16) * np.float16(2.0))


def _bits(a):
    return np.array(a).view(np.uint32)


def _unfused_weights(m: M.MoeGather, stack, name, e, tgk):
    return KN.Mul1Gemv(stack.expert_trellis(e), np.array(stack.suh[e]), np.array(stack.svh[e]), rows=1, out_dtype=mx.float32,
                       tgk=tgk, cb=stack.cb)


@pytest.mark.parametrize("rows", [1, 3])
def test_stage_rot(layer, rows):
    cb, (g, u, d), _ = layer
    m = M.MoeGather(g, u, d, TOPK)
    ids, w = _route(rows, 5)
    x = _x(rows, g.kin, 6)
    xh = np.array(m.rot_gu(x, ids.reshape(-1)))
    ids_h = np.array(ids)
    for r in range(rows):
        for j in range(TOPK):
            e = int(ids_h[r, j])
            for pj, st in enumerate((g, u)):
                want = np.array(KN.rot(x[r:r + 1], st.suh[e]))
                assert np.array_equal(xh[(r * TOPK + j) * 2 + pj][None].view(np.uint32), want.view(np.uint32)), (r, j, pj)


@pytest.mark.parametrize("tgk", [1, 2])
@pytest.mark.parametrize("rows", [1, 2])
def test_stage_gemv_partials(layer, rows, tgk):
    """Gather GEMV partials are bit-identical to Mul1Gemv.partial of the same expert (same arithmetic and order)."""
    cb, (g, u, d), _ = layer
    m = M.MoeGather(g, u, d, TOPK, tgk_gu=tgk, tgk_dn=tgk)
    ids, w = _route(rows, 7)
    flat = ids.reshape(-1)
    x = _x(rows, g.kin, 8)
    xh = m.rot_gu(x, flat)
    part = np.array(m.gemv_gu(xh, flat, rows))                  # [npairs*2, tgk, n]
    ids_h = np.array(flat).tolist()
    for p, e in enumerate(ids_h):
        for pj, st in enumerate((g, u)):
            ref = _unfused_weights(m, st, "x", e, tgk)
            want = np.array(ref.partial(xh[p * 2 + pj][None]))[:, 0, :]
            assert np.array_equal(part[p * 2 + pj].view(np.uint32), want.view(np.uint32)), (p, pj, e)
    xh2 = mx.array(np.random.default_rng(9).standard_normal((rows * TOPK, d.kin)).astype(np.float32))
    part2 = np.array(m.gemv_dn(xh2, flat, rows))
    for p, e in enumerate(ids_h):
        ref = _unfused_weights(m, d, "d", e, tgk)
        want = np.array(ref.partial(xh2[p][None]))[:, 0, :]
        assert np.array_equal(part2[p].view(np.uint32), want.view(np.uint32)), (p, e)


@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_end_to_end_vs_unfused(layer, rows, wpre):
    """Fused five-launch pipeline vs the per-expert path: bit-identical fp32 output (all stages share arithmetic and order)."""
    cb, (g, u, d), _ = layer
    for tgk in (1, 2):
        f = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=tgk, tgk_dn=tgk)
        r = M.MoeUnfused(g, u, d, TOPK, wpre=wpre, tgk_gu=tgk, tgk_dn=tgk)
        ids, w = _route(rows, 11 + rows)
        x = _x(rows, g.kin, 12 + rows)
        got, ref = np.array(f(x, ids, w)), np.array(r(x, ids, w))
        diff = float(np.abs(got - ref).max() / np.abs(ref).max())
        ulp_bad = int((got.view(np.uint32) != ref.view(np.uint32)).sum())
        assert diff < 1e-6, (cb, rows, wpre, tgk, diff, ulp_bad)
        if ulp_bad:
            print(f"NOT bit-exact: {cb} rows={rows} wpre={wpre} tgk={tgk}: {ulp_bad}/{got.size} words differ, max rel {diff:.2e}")
        assert ulp_bad == 0, (cb, rows, wpre, tgk, ulp_bad, diff)


def _np_moe(x, ids, w, parts, limit, wpre, cb, alpha=1.0, beta=0.0):
    gp, up, dp = parts
    out = np.zeros((x.shape[0], dp[0][2].shape[0]), np.float64)
    for r in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[r, j])
            xr = x[r:r + 1].astype(np.float64)
            gt = R.gemv(xr, *gp[e], dtype=np.float64, cb=cb)
            ut = R.gemv(xr, *up[e], dtype=np.float64, cb=cb)
            gt = np.minimum(gt, limit); ut = np.clip(ut, -limit, limit)
            a = gt / (1 + np.exp(-alpha * gt)) * (ut + beta)
            if wpre:
                a = a * w[r, j]
            y = R.gemv(a, *dp[e], dtype=np.float64, cb=cb)
            out[r] += y[0] if wpre else y[0] * w[r, j]
    return out


@pytest.mark.parametrize("rows", [1, 4])
def test_vs_numpy_fp64(layer, rows):
    """Against fp64 reconstruct math (the codebook values are exact fp16, so only fp32 accumulation differs)."""
    cb, (g, u, d), parts = layer
    ids, w = _route(rows, 21)
    x = _x(rows, g.kin, 22)
    for wpre in (False, True):
        f = M.MoeGather(g, u, d, TOPK, wpre=wpre)
        got = np.array(f(x, ids, w)).astype(np.float64)
        ref = _np_moe(np.array(x), np.array(ids), np.array(w), parts, 10.0, wpre, cb)
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
        assert rel < 5e-5 and cos > 1 - 1e-9, (cb, rows, wpre, rel, cos)


def test_clamp_active(layer):
    """The +-10 clamp must actually bite in the test data (otherwise a missing clamp would pass)."""
    cb, (g, u, d), parts = layer
    ids, w = _route(1, 31)
    x = mx.array(np.array(_x(1, g.kin, 32)) * np.float16(8.0))
    f = M.MoeGather(g, u, d, TOPK)
    r = M.MoeUnfused(g, u, d, TOPK)
    got, ref = np.array(f(x, ids, w)), np.array(r(x, ids, w))
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    e = int(np.array(ids)[0, 0])
    gt = np.array(KN.Mul1Gemv(g.expert_trellis(e), np.array(g.suh[e]), np.array(g.svh[e]), out_dtype=mx.float32, cb=cb)(x))
    assert (np.abs(gt) > 10).any(), "clamp not exercised"


def test_fq_matches_mlx_vlm():
    """The in-kernel fp8 fake quant is bit-identical to mlx-vlm's fake_quant_fp8_ue8m0 kernel."""
    fqm = pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    rng = np.random.default_rng(3)
    for scale in (0.03, 1.0, 40.0):
        a = (rng.standard_normal((6, 2304)) * scale).astype(np.float16)
        src = r"""
            const uint lane = thread_index_in_simdgroup;
            const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
            const uint r = threadgroup_position_in_grid.y;
            if (blk * 128 >= N) return;
            const ulong col = blk * 128 + lane * 4;
            float4 a4 = float4(*((const device half4*)(x + ulong(r) * N + col)));
        """ + __import__("kyojin_mlx.metal_src", fromlist=["x"]).FQ_FP8 + "\n*((device float4*)(y + ulong(r) * N + col)) = a4;"
        kern = mx.fast.metal_kernel(name="fqtest", input_names=["x"], output_names=["y"], source=src)
        grid, tg = ((2304 // 128 + 3) // 4 * 128, 6, 1), (128, 1, 1)
        got = np.array(kern(inputs=[mx.array(a)], template=[("N", 2304)], grid=grid, threadgroup=tg, output_shapes=[a.shape],
                            output_dtypes=[mx.float32])[0])
        want = np.array(fqm.fake_quant_fp8_ue8m0(mx.array(a))).astype(np.float32)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (scale, int((got != want).sum()))


@pytest.mark.parametrize("rows", [1, 4])
def test_fused_fq_vs_unfused(layer, rows):
    pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    cb, (g, u, d), _ = layer
    f = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True, tgk_gu=1, tgk_dn=1)
    r = M.MoeUnfused(g, u, d, TOPK, wpre=True, fq=True, tgk_gu=1, tgk_dn=1)
    ids, w = _route(rows, 41)
    x = _x(rows, g.kin, 42)
    got, ref = np.array(f(x, ids, w)), np.array(r(x, ids, w))
    bad = int((got.view(np.uint32) != ref.view(np.uint32)).sum())
    assert float(np.abs(got - ref).max() / np.abs(ref).max()) < 1e-3
    assert bad == 0, f"{bad}/{got.size} words differ (act ulp -> fp8 boundary flips)"


@pytest.mark.parametrize("rows", [1, 3])
def test_swigluoai(layer, rows):
    """MiniMax-M3 style activation g * sigmoid(1.702 g) * (u + 1), limit 7: fused vs per-expert bit-exact, and vs fp64."""
    cb, (g, u, d), parts = layer
    ids, w = _route(rows, 51)
    x = _x(rows, g.kin, 52)
    for wpre in (False, True):
        f = M.MoeGather(g, u, d, TOPK, limit=7.0, wpre=wpre, alpha=1.702, beta=1.0)
        r = M.MoeUnfused(g, u, d, TOPK, limit=7.0, wpre=wpre, alpha=1.702, beta=1.0)
        got, ref = np.array(f(x, ids, w)), np.array(r(x, ids, w))
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (cb, rows, wpre)
        want = _np_moe(np.array(x), np.array(ids), np.array(w), parts, 7.0, wpre, cb, 1.702, 1.0)
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        assert rel < 5e-5, (cb, rows, wpre, rel)


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("topk,ks", [(1, [4] * 12), (4, [3, 4, 5, 6, 4, 3, 5, 6, 4, 4, 3, 6]), (8, [2, 5, 3, 4, 2, 2, 3, 5, 4, 3, 2, 6])])
def test_any_topk_any_k(cb, topk, ks):
    """The gather path is arch-agnostic: any top-k, any mix of K (1-8), shared tables only through offset / k."""
    g, _ = make_stack(256, 384, ks, cb, 61)
    u, _ = make_stack(256, 384, ks, cb, 62)
    d, _ = make_stack(384, 256, ks, cb, 63)
    rng = np.random.default_rng(64)
    for rows in (1, 3):
        ids = mx.array(np.stack([rng.choice(len(ks), topk, replace=False) for _ in range(rows)]).astype(np.int32))
        w = mx.array(rng.uniform(0.1, 1.0, (rows, topk)).astype(np.float32))
        x = _x(rows, 256, 65)
        f = M.MoeGather(g, u, d, topk, tgk_gu=1, tgk_dn=1)
        r = M.MoeUnfused(g, u, d, topk, tgk_gu=1, tgk_dn=1)
        got, ref = np.array(f(x, ids, w)), np.array(r(x, ids, w))
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (cb, topk, rows)

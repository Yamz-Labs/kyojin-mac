"""Grouped MoE decode with the NAX GEMM for the groups of hotk or more rows (rounding-class): the layer output stays within fp16-level error of the
per-row gather path, rows of groups below hotk are untouched, and the plan lays the hot rows out group after group. Needs an M5 (Metal 4 tensor ops)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import metal_nax as NX, metal_src as S, moe as M
if not NX.nax_available():
    pytest.skip("no neural accelerators on this GPU", allow_module_level=True)
from test_moe import E, KS, TOPK, make_stack, _x
from test_moe_union import route
from test_moe_grouped import hot


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return g, u, d


def rel(a, b):
    a, b = np.array(a, np.float64), np.array(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("rows", [2, 4, 5, 8, 16, 17, 32, 42])
@pytest.mark.parametrize("mode", ["identical", "partial", "random", "hot"])
@pytest.mark.parametrize("hotk", [2, 4, 6])
@pytest.mark.parametrize("wpre", [False, True])
def test_nax_groups_stay_close(layer, rows, mode, hotk, wpre):
    g, u, d = layer
    plain = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=False, hotk=0)
    nax = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=True, hotk=hotk)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 10 * rows + len(mode))
    x = _x(rows, g.kin, 7)
    assert rel(nax(x, ids, w), plain(x, ids, w)) < 3e-3


@pytest.mark.parametrize("mode", ["identical", "partial", "random", "hot"])
def test_plan_hot_layout(mode):
    rows, hk = 12, 4
    ids, _ = hot(rows, 3) if mode == "hot" else route(rows, mode, 5)
    fl = ids.reshape(-1).astype(mx.int32)
    g, _ = make_stack(256, 384, KS, "mul1", 1)
    u, _ = make_stack(256, 384, KS, "mul1", 2)
    d, _ = make_stack(384, 256, KS, "mul1", 3)
    mo = M.MoeGather(g, u, d, TOPK, grouped=True, hotk=hk)
    lead, cnt, mem, hrow, heids, roff_gu, roff_d, hinfo = (np.array(a) for a in mo.plan(fl, rows))
    ids_h, npairs = np.array(fl), rows * TOPK
    counts = {int(e): int((ids_h == e).sum()) for e in set(ids_h.tolist())}
    hot_e = sorted([e for e, c in counts.items() if c >= hk], key=lambda e: int(np.nonzero(ids_h == e)[0][0]))      # by lowest pair
    rh = sum(counts[e] for e in hot_e)
    assert int(hinfo[0]) == rh
    assert list(heids[:len(hot_e)]) == hot_e
    assert list(roff_d[:len(hot_e) + 1]) == list(np.concatenate([[0], np.cumsum([counts[e] for e in hot_e])]))
    gh = S.hot_cap(npairs, hk)
    assert roff_gu[2 * gh] == 2 * rh and roff_d[gh] == rh and np.all(np.diff(roff_gu.astype(np.int64)) >= 0) and np.all(np.diff(roff_d.astype(np.int64)) >= 0)
    for q in range(npairs):
        e = int(ids_h[q])
        if counts[e] >= hk:
            l = hot_e.index(e)
            members = [p for p in range(npairs) if ids_h[p] == e]
            assert hrow[q] == roff_d[l] + members.index(q)
        else:
            assert hrow[q] == 0xffffffff
    listed = {int(ids_h[p]) for p in lead if p != 0xffffffff}
    assert listed == {e for e, c in counts.items() if c < hk}


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
@pytest.mark.parametrize("mode", ["identical", "partial", "hot"])
def test_hot_split_k_bit_identical(layer, rows, mode, monkeypatch):
    """Per-bitrate NAX passes of the hot slots (KYOJIN_VB_SPLIT_K) give the bits of the one mixed-K launch."""
    g, u, d = layer
    nax = M.MoeGather(g, u, d, TOPK, tgk_gu=2, tgk_dn=2, grouped=True, hotk=4)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 3 * rows + len(mode))
    x = _x(rows, g.kin, 11)
    monkeypatch.setattr(M, "VB_SPLIT_K", False)
    one = nax(x, ids, w)
    mx.eval(one)
    monkeypatch.setattr(M, "VB_SPLIT_K", True)
    split = nax(x, ids, w)
    assert len(nax.ks_gu) > 1 and bool(mx.array_equal(one, split).item())


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_expert_gemm_prev_carries_tiles(layer, rows):
    g = layer[0]
    G = g.experts
    rng = np.random.default_rng(rows)
    A = mx.array(NX.permute_k(rng.standard_normal((G * rows, g.kin)).astype(np.float16)))
    eids, ro = mx.arange(G, dtype=mx.int32), mx.array((np.arange(G + 1) * rows).astype(np.uint32))
    cfg = NX._BLOCK_CFG[16]
    one = NX.expert_gemm((g,), eids, ro, A, G * rows, rows, cfg, out_half=False)
    y = None
    for k in sorted(set(KS)):
        y = NX.expert_gemm((g,), eids, ro, A, G * rows, rows, cfg, out_half=False, ks=(k,), prev=y)
    assert bool(mx.array_equal(one, y).item())


@pytest.mark.parametrize("rows", [2, 3, 4, 8, 16, 42])
@pytest.mark.parametrize("mode", ["identical", "partial", "random", "hot"])
@pytest.mark.parametrize("wpre", [False, True])
def test_grouped_split_k_bit_identical(layer, rows, mode, wpre, monkeypatch):
    """Per-bitrate grouped GEMV passes (KYOJIN_VB_SPLIT_K, default MTP verify path) give the bits of the one mixed-K launch and of the per-row path."""
    g, u, d = layer
    grp = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    plain = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=False, hotk=0)
    ids, w = hot(rows, rows) if mode == "hot" else route(rows, mode, 7 * rows + len(mode))
    x = _x(rows, g.kin, 5)
    monkeypatch.setattr(M, "VB_SPLIT_K", False)
    one = grp(x, ids, w)
    mx.eval(one)
    monkeypatch.setattr(M, "VB_SPLIT_K", True)
    split = grp(x, ids, w)
    assert bool(mx.array_equal(one, split).item()) and bool(mx.array_equal(split, plain(x, ids, w)).item())


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_affine_fold_close(layer, rows):
    """KYOJIN_MOE_AFFOLD (cb mul1af: r * x and sum(x) accumulated, y = A * acc + B * sum(x)) stays within fp16-decode rounding of the default
    on the gather (rows 1) and grouped (rows >= 2) GEMVs; solo and grouped agree with each other per row."""
    g, u, d = layer
    if g.cb != "mul1":
        pytest.skip("the fold is for the mul1 codebook")
    ref = M.MoeGather(g, u, d, TOPK, wpre=True, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    af = M.MoeGather(g, u, d, TOPK, wpre=True, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
    af.cbg = "mul1af"
    ref.fastdec = af.fastdec = False                                             # KYOJIN_MOE_FASTDEC would give both the mul1fd gather
    ids, w = route(rows, "random", 11 * rows)
    x = _x(rows, g.kin, 9)
    y0, y1 = ref(x, ids, w).astype(mx.float32), af(x, ids, w).astype(mx.float32)
    rel = (mx.abs(y1 - y0).max() / mx.abs(y0).max()).item()
    assert 0 < rel < 2e-2
    if rows > 1:
        solo = mx.concatenate([af(x[i:i + 1], ids[i:i + 1], w[i:i + 1]) for i in range(rows)], 0).astype(mx.float32)
        assert bool(mx.array_equal(solo, y1).item())

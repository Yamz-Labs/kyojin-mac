"""S3 MoE prefill (kyojin_mlx.moe_prefill) vs the decode path (MoeGather, 1-4 rows at a time) on synthetic ragged stacks.
f32 mode (fp32 activations and weights, same plan) must agree to fp32 rounding: it proves sorting, groups, indices, scales and the fake quant.
f16 mode (production) differs by fp16 rounding of the activations and of the matmul output only."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import kernels as KN, moe as M, moe_prefill as MP
from test_moe import make_stack, E, TOPK, KS


@pytest.fixture(autouse=True)
def stock_decode(monkeypatch):
    """The decode reference is the stock gather: KYOJIN_MOE_FASTDEC (mul1fd, not bit-identical, KL-gated in test_moe_fastdec) stays off here."""
    monkeypatch.setattr(M, "FASTDEC", False)


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    g, _ = make_stack(256, 384, KS, cb, 1)
    u, _ = make_stack(256, 384, KS, cb, 2)
    d, _ = make_stack(384, 256, KS, cb, 3)
    return cb, g, u, d


def route(rows, seed, topk=TOPK, e=E, pool=None):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(pool or e, topk, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, topk)).astype(np.float32)
    return mx.array(ids), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def xs(rows, kin, seed, scale=2.0):
    return mx.array(np.random.default_rng(seed).standard_normal((rows, kin)).astype(np.float16) * np.float16(scale))


def decode_path(moe, x, ids, w, step=4):
    return np.concatenate([np.array(moe(x[i:i + step], ids[i:i + step], w[i:i + step])) for i in range(0, x.shape[0], step)])


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_decode_group_bit_exact(layer):
    cb, g, u, d = layer
    eids = mx.array(np.array([7, 0, 3, 11, 5], np.int32))
    w = np.array(MP.decode_group((g, u), eids, cb))
    for pj, st in enumerate((g, u)):
        for i, e in enumerate(np.array(eids).tolist()):
            want = np.array(KN.decode_wr(st.expert_trellis(e).view(mx.uint32).reshape(-1), int(st.k[e].item()), st.kt, st.nt, cb))
            assert np.array_equal(w[pj * 5 + i].view(np.uint16), want.view(np.uint16)), (pj, e)
    wd = np.array(MP.decode_group((d,), eids, cb))
    for i, e in enumerate(np.array(eids).tolist()):
        want = np.array(KN.decode_wr(d.expert_trellis(e).view(mx.uint32).reshape(-1), int(d.k[e].item()), d.kt, d.nt, cb))
        assert np.array_equal(wd[i].view(np.uint16), want.view(np.uint16)), e


@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("rows", [9, 70])
def test_f32_plan_matches_decode(layer, rows, wpre):
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=wpre)
    pf = MP.MoePrefill(dec, prec="f32", group_gib=1e-4)                  # tiny budget: one expert per group
    ids, w = route(rows, 70 + rows)
    x = xs(rows, g.kin, 71)
    got, ref = np.array(pf(x, ids, w)), decode_path(dec, x, ids, w)
    assert pf.last["groups"] == pf.last["active"]
    assert rel(got, ref) < 3e-6, (cb, rows, wpre, rel(got, ref))


@pytest.mark.parametrize("rows", [9, 70])
def test_f16_error_vs_decode(layer, rows):
    cb, g, u, d = layer
    for wpre in (False, True):
        dec = M.MoeGather(g, u, d, TOPK, wpre=wpre)
        ids, w = route(rows, 80 + rows)
        x = xs(rows, g.kin, 81)
        got, ref = np.array(MP.MoePrefill(dec)(x, ids, w)), decode_path(dec, x, ids, w)
        per_row = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(f"{cb} rows={rows} wpre={wpre}: rel L2 {rel(got, ref):.2e}, per-row max {per_row.max():.2e}")
        assert rel(got, ref) < 3e-3 and per_row.max() < 1e-2, (cb, rows, wpre, rel(got, ref), per_row.max())


def test_groups_do_not_change_result(layer):
    """Cutting the active experts into groups of any size gives bit-identical output (each pair only sees its own expert)."""
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, w = route(50, 91)
    x = xs(50, g.kin, 92)
    ref = np.array(MP.MoePrefill(dec, group_gib=64, fused=False)(x, ids, w))
    for gib in (1e-4, 0.0012, 0.0024):
        pf = MP.MoePrefill(dec, group_gib=gib, fused=False)
        got = np.array(pf(x, ids, w))
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (gib, pf.last)
    assert pf.last["groups"] > 1


def test_unused_experts_skipped(layer):
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, w = route(30, 93, pool=8)                                       # experts 0..7 only
    x = xs(30, g.kin, 94)
    pf = MP.MoePrefill(dec, prec="f32")
    got = np.array(pf(x, ids, w))
    assert pf.last["active"] <= 8
    assert rel(got, decode_path(dec, x, ids, w)) < 3e-6


def test_chunking(layer):
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, w = route(45, 95)
    x = xs(45, g.kin, 96)
    a = np.array(MP.MoePrefill(dec, chunk=45)(x, ids, w))
    b = np.array(MP.MoePrefill(dec, chunk=16)(x, ids, w))
    assert rel(b, a) < 3e-3 and a.shape == b.shape == (45, g.kin)


def test_fq_f32_close_f16_bounded(layer):
    """DS4.1 path (routing weight before down, fp8 fake quant). f32 plan: same flips as the decode path up to rare boundary cases;
    f16: fp8 bin flips from the fp16 rounding, bounded."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True)
    ids, w = route(40, 97)
    x = xs(40, g.kin, 98)
    ref = decode_path(dec, x, ids, w)
    e32 = rel(np.array(MP.MoePrefill(dec, prec="f32")(x, ids, w)), ref)
    e16 = rel(np.array(MP.MoePrefill(dec)(x, ids, w)), ref)
    print(f"{cb} fq: f32 {e32:.2e}  f16 {e16:.2e}")
    assert e32 < 5e-4 and e16 < 2e-2, (e32, e16)


@pytest.mark.parametrize("topk,ks", [(1, [4] * 12), (4, [3, 4, 5, 6, 4, 3, 5, 6, 4, 4, 3, 6]), (8, [2, 5, 3, 4, 2, 2, 3, 5, 4, 3, 2, 6])])
def test_any_topk_any_k(layer, topk, ks):
    cb = layer[0]
    g, _ = make_stack(256, 384, ks, cb, 61)
    u, _ = make_stack(256, 384, ks, cb, 62)
    d, _ = make_stack(384, 256, ks, cb, 63)
    dec = M.MoeGather(g, u, d, topk)
    ids, w = route(33, 99, topk=topk)
    x = xs(33, 256, 100)
    got = np.array(MP.MoePrefill(dec, prec="f32")(x, ids, w))
    assert rel(got, decode_path(dec, x, ids, w)) < 3e-6


@pytest.mark.parametrize("rows", [9, 70, 200])
def test_nax_matches_s3(layer, rows):
    """Fused trellis-in-registers NAX path (nax= alias of fused=) vs S3 (decode to fp16 + gather_mm): same arithmetic up to the fp32 accumulation order."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no neural accelerators on this GPU")
    cb, g, u, d = layer
    for wpre in (False, True):
        dec = M.MoeGather(g, u, d, TOPK, wpre=wpre)
        ids, w = route(rows, 120 + rows)
        x = xs(rows, g.kin, 121)
        pf = MP.MoePrefill(dec, nax=True, fused_max=10 ** 9)
        got, ref = np.array(pf(x, ids, w)), np.array(MP.MoePrefill(dec, fused=False)(x, ids, w))
        assert pf.nax and pf.last["fused_experts"] == pf.last["active"]
        assert rel(got, ref) < 2e-3, (cb, rows, wpre, rel(got, ref))
        assert rel(got, decode_path(dec, x, ids, w)) < 3e-3


def test_nax_skewed_routing(layer):
    """Few experts take most rows (long per-expert row blocks) and some experts are never used."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no neural accelerators on this GPU")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, w = route(150, 130, pool=7)
    x = xs(150, g.kin, 131)
    got = np.array(MP.MoePrefill(dec, nax=True, fused_max=10 ** 9)(x, ids, w))
    ref = np.array(MP.MoePrefill(dec, fused=False)(x, ids, w))
    assert rel(got, ref) < 2e-3


@pytest.mark.parametrize("rows", [9, 200])
def test_nax_split_gu_bit_identical(layer, rows, monkeypatch):
    """Gate and up as two launches (SPLIT_GU, the default) vs one combined launch: disjoint outputs, same per-row arithmetic."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no neural accelerators on this GPU")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True)
    ids, w = route(rows, 150 + rows)
    x = xs(rows, g.kin, 151)
    pf = MP.MoePrefill(dec, nax=True, fused_max=10 ** 9)
    monkeypatch.setattr(MP, "SPLIT_GU", True)
    got = np.array(pf(x, ids, w))
    monkeypatch.setattr(MP, "SPLIT_GU", False)
    ref = np.array(pf(x, ids, w))
    assert np.array_equal(got, ref), (cb, rows)


@pytest.mark.parametrize("rows", [9, 200])
def test_nax_split_k_bit_identical(layer, rows, monkeypatch):
    """One fused pass per bitrate class (PF_SPLIT_K, the default) vs one launch for all bitrates: same tiles, same per-row arithmetic."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no neural accelerators on this GPU")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True)
    ids, w = route(rows, 140 + rows)
    x = xs(rows, g.kin, 141)
    pf = MP.MoePrefill(dec, nax=True, fused_max=10 ** 9)
    monkeypatch.setattr(MP, "PF_SPLIT_K", True)
    got = np.array(pf(x, ids, w))
    monkeypatch.setattr(MP, "PF_SPLIT_K", False)
    ref = np.array(pf(x, ids, w))
    assert np.array_equal(got, ref), (cb, rows)


@pytest.mark.parametrize("rows", [300, 1500])
@pytest.mark.parametrize("glu", [False, True])
def test_nax_cap_rows_bit_identical(layer, rows, glu, monkeypatch):
    """Pair buffers rounded up to a few sizes per octave (PF_CAP, the default) vs exact sizes: the kernels only see the real pair count."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no neural accelerators on this GPU")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True)
    ids, w = route(rows, 160 + rows)
    x = xs(rows, g.kin, 161)
    pf = MP.MoePrefill(dec, nax=True, fused_max=10 ** 9)
    monkeypatch.setattr(MP, "GLU", glu)
    monkeypatch.setattr(MP, "CAP_STEPS", 4)
    got = np.array(pf(x, ids, w))
    monkeypatch.setattr(MP, "CAP_STEPS", 0)
    ref = np.array(pf(x, ids, w))
    assert np.array_equal(got, ref), (cb, rows, glu)
    monkeypatch.setattr(MP, "CAP_STEPS", 4)
    assert MP.cap_rows(1000) == 1024 and MP.cap_rows(1025) == 1280 and MP.cap_rows(200) == 200

"""S6 fused EXL3 trellis decode + M5 neural-accelerator GEMM (kyojin_mlx.metal_nax) vs the decode_pf + fp32 reference, vs the S3 group path
(MoePrefill fused=False) and vs the dense decode + matmul path (Exl3Linear). Needs a GPU with NAX (M5)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX, moe as M, moe_prefill as MP, patch as P
if not NX.nax_available():
    pytest.skip("no neural accelerators on this GPU", allow_module_level=True)
from test_moe import make_stack, TOPK, KS
from test_moe_prefill import route, xs, rel, decode_path


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    g, _ = make_stack(256, 384, KS, cb, 1)
    u, _ = make_stack(256, 384, KS, cb, 2)
    d, _ = make_stack(384, 256, KS, cb, 3)
    return cb, g, u, d


def test_permute_k_rotation_kernel():
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((7, 256)).astype(np.float16))
    suh = mx.array((rng.choice([-1.0, 1.0], 256) * rng.uniform(0.5, 1.5, 256)).astype(np.float16))
    for out in (mx.float16, mx.float32):
        a, b = np.array(KN.rot(x, suh, out)), np.array(KN.rot(x, suh, out, perm=True))
        assert np.array_equal(b, NX.permute_k(a)), out


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("cfg", [(16, 1, 32, 32), (32, 1, 32, 32), (64, 1, 32, 32), (16, 1, 32, 64), (32, 1, 32, 64), (32, 2, 32, 64)])
def test_expert_gemm_all_bitrates(cb, cfg):
    """Every bitrate 1..8 (one and two 64-bit window chunks), ragged row counts, vs fp32 matmul on the decoded fp16 weights."""
    ks = [1, 2, 3, 4, 5, 6, 7, 8]
    st, _ = make_stack(256, 384, ks, cb, 11)
    G = len(ks)
    eids = mx.array(np.arange(G, dtype=np.int32))
    counts = np.array([1, 5, 17, 64, 70, 130, 3, 40])
    rowoff = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    rng = np.random.default_rng(12)
    A = rng.standard_normal((int(counts.sum()), 256)).astype(np.float16)
    w = np.array(MP.decode_group((st,), eids, cb)).astype(np.float32)
    y = np.array(NX.expert_gemm((st,), eids, mx.array(rowoff), mx.array(NX.permute_k(A)), A.shape[0], int(counts.max()),
                                cfg)).astype(np.float32)
    for e in range(G):
        a, b = rowoff[e], rowoff[e + 1]
        ref = A[a:b].astype(np.float32) @ w[e].T
        assert rel(y[a:b], ref) < 1.5e-3, (cb, cfg, ks[e], rel(y[a:b], ref))


def test_expert_gemm_gate_up_one_launch(layer):
    cb, g, u, d = layer
    sel = np.array([9, 2, 5, 0, 7], np.int32)
    counts = np.array([20, 3, 64, 9, 33])
    n = int(counts.sum())
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    rng = np.random.default_rng(13)
    A = rng.standard_normal((2 * n, 256)).astype(np.float16)
    wgu = np.array(MP.decode_group((g, u), mx.array(sel), cb)).astype(np.float32)
    y = np.array(NX.expert_gemm((g, u), mx.array(sel), mx.array(np.concatenate([cl[:-1], n + cl])), mx.array(NX.permute_k(A)), 2 * n,
                                int(counts.max()), NX.pick_cfg(counts))).astype(np.float32)
    for pj in range(2):
        for i in range(len(sel)):
            a, b = pj * n + cl[i], pj * n + cl[i + 1]
            ref = A[a:b].astype(np.float32) @ wgu[pj * len(sel) + i].T
            assert rel(y[a:b], ref) < 1.5e-3, (cb, pj, i, rel(y[a:b], ref))


@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("rows,fused_max", [(9, 192), (70, 192), (70, "mid"), (70, 0), (300, "mid")])
def test_moe_prefill_fused_vs_group_path(layer, rows, fused_max, wpre):
    """Same plan, same activations: only the matmul differs (fused NAX vs decode_pf + gather_mm). fused_max 'mid' = hybrid (hot experts on
    the group path), 0 = nothing fused. Both are within fp16 rounding of the decode path."""
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=wpre)
    ids, w = route(rows, 300 + rows)
    x = xs(rows, g.kin, 301)
    ref = decode_path(dec, x, ids, w)
    cnt = np.bincount(np.array(ids).reshape(-1), minlength=12)
    if fused_max == "mid":                                                # about half of the experts are hot (group path)
        fused_max = int(np.median(cnt[cnt > 0]))
    pf = MP.MoePrefill(dec, fused=True, fused_max=fused_max)
    got = np.array(pf(x, ids, w))
    base = np.array(MP.MoePrefill(dec, fused=False)(x, ids, w))
    print(f"{cb} rows={rows} fused_max={fused_max} wpre={wpre}: fused vs decode {rel(got, ref):.2e}, group vs decode {rel(base, ref):.2e}, "
          f"fused vs group {rel(got, base):.2e} {pf.last}")
    assert rel(got, ref) < 3e-3 and rel(got, base) < 2e-3, (cb, rows, fused_max, wpre)
    if fused_max == 0:
        assert pf.last["fused_experts"] == 0 and np.array_equal(got.view(np.uint32), base.view(np.uint32))
    else:
        assert 0 < pf.last["fused_experts"] <= pf.last["active"]
        assert (pf.last["groups"] > 0) == (fused_max < cnt.max())


def test_moe_prefill_fused_fq(layer):
    """DS4.1 path (routing weight before down, fp8 fake quant): fp16 rounding can flip fp8 bins, bounded like the group path."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=True)
    ids, w = route(40, 97)
    x = xs(40, g.kin, 98)
    ref = decode_path(dec, x, ids, w)
    ef = rel(np.array(MP.MoePrefill(dec, fused=True)(x, ids, w)), ref)
    eg = rel(np.array(MP.MoePrefill(dec, fused=False)(x, ids, w)), ref)
    print(f"{cb} fq: fused {ef:.2e}  group {eg:.2e}")
    assert ef < 2e-2


@pytest.mark.parametrize("rows", [5, 16, 100])
def test_dense_linear_fused_vs_decode_matmul(rows, monkeypatch):
    rng = np.random.default_rng(21)
    kt, nt, K = 16, 24, 3
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    suh = mx.array((rng.choice([-1.0, 1.0], kt * 16) * rng.uniform(0.5, 1.5, kt * 16)).astype(np.float16))
    svh = mx.array((rng.choice([-1.0, 1.0], nt * 16) * rng.uniform(0.5, 1.5, nt * 16)).astype(np.float16))
    lin = EL.Exl3Linear(tr, suh, svh, cb="mul1")
    x = xs(rows, kt * 16, 22)
    monkeypatch.setattr(EL, "DENSE_FUSED_MAX", 0)
    ref = np.array(lin(x)).astype(np.float32)
    monkeypatch.setattr(EL, "DENSE_FUSED_MAX", 128)
    got = np.array(lin(x)).astype(np.float32)
    assert rel(got, ref) < 1.5e-3, rel(got, ref)


def test_s3_gate_follows_the_fused_path(layer, monkeypatch):
    """With the fused path S3 beats the 4-row decode chunks from about 2.5 rows per expert (measured on the Air: DS4.1 ~160 tokens, GLM ~80),
    without it from 14 rows per expert (the group path)."""
    cb, g, u, d = layer
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 0)                     # rows.enable (other tests) raises it: the multi-user band keeps the decode path
    monkeypatch.setattr(P, "PF_CROSS_NAX", 2.5)                     # the Air value (the Ultra default is 1.75)
    dec = M.MoeGather(g, u, d, TOPK)
    assert P.pf_min_rows(dec) == int(P.PF_CROSS_NAX * g.experts / TOPK) == 5
    ids, w = route(6, 401)
    x = xs(6, g.kin, 402)
    got = np.array(P._moe_rows(dec, x, ids, w))                     # 6 rows >= 5: S3
    assert dec._pf is not None and rel(got, decode_path(dec, x, ids, w)) < 3e-3
    monkeypatch.setattr(MP, "FUSED", "0")
    assert P.pf_min_rows(dec) == int(P.PF_CROSS * g.experts / TOPK) == 28


def test_pick_cfg_follows_rows_per_expert(monkeypatch):
    monkeypatch.setattr(NX, "_BLOCK_COST", {16: 0.78, 32: 1.01, 64: 1.52})        # the Air tables this test was written for
    monkeypatch.setattr(NX, "_BLOCK_CFG", {16: (16, 1, 32, 64), 32: (32, 1, 32, 64), 64: (32, 2, 32, 64)})
    assert NX.pick_cfg([1] * 8) == (16, 1, 32, 64)
    assert NX.pick_cfg([16, 9, 3]) == (16, 1, 32, 64)
    assert NX.pick_cfg([32] * 4) == (32, 1, 32, 64)
    assert NX.pick_cfg([64] * 4) == (32, 2, 32, 64)
    assert NX.pick_cfg([300, 1, 2]) == (32, 2, 32, 64)


def test_expert_gemm_k_step_falls_back_when_kt_is_not_a_multiple_of_four():
    """kt = 18 (k = 288): a 64-wide k step would read past the trellis, `expert_gemm` runs the 32-wide step (same bits as asking for it)."""
    rng = np.random.default_rng(31)
    kt, nt, K = 18, 16, 3
    st = NX.SingleStack(mx.array(rng.integers(0, 2 ** 32, kt * nt * 8 * K, dtype=np.uint32)), K, kt, nt, "mul1")
    A = mx.array(rng.standard_normal((40, kt * 16)).astype(np.float16))
    eids, rowoff = mx.array([0], dtype=mx.int32), mx.array([0, 40], dtype=mx.uint32)
    a = np.array(NX.expert_gemm((st,), eids, rowoff, A, 40, 40, (32, 2, 32, 64))).astype(np.float32)
    b = np.array(NX.expert_gemm((st,), eids, rowoff, A, 40, 40, (32, 2, 32, 32))).astype(np.float32)
    assert np.array_equal(a, b) and np.isfinite(a).all()


def _dense_lin(kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb="mul1")


@pytest.mark.parametrize("kin,nout", [(512, 768), (256, 128)])
def test_rows_nax_in_the_split_band(kin, nout, monkeypatch):
    """ROWS_NAX: the rows above one GEMV chunk run the fused NAX kernel (small linears from the third chunk). Within fp16 rounding of the GEMV chunks,
    a row does not depend on the other rows (same bits at 6, 9 and 16 rows) and the default (off) is untouched."""
    lin = _dense_lin(kin, nout, 3, 41)
    x = xs(16, kin, 42)
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 16)
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    ref = {r: np.array(lin(x[:r])).astype(np.float32) for r in (6, 9, 16)}
    calls = []
    orig = EL.Exl3Linear._dense
    monkeypatch.setattr(EL.Exl3Linear, "_dense", lambda self, x2, *a, **k: calls.append(x2.shape[0]) or orig(self, x2, *a, **k))
    assert [lin(x[:r]) for r in (6, 9, 16)] and not calls              # off: GEMV chunks only
    monkeypatch.setattr(EL, "ROWS_NAX", True)
    got = {r: np.array(lin(x[:r])).astype(np.float32) for r in (6, 9, 16)}
    small = kin * nout < EL.NAX_SMALL_WEIGHTS
    assert calls == ([9, 16] if small else [6, 9, 16]), calls
    for r in (6, 9, 16):
        assert rel(got[r], ref[r]) < 1.5e-3, (r, rel(got[r], ref[r]))
    assert np.array_equal(got[6][:6], got[16][:6]) and np.array_equal(got[9][:6], got[16][:6]) or small
    solo = np.array(lin(x[:1])).astype(np.float32)                     # one row: GEMV, not NAX
    assert rel(got[16][:1], solo) < 1.5e-3


@pytest.mark.parametrize("cb,K", [("mul1", 5), ("mcg", 2)])
@pytest.mark.parametrize("users", [5, 9, 16])
def test_grouped_wo_a_nax(cb, K, users, monkeypatch):
    """wo_a (8 groups) with more than 4 users: one fused NAX launch for all groups against the GEMV chunks, within fp16 rounding; a user's
    row does not depend on the other users."""
    from test_grouped import make
    from kyojin_mlx.patch import Exl3MultiLinear
    ml = Exl3MultiLinear(*make(8, 512, 256, K, 70 + K), cb)
    x = xs(16 * 8, 512, 71).reshape(16, 8, 1, 512)
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 16)
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    ref = np.array(ml(x[:users])).astype(np.float32)
    monkeypatch.setattr(EL, "ROWS_NAX", True)
    calls = []
    orig = type(ml.grouped).nax
    monkeypatch.setattr(type(ml.grouped), "nax", lambda self, xt: calls.append(xt.shape[2]) or orig(self, xt))
    got = np.array(ml(x[:users])).astype(np.float32)
    assert calls == [users]
    assert rel(got, ref) < 1.5e-3, rel(got, ref)
    full = np.array(ml(x)).astype(np.float32)
    assert np.array_equal(full[:users], got)


@pytest.mark.parametrize("cb,K", [("mul1", 5), ("mcg", 2)])
@pytest.mark.parametrize("rows", [5, 70, 200])
def test_grouped_wo_a_prompt_rows(cb, K, rows, monkeypatch):
    """wo_a on prompt rows (batch 1, 5 .. DENSE_FUSED_MAX rows per group): one fused launch set for the 8 groups vs the 8 per-group dense calls (decode + matmul)."""
    from test_grouped import make
    from kyojin_mlx.patch import Exl3MultiLinear
    ml = Exl3MultiLinear(*make(8, 512, 256, K, 90 + K), cb)
    x = xs(8 * rows, 512, 91).reshape(1, 8, rows, 512)
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 0)                     # rows.enable (other tests) raises it: the multi-user band stays GEMV
    monkeypatch.setattr(EL, "DENSE_FUSED_MAX", 0)
    ref = np.array(ml(x)).astype(np.float32)
    monkeypatch.setattr(EL, "DENSE_FUSED_MAX", 1024)
    calls = []
    orig = type(ml.grouped).nax
    monkeypatch.setattr(type(ml.grouped), "nax", lambda self, xt: calls.append(xt.shape[2]) or orig(self, xt))
    got = np.array(ml(x)).astype(np.float32)
    assert calls == [rows]
    assert rel(got, ref) < 1.5e-3, rel(got, ref)


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("cfg", [(32, 2, 32, 64), (32, 2, 32, 32), (16, 2, 32, 64)])
def test_skip_tail_is_bit_identical(cb, cfg, monkeypatch):
    """A tail block whose rows all fit in the first sub-block skips the second matmul: same bits as running it on zero rows. Counts around
    the block edges (bmu, bmu + 1, bmu * nmb, bmu * nmb + 1, ...) and an expert with no row."""
    ks = [2, 3, 4, 5, 3, 3, 6, 4, 2, 3]
    st, _ = make_stack(256, 384, ks, cb, 21)
    G, bmu, rows = len(ks), cfg[0], cfg[0] * cfg[1]
    counts = np.array([bmu, bmu + 1, rows, rows + 1, rows + bmu, rows + bmu + 1, 1, 0, 2 * rows + 5, 7])
    rowoff = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    A = np.random.default_rng(22).standard_normal((int(counts.sum()), 256)).astype(np.float16)
    outs = []
    for skip in (False, True):
        monkeypatch.setattr(NX, "SKIP_TAIL", skip)
        outs.append(np.array(NX.expert_gemm((st,), mx.array(np.arange(G, dtype=np.int32)), mx.array(rowoff), mx.array(NX.permute_k(A)),
                                            A.shape[0], int(counts.max()), cfg)))
    assert np.array_equal(outs[0].view(np.uint16), outs[1].view(np.uint16))


def test_pick_cfg_counts_a_skipped_tail_as_a_half_block(monkeypatch):
    """Experts of 33..64 rows: with the tail skip a 64-row block costs one 32-row block for a tail up to 32 rows, so 64-row blocks win for
    rows around 32 where plain block counting picked 32-row blocks."""
    monkeypatch.setattr(NX, "_BLOCK_COST", {16: 0.78, 32: 1.01, 64: 1.52})        # the Air table this test was written for (the Ultra table makes 64-row blocks win everywhere)
    counts = np.array([20, 28, 32, 40, 44, 30, 36, 50] * 8)
    monkeypatch.setattr(NX, "SKIP_TAIL", True)
    assert NX.pick_cfg(counts) == NX._BLOCK_CFG[64]
    monkeypatch.setattr(NX, "SKIP_TAIL", False)
    assert NX.pick_cfg(counts) == NX._BLOCK_CFG[32]
    assert NX.pick_cfg(np.full(100, 6)) == NX._BLOCK_CFG[16]

"""foreign_moe: the fused decode MoE block (router, experts, combine) against the stock mlx-lm block on the same stored packs. Metal only; small shapes."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)


def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")
E, TOP, H, I = 40, 6, 512, 512


def _packs8(t, n, k, seed, count=E):
    """FP8 experts as the safetensors loader builds them: fp8 (one scale per row) or fp8b (block scale per 32-unit over 128 x 128, power-of-two row scale)."""
    from kyojin_mlx import foreign_pack as FP
    out = []
    for e in range(count):
        r = RNG(seed + e)
        q = r.integers(0, 256, (n, k), dtype=np.uint8)
        q[(q & 127) == 127] = 0x38
        if t == "FP8":
            out.append(FP.Pack("fp8", n, k, q, None, r.uniform(5e-5, 2e-4, n).astype(np.float32)))
        else:
            g = r.uniform(64.0, 127.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
            out.append(FP.Pack("fp8b", n, k, q, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -20)))
    return out


def _packs(t, n, k, seed):
    if t in ("FP8", "FP8B"):
        return _packs8(t, n, k, seed)
    out = []
    _, be, bb = GF.TYPES_BY_NAME[t]
    for e in range(E):
        raw = SY.random_blocks(t, n, k, RNG(seed + e))
        if be:
            SY.set_scales(t, raw.reshape(n, k // be, bb), RNG(seed + 90 + e).uniform(1e-4, 5e-4, (n, k // be)))
        out.append(GF.to_pack(t, raw, k))
    return out


def _block(types, norm, shared, seed=0):
    import mlx.core as mx
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from kyojin_mlx import foreign_linear as FL

    class A:
        hidden_size, moe_intermediate_size, shared_expert_intermediate_size = H, I, I
        norm_topk_prob, num_experts, num_experts_per_tok = norm, E, TOP

    blk = Qwen3NextSparseMoeBlock(A)
    rng = RNG(seed)
    blk.gate.weight = mx.array(rng.standard_normal((E, H)).astype(np.float32) * 0.3)
    blk.shared_expert_gate.weight = mx.array(rng.standard_normal((1, H)).astype(np.float32) * 0.3)
    for m in (blk.shared_expert.gate_proj, blk.shared_expert.up_proj, blk.shared_expert.down_proj):
        m.weight = mx.array((rng.standard_normal(m.weight.shape) * 0.05).astype(np.float16))
    blk.switch_mlp = FL.ForeignSwitchGLU(FL.DevPack(_packs(types[0], I, H, seed + 1)), FL.DevPack(_packs(types[1], I, H, seed + 300)),
                                         FL.DevPack(_packs(types[2], H, I, seed + 600)))
    if not shared:
        blk.shared_expert = None
    return blk


@needs_gpu
@pytest.mark.parametrize("types", [("Q4_K", "Q4_K", "Q5_K"), ("Q5_K", "Q5_K", "Q6_K"), ("Q8_0", "Q8_0", "Q8_0"), ("IQ4_XS", "IQ4_XS", "Q4_K"), ("FP8", "FP8", "FP8"), ("FP8B", "FP8B", "FP8B")])
@pytest.mark.parametrize("T,norm", [(1, True), (4, True), (3, False)])
def test_fused_block_matches_stock(types, T, norm, monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_moe as FM
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    blk = _block(types, norm, True)
    x = mx.array(RNG(5).standard_normal((1, T, H)).astype(np.float16))
    ref = blk(x)
    mx.eval(ref)
    assert FM.install(type("M", (), {"named_modules": lambda s: [("b", blk)]})(), log=lambda *_: None) == 1
    y = blk(x)
    mx.eval(y)
    assert y.dtype == (mx.float16 if FM.MODE == "2" else ref.dtype) and y.shape == ref.shape
    r, f = np.array(ref, np.float32), np.array(y, np.float32)
    assert np.isfinite(f).all()
    assert np.abs(f - r).max() <= 2e-3 * max(np.abs(r).max(), 1e-6), np.abs(f - r).max() / np.abs(r).max()


@needs_gpu
def test_router_matches_stock_topk():
    """Indices (as a set per token) and scores of the fused router against softmax / argpartition / renormalise; ties are not in random data."""
    import mlx.core as mx
    from kyojin_mlx import foreign_moe as FM
    rng = RNG(3)
    T = 5
    gw = mx.array(rng.standard_normal((E, H)).astype(np.float32))
    sgw = mx.array(rng.standard_normal((H,)).astype(np.float32))
    x = mx.array(rng.standard_normal((T, H)).astype(np.float16))
    inds, scs, sgs = FM._select(FM._logits(x, gw, sgw, T, E, H, True), T, E, TOP, True)
    xn, gn = np.array(x, np.float64), np.array(gw, np.float64)           # MLX's f32 matmul is reduced precision (3e-2 abs here): the float64 product is the reference
    lgt = xn @ gn.T
    g = np.exp(lgt - lgt.max(-1, keepdims=True))
    g /= g.sum(-1, keepdims=True)
    ri = np.argsort(-g, axis=-1)[:, :TOP]
    rs = np.take_along_axis(g, ri, axis=-1)
    rs = rs / rs.sum(-1, keepdims=True)
    ref_sg = 1 / (1 + np.exp(-(xn @ np.array(sgw, np.float64))))
    mx.eval(inds, scs, sgs)
    inds, scs = (np.array(a).reshape(T, TOP) for a in (inds, scs))
    for t in range(T):
        assert set(inds[t]) == set(ri[t].tolist())
        assert np.allclose(np.sort(scs[t]), np.sort(rs[t]), rtol=1e-3)
    assert np.allclose(np.array(sgs), ref_sg, rtol=1e-4)


@needs_gpu
def test_falls_back_over_the_gather_limit_and_without_shared(monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL, foreign_moe as FM
    blk = _block(("Q4_K", "Q4_K", "Q5_K"), True, True)
    for m in (blk.shared_expert.gate_proj, blk.shared_expert.up_proj, blk.shared_expert.down_proj):
        m.weight = mx.zeros(m.weight.shape, mx.float16)       # a zero shared expert is a block without one
    x = mx.array(RNG(5).standard_normal((1, 2, H)).astype(np.float16))
    ref = blk(x)
    mx.eval(ref)
    blk.shared_expert = None
    FM.install(type("M", (), {"named_modules": lambda s: [("b", blk)]})(), log=lambda *_: None)
    y = blk(x)                                              # no shared expert: still fused
    big = mx.array(RNG(6).standard_normal((1, FL.GATHER_MAX // TOP + 1, H)).astype(np.float16))
    blk2 = _block(("Q4_K", "Q4_K", "Q5_K"), True, True)
    FM.install(type("M", (), {"named_modules": lambda s: [("b", blk2)]})(), log=lambda *_: None)
    z = blk2(big)                                            # over the gather limit: the stock call
    mx.eval(ref, y, z)
    assert np.abs(np.array(y, np.float32) - np.array(ref, np.float32)).max() <= 2e-3 * np.abs(np.array(ref, np.float32)).max()
    assert z.shape == big.shape and np.isfinite(np.array(z, np.float32)).all()


@needs_gpu
@pytest.mark.parametrize("t", ["Q5_K", "Q6_K", "IQ4_XS", "Q3_K", "IQ2_S"])
def test_glu_down_is_bit_identical_to_silu_mul_then_down(t, monkeypatch):
    """gather_glu stages silu(a) * b itself; the product must equal the silu-mul launch followed by the plain gather, bit for bit."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL, foreign_moe as FM
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    d = FL.DevPack(_packs(t, H, I, 11))
    assert d.glu_ok()
    S = 6
    rng = RNG(2)
    a = mx.array((rng.standard_normal((S, I)) * 3).astype(np.float16))
    b = mx.array(rng.standard_normal((S, I)).astype(np.float16))
    ids = mx.array(np.array([3, 3, 0, E - 1, 7, 12], np.uint32))
    ar = mx.arange(S).astype(mx.uint32)
    ref = d.gather(FM._silu_mul(a, b), ar, ids)
    y = d.gather_glu(a, b, ar, ids)
    mx.eval(ref, y)
    assert np.array_equal(np.array(ref), np.array(y))


@needs_gpu
@pytest.mark.parametrize("t", ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS", "Q4_0", "IQ4_NL", "Q3_K", "FP8", "FP8B"])
@pytest.mark.parametrize("n_exp,top,k_in,n_out", [(8, 2, 512, 512), (64, 4, 512, 1024), (128, 8, 1024, 512), (8, 8, 2048, 512), (256, 8, 512, 512)])
def test_gather_pair_is_bit_identical_to_two_gathers(t, n_exp, top, k_in, n_out, monkeypatch):
    """gate | up in one launch (gather_pair) equals the two gathers bit for bit, across expert counts, top-k, hidden sizes and the three kernel families."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    _, be, bb = GF.TYPES_BY_NAME.get(t, (None, 0, 0))

    def mk(seed):
        if t in ("FP8", "FP8B"):
            return FL.DevPack(_packs8(t, n_out, k_in, seed, n_exp))
        out = []
        for e in range(n_exp):
            raw = SY.random_blocks(t, n_out, k_in, RNG(seed + e))
            if be:
                SY.set_scales(t, raw.reshape(n_out, k_in // be, bb), RNG(seed + 90 + e).uniform(1e-4, 5e-4, (n_out, k_in // be)))
            out.append(GF.to_pack(t, raw, k_in))
        return FL.DevPack(out)
    g, u = mk(1), mk(500)
    if not g.gemv_ok():
        pytest.skip("shape outside the gemv class")
    assert g.pair_ok(u)
    S = top * 2                                              # two tokens
    x = mx.array(RNG(3).standard_normal((2, k_in)).astype(np.float16))
    tok = mx.array((np.arange(S) // top).astype(np.uint32))
    ids = mx.array(RNG(4).integers(0, n_exp, S).astype(np.uint32))
    ra, rb = g.gather(x, tok, ids), u.gather(x, tok, ids)
    pa, pb = g.gather_pair(u, x, tok, ids)
    mx.eval(ra, rb, pa, pb)
    assert np.array_equal(np.array(ra), np.array(pa)) and np.array_equal(np.array(rb), np.array(pb))


@needs_gpu
@pytest.mark.parametrize("types", [("Q4_K", "Q4_K", "Q5_K"), ("Q5_K", "Q5_K", "Q6_K"), ("Q8_0", "Q8_0", "Q8_0"), ("IQ4_XS", "IQ4_XS", "Q4_K"), ("FP8", "FP8", "FP8"), ("FP8B", "FP8B", "FP8B")])
def test_prefill_sorted_gmm_matches_per_expert(types, monkeypatch):
    """Prefill slots (above GATHER_MAX): the sorted gathered matmul over the decoded stacks gives the per-expert path's output."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    blk = _block(types, True, True)
    sm = blk.switch_mlp
    x = mx.array(RNG(7).standard_normal((1, 40, H)).astype(np.float16))
    inds = mx.array(RNG(8).integers(0, blk.num_experts, (1, 40, blk.top_k)).astype(np.uint32))
    monkeypatch.setattr(FL, "GATHER_MAX", 8)
    monkeypatch.setattr(FL, "GMM", False)
    ref = sm(x, inds)
    monkeypatch.setattr(FL, "GMM", True)
    assert sm._sorted_gmm(x.reshape(-1, H), inds.reshape(-1), 40, blk.top_k) is not None
    y = sm(x, inds)
    mx.eval(ref, y)
    r, f = np.array(ref, np.float32), np.array(y, np.float32)
    assert y.shape == ref.shape and np.isfinite(f).all()
    assert np.abs(f - r).max() <= 3e-3 * max(np.abs(r).max(), 1e-6)


@needs_gpu
@pytest.mark.parametrize("t", ["Q4_K", "Q5_K", "Q6_K", "Q8_0"])
@pytest.mark.parametrize("R", [1, 3, 4])
def test_dense_glu_down_is_bit_identical_to_silu_mul_then_apply(t, R):
    """DevPack.apply_glu (flag KYOJIN_FOREIGN_DENSE_GLU) stages silu(a) * b itself, from two arrays or from one [R, 2 k] gate | up product."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL, foreign_moe as FM
    n, k = 512, 1024
    raw = SY.random_blocks(t, n, k, RNG(5))
    _, be, bb = GF.TYPES_BY_NAME[t]
    if be:
        SY.set_scales(t, raw.reshape(n, k // be, bb), RNG(6).uniform(1e-4, 5e-4, (n, k // be)))
    d = FL.DevPack(GF.to_pack(t, raw, k))
    if not d.glu_dense_ok(R):          # apply() takes another kernel class at this row count: the MLP keeps its own call
        assert R > 1
        pytest.skip("apply() leaves the GEMV class here")
    rng = RNG(7)
    a = mx.array((rng.standard_normal((R, k)) * 3).astype(np.float16))
    b = mx.array(rng.standard_normal((R, k)).astype(np.float16))
    ref = d.apply(FM._silu_mul(a, b))
    y = d.apply_glu(a, b)
    gu = mx.concatenate([a, b], axis=1)
    z = d.apply_glu(gu, gu, k, 2 * k)
    mx.eval(ref, y, z)
    assert np.array_equal(np.array(ref), np.array(y)) and np.array_equal(np.array(ref), np.array(z))


@needs_gpu
def test_bf16_router_gets_the_fused_route(monkeypatch):
    """A safetensors release stores the router in bf16: install promotes it to f32, so the 1-row decode takes the fused kernels and matches the stock block."""
    import mlx.core as mx
    from kyojin_mlx import foreign_moe as FM
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    blk = _block(("FP8B", "FP8B", "FP8B"), True, True)
    blk.gate.weight = blk.gate.weight.astype(mx.bfloat16)
    blk.shared_expert_gate.weight = blk.shared_expert_gate.weight.astype(mx.bfloat16)
    x = mx.array(RNG(5).standard_normal((1, 1, H)).astype(np.float16))
    ref = blk(x)
    mx.eval(ref)
    assert not FM._eligible(blk, x)
    FM.install(type("M", (), {"named_modules": lambda s: [("b", blk)]})(), log=lambda *_: None)
    assert blk.gate.weight.dtype == mx.float32 and FM._eligible(blk, x)
    y = blk(x)
    mx.eval(y)
    r, f = np.array(ref, np.float32), np.array(y, np.float32)
    assert np.abs(f - r).max() <= 2e-2 * np.abs(r).max()

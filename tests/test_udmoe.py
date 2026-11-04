"""Routed-expert stacks in the stored GGUF format (K / IQ / Q8_0 / Q4_K ...): E matrices as E * n rows, a gather GEMV by expert id for decode, per-expert views for
prefill. Against the float32 ggml decode. Metal only; small shapes."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)
TYPES = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS", "IQ4_NL", "Q4_0", "Q3_K", "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M")
E, N, K = 6, 64, 1024


def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")


def _experts(t, seed=0):
    """E packs with small scales (no fp16 overflow in the reference) and their float32 decode [E, N, K]."""
    packs, ws = [], []
    _, be, bb = GF.TYPES_BY_NAME[t]
    for e in range(E):
        raw = SY.random_blocks(t, N, K, RNG(seed + e))
        if be:
            SY.set_scales(t, raw.reshape(N, K // be, bb), RNG(seed + 50 + e).uniform(1e-4, 5e-4, (N, K // be)))
        packs.append(GF.to_pack(t, raw, K))
        ws.append(GF.dequant(t, raw, K))
    return packs, np.stack(ws)


def _check(y, ref, tol=2e-3):
    y = np.array(y, np.float32)
    assert np.isfinite(y).all()
    assert np.abs(y - ref).max() <= tol * max(np.abs(ref).max(), 1e-6), np.abs(y - ref).max() / np.abs(ref).max()


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
def test_gather_matches_per_expert_decode(t, monkeypatch):
    """Slots = tokens x top-k: slot s reads row xrows[s] of x and expert eids[s] (repeats and unused experts included); one launch."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    packs, ws = _experts(t)
    d = FL.DevPack(packs)
    assert d.E == E and d.gemv_ok() and d.w16 is None and d.mq is None
    T, top = 3, 4
    x = RNG(7).standard_normal((T, K)).astype(np.float16)
    eids = np.array([5, 0, 2, 5, 1, 1, 3, 0, 4, 2, 2, 5], np.uint32)
    xrows = np.repeat(np.arange(T), top).astype(np.uint32)
    y = d.gather(mx.array(x), mx.array(xrows), mx.array(eids))
    mx.eval(y)
    assert y.dtype == mx.float16 and y.shape == (T * top, N)
    ref = np.stack([x[xrows[s]].astype(np.float32) @ ws[eids[s]].T for s in range(len(eids))])
    _check(y, ref)


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("R", (1, 6, 16))
def test_expert_view_apply_matches_decode(t, R, monkeypatch):
    """The per-expert view (prefill route) of a stack, R rows, equals the decode of that expert."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    packs, ws = _experts(t, seed=3)
    d = FL.DevPack(packs)
    x = RNG(8).standard_normal((R, K)).astype(np.float16)
    for e in (0, 3, E - 1):
        _check(d.expert(e).apply(mx.array(x)), x.astype(np.float32) @ ws[e].T)


@needs_gpu
@pytest.mark.parametrize("t", ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ2_XXS"))
def test_switch_glu_gather_equals_grouped(t, monkeypatch):
    """ForeignSwitchGLU: the decode gather launch (S <= GATHER_MAX) and the sorted per-expert route (S above) give the same answer."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    H, I = 512, 512

    def stack(n, k, seed):
        _, be, bb = GF.TYPES_BY_NAME[t]
        ps = []
        for e in range(E):
            raw = SY.random_blocks(t, n, k, RNG(seed + e))
            SY.set_scales(t, raw.reshape(n, k // be, bb), RNG(seed + 40 + e).uniform(1e-4, 5e-4, (n, k // be)))
            ps.append(GF.to_pack(t, raw, k))
        return FL.DevPack(ps)

    glu = FL.ForeignSwitchGLU(stack(I, H, 1), stack(I, H, 100), stack(H, I, 200))
    x = mx.array(RNG(4).standard_normal((2, 3, H)).astype(np.float16))
    inds = mx.array(RNG(5).integers(0, E, (2, 3, 2)).astype(np.uint32))
    a = glu(x, inds)
    monkeypatch.setattr(FL, "GATHER_MAX", 0)
    b = glu(x, inds)
    mx.eval(a, b)
    assert a.shape == (2, 3, 2, H)
    _check(a, np.array(b, np.float32), tol=5e-3)

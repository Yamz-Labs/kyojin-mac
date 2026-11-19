"""MEM-CSF: compact expert scale form (flag KYOJIN_MOE_SCALE_COMPACT). CPU: exact round trip with csf-like statistics. GPU (micro, tiny packs): every decode MoE kernel reading the
compact arrays is bit-identical to the same kernel reading the expanded `meta`; with scm off the kernel sources are untouched."""
import types

import numpy as np
import pytest

from kyojin_mlx import moe_scale_compact as MC


def csf_like(rows, cols, seed, exc=0.003, hi_base=False):
    """Scale bytes like a csf32 tensor: row base + 4-bit code (all 16 values used), ~0.3 % exceptions (bytes below / above the window, some equal to the window edge)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0xE8 if hi_base else 0x20, 0xF1 if hi_base else 0x41, size=rows)
    full = base[:, None] + rng.integers(0, 16, size=(rows, cols))
    m = rng.random((rows, cols)) < exc
    full[m] = rng.integers(0, 256, size=int(m.sum()))
    return full.astype(np.uint8)


@pytest.mark.parametrize("rows,cols,exc,hi", [(64, 128, 0.003, False), (96, 256, 0.003, False), (64, 192, 0.03, False), (32, 64, 0.3, False), (32, 128, 0.003, True), (16, 2, 0.5, False)])
def test_roundtrip(rows, cols, exc, hi):
    full = csf_like(rows, cols, rows + cols, exc, hi)
    c = MC.pack_compact(full)
    assert (MC.expand(c) == full).all()
    st = MC.stack([MC.pack_compact(csf_like(rows, cols, s, exc, hi)) for s in range(3)])
    for e in range(3):
        assert (MC.expand(st, e) == csf_like(rows, cols, e, exc, hi)).all()
    assert MC.pack_compact(np.zeros((8, 16), np.uint8))["ecol"].size == 0


def test_bytes_vs_expanded():
    c = MC.pack_compact(csf_like(2048, 256, 1))
    assert MC.nbytes(c) < 0.55 * 2048 * 256


mx = pytest.importorskip("mlx.core")
gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
H, I, K, LIMIT, E = 4096, 1024, 8, 10.0, 24


def mk(k, n, seed, exc=0.003):
    rng = np.random.default_rng(seed)
    qs = mx.array(rng.integers(0, 2 ** 32, size=(E, n, k // 8), dtype=np.uint32))
    fulls = [csf_like(n, k // 16, seed * 100 + e, exc, hi_base=(e % 5 == 0)) for e in range(E)]
    meta = mx.array(np.stack(fulls).view(np.uint16).reshape(E, n, k // 32))
    st = MC.stack([MC.pack_compact(f) for f in fulls])
    cm = {a: mx.array(st[a]) for a in MC.ARRAYS}
    gs = mx.full((E,), 0.004, dtype=mx.float32)
    return types.SimpleNamespace(qs=qs, meta=meta, gs=gs, k=k, n=n, cm=cm), types.SimpleNamespace(qs=qs, meta=meta, gs=gs, k=k, n=n)


@gpu
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("exc", [0.003, 0.05])
def test_decode_kernels_bit_identical(T, exc):
    from kyojin_mlx import glm_moe_e as GE
    from kyojin_mlx import glm_moe_rows as R
    (gc, g), (uc, u), (dc, d) = mk(H, I, 1, exc), mk(H, I, 2, exc), mk(I, H, 3, exc)
    rng = np.random.default_rng(T)
    inds = mx.array(np.concatenate([rng.choice(E, size=K, replace=False) if t == 0 else np.concatenate([np.arange(E)[:3], rng.permutation(np.arange(3, E))[:K - 3]]) for t in range(T)]).astype(np.uint32))
    wts = mx.array(rng.uniform(0.05, 0.4, size=(T * K,)).astype(np.float32))
    x2 = mx.array((rng.normal(size=(T, H)) * 0.5).astype(np.float16))
    sh = mx.array((rng.normal(size=(T, H)) * 0.1).astype(np.float16))
    tab = R.group(inds, T, K) if T > 1 else None
    for kind in ("g2m", "inl", "omx"):
        h0 = GE.pair_glu(kind, g, u, x2, inds, T, K, LIMIT)
        h1 = GE.pair_glu(kind, gc, uc, x2, inds, T, K, LIMIT)
        assert mx.array_equal(h0, h1), kind
        assert mx.array_equal(GE.down(kind, d, h0, inds, T, K), GE.down(kind, dc, h0, inds, T, K)), kind
    h0 = GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT)
    assert mx.array_equal(GE.down_combine(d, h0, inds, wts, sh, T, K), GE.down_combine(dc, h0, inds, wts, sh, T, K))
    for kind in ("g2m", "omx"):
        assert mx.array_equal(GE.down_combine_seq(d, h0, inds, wts, sh, T, K, 2, 2, kind), GE.down_combine_seq(dc, h0, inds, wts, sh, T, K, 2, 2, kind)), kind
    assert mx.array_equal(R.pair_glu(g, u, x2, inds, tab, T, K, LIMIT), R.pair_glu(gc, uc, x2, inds, tab, T, K, LIMIT))
    hr = R.pair_glu(g, u, x2, inds, tab, T, K, LIMIT)
    assert mx.array_equal(R.down(d, hr, inds, tab, T, K), R.down(dc, hr, inds, tab, T, K))
    # the compact path really ran (the kernels are new ones, not the expanded ones)
    from kyojin_mlx import kernels as KN
    assert any("scm" in (k if isinstance(k, tuple) else (k,)) for k in KN._cache)


@gpu
def test_devpack_loader_flag(monkeypatch):
    """Load path: DevPack of E nvfp4 Packs builds `cm` only with KYOJIN_MOE_SCALE_COMPACT, and the decode kernels then give the outputs of the expanded path bit for bit."""
    from kyojin_mlx import foreign_linear as FL, foreign_pack as FP, glm_moe_e as GE
    Ev, k, n = 6, 4096, 1024
    rng = np.random.default_rng(5)

    def packs(seed):
        out = []
        for e in range(Ev):
            full = csf_like(n, k // 16, seed * 31 + e)
            out.append(FP.Pack("nvfp4", n, k, rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8), full.view("<u2").reshape(n, k // 32), 0.004))
        return out
    pk = [packs(1), packs(2)]
    monkeypatch.delenv(MC.FLAG, raising=False)
    off = [FL.DevPack(p) for p in pk]
    monkeypatch.setenv(MC.FLAG, "1")
    on = [FL.DevPack(p) for p in pk]
    assert all(d.cm is None for d in off) and all(d.cm is not None for d in on)
    assert sum(a.nbytes for a in MC.cm_list(on[0])) < 0.55 * off[0].meta.nbytes
    T = 2
    inds = mx.array(np.array([0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0][:T * 6], np.uint32))
    x2 = mx.array((rng.normal(size=(T, k)) * 0.5).astype(np.float16))
    h0 = GE.pair_glu("g2m", off[0], off[1], x2, inds, T, 6, 10.0)
    h1 = GE.pair_glu("g2m", on[0], on[1], x2, inds, T, 6, 10.0)
    assert mx.array_equal(h0, h1)


@gpu
def test_expand_dev_and_devpack_drop(monkeypatch):
    """Transient GPU expand == the stored meta, whole stack and per expert; a DevPack built with the flag keeps no resident meta yet every `meta` read returns the same bytes
    (stack, expert view, row-gathered copy); KEEP_META keeps it; a view of a stack without cm fails loudly."""
    from kyojin_mlx import foreign_linear as FL
    k, n, E_ = 512, 64, 5
    fulls = [csf_like(n, k // 16, 40 + e, 0.05, hi_base=(e % 2 == 0)) for e in range(E_)]
    st = MC.stack([MC.pack_compact(f) for f in fulls])
    cm = {a: mx.array(st[a]) for a in MC.ARRAYS}
    ref = np.stack(fulls).view(np.uint16).reshape(E_, n, k // 32)
    assert (np.array(MC.expand_dev(cm, n, k // 16, 0, E_)) == ref).all()
    for e in (0, 3, 4):
        assert (np.array(MC.expand_dev(cm, n, k // 16, e, 1))[0] == ref[e]).all()
    rng = np.random.default_rng(5)
    mkp = lambda e: types.SimpleNamespace(fmt="nvfp4", n=n, k=k, qs=rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8), meta=ref[e].copy(), gscale=0.01, extra={})
    ps = [mkp(e) for e in range(E_)]
    monkeypatch.setenv(MC.FLAG, "1")
    monkeypatch.delenv(MC.KEEP_FLAG, raising=False)
    d = FL.DevPack(ps)
    assert d.cm is not None and d._meta is None and d.nbytes() < sum(p.qs.nbytes for p in ps) + 4 * E_ + sum(a.nbytes for a in MC.cm_list(d)) + 1
    assert (np.array(d.meta) == ref).all()
    for e in (1, 4):
        assert (np.array(d.expert(e).meta) == ref[e]).all()
    monkeypatch.setenv(MC.KEEP_FLAG, "1")
    assert FL.DevPack(ps)._meta is not None
    v = d.expert(2)
    v._msrc = (types.SimpleNamespace(cm=None, n=n, k=k), 2)
    with pytest.raises(AssertionError, match="compact scales missing"):
        v.meta

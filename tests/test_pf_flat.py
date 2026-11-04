"""pf_flat (prefill flatness past 128K): the exact selection glue of the block-max / candidate-id levers against the stock chain, on the CPU with numpy scoring.
The Metal kernels (block-max score kernel, fused gathered rescoring) are checked on the Mac by tools/bench_pf_flat.py; here: kernel source structure, ranking,
tie-breaking, compaction, the forced last block of the candidate source, and the flag table."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL          # noqa: E402
from kyojin_mlx import idx_topk as IT       # noqa: E402
from kyojin_mlx import pf_flat as PF        # noqa: E402
from kyojin_mlx import pfroof2 as PR        # noqa: E402

NH, D, BS = 32, 128, 8


@pytest.fixture(autouse=True)
def cpu_glue(monkeypatch):
    monkeypatch.setattr(PF, "METAL_OPT", False)
    monkeypatch.setattr(PF, "BMAX", True)
    monkeypatch.setattr(PF, "BMAX_MIN", 0)
    monkeypatch.setattr(PF, "GSCORE", True)


def make(T, Tc, seed=0, quant=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, NH, D)).astype(np.float16)
    k = rng.standard_normal((1, Tc, D)).astype(np.float16)
    w = (rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16)
    if quant:                                                          # few distinct values: many exact ties
        q, k = (np.round(q * 2) / 2).astype(np.float16), (np.round(k * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


def full_scores(q, k, w, start, ratio):
    return mx.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, ratio))


def stock_topk(s, topk):
    return np.array(IT.topk(s, topk))


def stock_select_blocks(s, clen, K):
    """select_candidate_blocks as the kernel `_SELECT` (MODE 1) defines it: block maxima over keys < clen, the last visible block first whatever its maximum, then the
    K best (ties to the lower block), kept only when the maximum is finite. Returns bool [T, nblk]."""
    s = np.array(s)
    T, Tc = s.shape
    nblk = -(-Tc // BS)
    sp = np.pad(s, [(0, 0), (0, nblk * BS - Tc)], constant_values=-np.inf)
    bm = sp.reshape(T, nblk, BS).max(-1)
    out = np.zeros((T, nblk), dtype=bool)
    for t in range(T):
        cnt = -(-int(clen[t]) // BS)
        key = bm[t, :cnt].copy()
        key[(int(clen[t]) - 1) // BS] = np.inf
        order = np.argsort(-key, kind="stable")[:min(K, cnt)]
        out[t, order] = key[order] > -np.inf
    return out


@pytest.mark.parametrize("T,Tc,topk,quant,ratio", [(48, 4000, 32, False, 4), (40, 6001, 64, True, 4), (33, 3000, 16, True, 2), (64, 5000, 100, False, 1)])
def test_plain_topk_equals_stock(T, Tc, topk, quant, ratio):
    q, k, w = make(T, Tc, quant=quant)
    start = ratio * (Tc - T)
    s = full_scores(q, k, w, start, ratio)
    ref = stock_topk(s, topk)
    for qb in ("0", "7"):                                            # one row block, and 7-row blocks
        import os
        os.environ[IT.QBLOCK_ENV] = qb
        try:
            got = np.array(PF.plain_topk(q, k, w, start, ratio, topk, T))
        finally:
            os.environ.pop(IT.QBLOCK_ENV, None)
        assert np.array_equal(got, ref)


def test_plain_topk_ties_all_equal_keys():
    T, Tc, topk = 24, 2048, 20
    q, _, w = make(T, Tc)
    k = mx.array(np.ones((1, Tc, D), dtype=np.float16) * 0.5)
    q = mx.array(np.ones((1, T, NH, D), dtype=np.float16) * 0.25)
    w = mx.array(np.ones((1, T, NH), dtype=np.float16) * 0.03125)
    start = 4 * (Tc - T)
    s = full_scores(q, k, w, start, 4)
    assert np.array_equal(np.array(PF.plain_topk(q, k, w, start, 4, topk, T)), stock_topk(s, topk))      # every score equal: the lowest positions win


@pytest.mark.parametrize("T,Tc,topk,K,quant", [(48, 4001, 32, 40, False), (40, 5000, 32, 40, True), (33, 3000, 16, 17, True)])
def test_source_blocks_and_topk_equal_stock(T, Tc, topk, K, quant):
    q, k, w = make(T, Tc, quant=quant)
    ratio, start = 4, 4 * (Tc - T)
    s = full_scores(q, k, w, start, ratio)
    clen = (start + np.arange(T) + 1) // ratio
    idxs, ids = PF.source_topk(q, k, w, start, ratio, topk, T, K)
    assert np.array_equal(np.array(idxs), stock_topk(s, topk))                               # the source's own top-k
    want = stock_select_blocks(s, clen, K)
    got = np.array(PF.mask_from_ids(ids, -(-Tc // BS), -(-Tc // BS) * BS))[:, ::BS]
    assert np.array_equal(got, want)                                                          # the candidate blocks (forced last, ties to the lower block)
    ii = np.array(ids)
    assert np.all(np.diff(np.where(ii >= 0, ii, 10 ** 9), axis=1) >= 0)                       # ascending, -1 padding last


@pytest.mark.parametrize("quant", [False, True])
def test_consumer_over_source_blocks_equals_masked_stock(quant):
    T, Tc, topk, K = 40, 6000, 32, 48
    qs, ks, ws = make(T, Tc, seed=1, quant=quant)                                              # the source layer
    qc, kc, wc = make(T, Tc, seed=2, quant=quant)                                              # a consumer layer (its own q / keys / weights)
    ratio, start = 4, 4 * (Tc - T)
    _, ids = PF.source_topk(qs, ks, ws, start, ratio, topk, T, K)
    mask = np.array(PF.mask_from_ids(ids, -(-Tc // BS), Tc))
    sc = np.array(full_scores(qc, kc, wc, start, ratio))
    ref = stock_topk(mx.array(np.where(mask, sc, -np.inf).astype(np.float32)), topk)
    got = np.array(PF.consumer_topk(qc, kc, wc, start, ratio, topk, T, ids))
    assert np.array_equal(got, ref)


def test_pick_blocks_force_last_and_no_force():
    bm = mx.array(np.array([[1., 5., 3., 4., 2., 0.5], [9., 9., 9., 9., 9., 9.], [1., 2., -np.inf, -np.inf, -np.inf, -np.inf]], dtype=np.float32))
    clen = mx.array([6 * BS, 6 * BS - 3, 2 * BS + 1])                                           # last visible block: 5, 5, 2 (block 2 is partially visible, its max is -inf)
    got = np.array(PF.pick_blocks(bm, clen, 3, False))
    assert got.tolist() == [[1, 2, 3], [0, 1, 2], [0, 1, 2]]                                   # value desc then lower block; row 2 sees 3 blocks (17 keys)
    got = np.array(PF.pick_blocks(bm, clen, 3, True))
    assert got.tolist() == [[1, 3, 5], [0, 1, 5], [0, 1, 2]]                           # last visible block forced in (row 2: block 2 forced although its keys are -inf-free)


def test_ids_from_mask_equals_argsort_and_caps_at_k():
    rng = np.random.default_rng(3)
    flag = rng.random((20, 500)) < 0.2
    K = 60
    got = np.array(PF.ids_from_mask(mx.array(flag), K))
    order = np.argsort(~flag, axis=1, kind="stable")[:, :K]
    ok = np.take_along_axis(flag, order, axis=1)
    assert np.array_equal(got, np.where(ok, order, -1))
    flag[0, :] = True                                                                           # more than K kept: the first K, like argsort(~flag)[:K]
    assert np.array_equal(np.array(PF.ids_from_mask(mx.array(flag), K))[0], np.arange(K))
    back = np.array(PF.mask_from_ids(PF.ids_from_mask(mx.array(flag), 200), 500, 500 * BS))[:, ::BS]
    assert np.array_equal(back[1:], flag[1:])


def test_make_ids_reads_block_flags_of_the_mask():
    rng = np.random.default_rng(4)
    flag = rng.random((2100, 90)) < 0.1
    masks = mx.array(np.repeat(flag, BS, axis=1)[:, :90 * BS])
    ids = np.array(PF.make_ids(masks, 30))                                                      # 2100 rows: three 1024-row pieces
    ref = np.array(PF.ids_from_mask(mx.array(flag), 30))
    assert np.array_equal(ids, ref)


def test_stash_fetch_clear_are_chunk_scoped():
    cache = types.SimpleNamespace()
    ids = mx.zeros((4, 2), dtype=mx.int32)
    assert PF.fetch(cache, 0, 4, 100) is None
    PF.stash(cache, ids, 128, 4, 100)
    assert PF.fetch(cache, 128, 4, 100) is ids
    assert PF.fetch(cache, 0, 4, 100) is None and PF.fetch(cache, 128, 4, 101) is None      # another chunk's ids are never used
    PF.clear(cache)
    assert PF.fetch(cache, 128, 4, 100) is None


def test_bmax_ok_gates(monkeypatch):
    q, k, w = make(8, 4096)
    assert PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, False)
    assert PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, True, 40)
    assert not PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, True, 24)       # fewer candidate blocks than picks: the source top-k could leave the kept blocks
    monkeypatch.setattr(PF, "BMAX", False)
    assert not PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, False)
    monkeypatch.setattr(PF, "BMAX", True)
    assert not PF.bmax_ok(q, k, w, 4096, 32, 4 * 10, 4, False)            # the first row sees fewer than topk keys: the stock path (filler picks) keeps it
    assert not PF.bmax_ok(q, k, w, 200, 32, 4 * 4000, 4, False)           # too few keys for 8 blocks per pick
    monkeypatch.setattr(PF, "BMAX_MIN", 8192)
    assert not PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, False)
    monkeypatch.setattr(PF, "BMAX_MIN", 0)
    monkeypatch.setattr(PF, "GSCORE", False)
    assert not PF.bmax_ok(q, k, w, 4096, 32, 4 * 4000, 4, True, 40)       # the source needs the fused rescoring


def test_kernel_sources():
    src = PF.bmax_source()
    assert "simd_shuffle_xor(s, 4u)" in src and "S[ulong(t) * Tc + n]" not in src and src.count("-INFINITY") >= 2
    gs = PF._GSCORE.replace("NSG", str(PF.GS_NSG))
    assert "NSG" not in gs and "stage[2]" in gs
    assert gs.index("op.run(tK, tQ, cT)") < gs.index("scr[uint(ix[1]) * 33")                  # the score tile overwrites the staged keys only after the matmul read them
    assert 8192 + 128 + PF.GS_NSG * 8192 <= 32768                                              # threadgroup memory of the fused rescoring kernel


def test_flag_table_matches_module_literals():
    for name, mod, attr in (("pf_idx_bmax", "pf_flat", "BMAX"), ("pf_idx_gscore", "pf_flat", "GSCORE"), ("pf_cand_ids", "pf_flat", "CAND_IDS"),
                            ("pf_idx_selfuse", "pf_flat", "SELFUSE")):
        f = FL.FLAGS[name]
        assert f.default == "1" and (f.attrs[0][0], f.attrs[0][1]) == (mod, attr)
    assert FL.FLAGS["pf_idx_bmax_min"].kind == "int" and FL.FLAGS["pf_idx_bmax_min"].default == str(65536)


# ----------------------------------------------------------------------------------------------------------------- int8 first pass (pf_idx_i8)
def _fp4_like(rng, shape):
    """Values on an FP4-like grid: E2M1 magnitudes times a power of two per 32 dims."""
    mags = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    v = mags[rng.integers(0, 8, shape)] * rng.choice([-1.0, 1.0], shape)
    sc = np.exp2(rng.integers(-5, -3, shape[:-1] + (shape[-1] // 32,))).astype(np.float32)
    return (v * np.repeat(sc, 32, axis=-1)).astype(np.float16)


@pytest.mark.parametrize("metal", [False, True])
def test_i8pack_is_lossless_on_the_grid_and_rounds_off_it(metal, monkeypatch):
    if metal and not mx.metal.is_available():
        pytest.skip("needs Metal")
    monkeypatch.setattr(PF, "METAL_OPT", metal)
    rng = np.random.default_rng(8)
    a = _fp4_like(rng, (4096, 128))
    a[7] = 0.0
    c, e = PF.i8pack(mx.array(a))
    mx.eval(c, e)
    c, e = np.array(c), np.array(e)
    assert c.dtype == np.int8 and e.dtype == np.float32 and np.abs(c).max() <= 127
    assert np.array_equal(c.astype(np.float32) * e[:, None], a.astype(np.float32))
    assert e[7] == 1.0 and not c[7].any()
    wide = np.zeros((1, 128), np.float16)
    wide[0, 0], wide[0, 1] = 48.0, 1 / 32                                       # range past int8: the small value rounds, the row stays within bounds
    c, e = PF.i8pack(mx.array(wide))
    mx.eval(c, e)
    assert abs(float(np.array(c)[0, 0]) * float(np.array(e)[0]) - 48.0) < 1e-3


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_i8_bmax_matches_the_half_kernel_up_to_the_rounding(monkeypatch):
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no matmul2d")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    rng = np.random.default_rng(9)
    T, Tc = 64, 4096
    q = mx.array(_fp4_like(rng, (T, 32, 128)))[None]
    k = mx.array(_fp4_like(rng, (Tc, 128)))[None]
    w = mx.array((rng.standard_normal((T, 32)) * .05).astype(np.float16))[None]
    start, ratio = 2 * Tc, 2
    ref = PF.bmax_scores(q, k, w, start, ratio, 0, T)
    ki, ke = PF.i8pack(k[0])
    qi, qe = PF.i8pack(q[0].reshape(-1, 128))
    w8 = w[0].astype(mx.float32) * qe.reshape(T, 32)
    got = PF.bmax_i8(qi, ki, w8, ke, start, ratio, 0, T)
    mx.eval(ref, got)
    r, g = np.array(ref), np.array(got)
    assert np.allclose(r, g, rtol=2e-2, atol=2e-2)
    for a, b in zip(r, g):                                                      # the best blocks survive
        assert np.argmax(a) in np.argsort(-b)[:8]


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("tpt", [8, 32])
def test_i8_register_post_equals_the_slow_post(tpt, monkeypatch):
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available() or PF.i8_tile_map() is None:
        pytest.skip("no matmul2d or another tile layout")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    monkeypatch.setattr(PF, "I8_TPT", tpt)
    rng = np.random.default_rng(10)
    T, Tc = 40, 128 * 40 + 37                                                   # a tail tile that is not on the block grid, and rows whose causal length cuts the keys
    q = mx.array(_fp4_like(rng, (T, 32, 128)))[None]
    k = mx.array(_fp4_like(rng, (Tc, 128)))[None]
    w = (rng.standard_normal((T, 32)) * .05).astype(np.float32)
    ki, ke = PF.i8pack(k[0])
    qi, qe = PF.i8pack(q[0].reshape(-1, 128))
    w8 = mx.array(w) * qe.reshape(T, 32)
    start, ratio = Tc // 2, 1
    out = {}
    for reg in (0, 1):
        monkeypatch.setattr(PF, "I8_REG", bool(reg))
        out[reg] = PF.bmax_i8(qi, ki, w8, ke, start, ratio, 0, T)
        mx.eval(out[reg])
    a, b = np.array(out[0]), np.array(out[1])
    assert np.array_equal(np.isinf(a), np.isinf(b)) and np.isinf(a).any() and np.isfinite(a).any()
    f = np.isfinite(a)
    assert np.allclose(a[f], b[f], rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_i8_plain_topk_equals_the_half_path(monkeypatch):
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no matmul2d")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    rng = np.random.default_rng(10)
    T, Tc, topk = 1024, 16384, 64
    q = mx.array(_fp4_like(rng, (T, 32, 128)))[None]
    k = mx.array(_fp4_like(rng, (Tc, 128)))[None]
    w = mx.array((rng.standard_normal((T, 32)) * .05).astype(np.float16))[None]
    start, ratio = 2 * Tc, 2
    monkeypatch.setattr(PF, "I8", False)
    ref = PF.plain_topk(q, k, w, start, ratio, topk, T)
    monkeypatch.setattr(PF, "I8", True)
    before = PF.HITS["i8_plain"]
    got = PF.plain_topk(q, k, w, start, ratio, topk, T)
    mx.eval(ref, got)
    assert PF.HITS["i8_plain"] == before + 1
    assert np.array_equal(np.array(ref), np.array(got))


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("cand", [80, 96])
def test_i8_source_equals_the_half_path(cand, monkeypatch):
    """Candidate source on the int8 first pass: same kept blocks and same top-k as the half path on keys on the grid."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no matmul2d")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    monkeypatch.setattr(PF, "_I8_GRID", True)
    rng = np.random.default_rng(11)
    T, Tc, topk = 1024, 16384, 64
    q = mx.array(_fp4_like(rng, (T, 32, 128)))[None]
    k = mx.array(_fp4_like(rng, (Tc, 128)))[None]
    w = mx.array((rng.standard_normal((T, 32)) * .05).astype(np.float16))[None]
    start = Tc - T
    monkeypatch.setattr(PF, "I8_SRC", False)
    a = PF.source_topk(q, k, w, start, 1, topk, T, cand)
    monkeypatch.setattr(PF, "I8_SRC", True)
    before = PF.HITS["i8_source"]
    b = PF.source_topk(q, k, w, start, 1, topk, T, cand)
    mx.eval(a[0], a[1], b[0], b[1])
    assert PF.HITS["i8_source"] == before + 1
    assert np.array_equal(np.array(a[0]), np.array(b[0]))
    assert np.array_equal(np.array(a[1]), np.array(b[1]))


def test_rerank_blocks_keeps_last_block_and_drops_empty():
    """The last visible block ranks first whatever its maximum, the rest by maximum with ties to the lower block, a block with no visible key is not kept."""
    ids = mx.array([[0, 2, 5, 7, -1]], dtype=mx.int32)
    sc = np.full((1, 5, 8), -np.inf, dtype=np.float32)
    sc[0, 0, 3], sc[0, 1, 0], sc[0, 2, 1], sc[0, 3, 7] = 5.0, 5.0, 9.0, 0.5
    clen = mx.array([8 * 7 + 1])
    out = PF.rerank_blocks(mx.array(sc.reshape(1, 40)), ids, clen, 3)
    assert np.array(out).tolist() == [[0, 5, 7]]
    out = PF.rerank_blocks(mx.array(sc.reshape(1, 40)), ids, clen, 5)
    assert np.array(out).tolist()[0][:4] == [0, 2, 5, 7] and np.array(out).tolist()[0][4] == -1


def test_i8_guard_keeps_the_half_pass_on_a_pack_off_the_grid(monkeypatch):
    monkeypatch.setattr(PF, "_I8_GRID", None)
    on = mx.array((np.random.default_rng(1).integers(-16, 16, (256, 128)) / 16.0).astype(np.float16))[None]
    assert PF.i8_on_grid(on)
    monkeypatch.setattr(PF, "_I8_GRID", None)
    off = mx.array(np.random.default_rng(2).normal(size=(256, 128)).astype(np.float16))[None]
    assert not PF.i8_on_grid(off)
    assert PF.i8_on_grid(on) is False                                            # once per process

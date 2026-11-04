"""pf_idx_heads (prefill indexer first pass on a query-head subset): the flag stays today's path at 0, every head is the exact pass, the kept scores are
the exact ones whatever the subset, and the recall counter of pf_idx_heads_check reports what the served indices kept of the exact top-k.

The glue (selection, packing, ranking, rescoring, counting) runs on the CPU with the numpy score mirror; the packed int8 kernel is checked against the
same numpy score mirror on the Mac (tests 14-15, skipped without Metal + matmul2d). tools/pfheads_bench.py measures the ms, tools/pfheads_recall.py the
recall on the real dumps in /tmp/idxcap."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL          # noqa: E402
from kyojin_mlx import idx_topk as IT       # noqa: E402
from kyojin_mlx import pf_flat as PF        # noqa: E402
from kyojin_mlx import pfroof2 as PR        # noqa: E402
from test_pf_flat import _fp4_like          # noqa: E402

NH, D, BS, TOPK_LAYER = 32, 128, 8, 16
metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def cpu_glue(monkeypatch):
    """Today's exact chain on the CPU: the numpy score mirror, the block-max path on from 0 keys, no capture, no head subset."""
    monkeypatch.setattr(PF, "METAL_OPT", False)
    monkeypatch.setattr(PF, "BMAX", True)
    monkeypatch.setattr(PF, "BMAX_MIN", 0)
    monkeypatch.setattr(PF, "GSCORE", True)
    monkeypatch.setattr(PF, "HEADS", 0)
    monkeypatch.setattr(PF, "HEADS_M", 2048)
    monkeypatch.setattr(PF, "HEADS_CHECK", False)
    monkeypatch.setattr(PF, "CAPTURE", False)


def make(T, Tc, seed=0, quant=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, NH, D)).astype(np.float16)
    k = rng.standard_normal((1, Tc, D)).astype(np.float16)
    w = (rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16)
    if quant:
        q, k = (np.round(q * 2) / 2).astype(np.float16), (np.round(k * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


def scores(q, k, w, start, ratio):
    return mx.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, ratio))


def exact(q, k, w, start, ratio, topk):
    return np.array(IT.topk(scores(q, k, w, start, ratio), topk))


def case(T=32, Tc=3000, topk=16, ratio=4, seed=0, quant=False):
    q, k, w = make(T, Tc, seed, quant)
    start = ratio * (Tc - T)
    return q, k, w, start, ratio, exact(q, k, w, start, ratio, topk), topk


def nblk_of(Tc):
    return -(-Tc // BS)


# ----------------------------------------------------------------------------------------------------------------- flag 0 and all heads
def test_flag_zero_keeps_todays_path(monkeypatch):
    """pf_idx_heads=0 never selects or packs a head: the same call, the same indices, no counter."""
    q, k, w, start, ratio, ref, topk = case()
    monkeypatch.setattr(PF, "head_sel", lambda *a, **kw: pytest.fail("pf_idx_heads=0 selected a head subset"))
    before = dict(PF.HITS)
    assert np.array_equal(np.array(PF.plain_topk(q, k, w, start, ratio, topk, 32)), ref)
    assert PF.HITS["heads_plain"] == before["heads_plain"] and PF.HITS["heads_source"] == before["heads_source"]
    assert PF.heads_n() == 0


@pytest.mark.parametrize("n", [1, 2, 3, 12, 20, 24, 31, 33, 99])
def test_a_head_count_that_does_not_split_the_tile_keeps_32(monkeypatch, n):
    """32 head slots split into 32 / N rows only for N = 4, 8, 16; 32 is the whole pass and anything else is today (no silent rounding)."""
    monkeypatch.setattr(PF, "HEADS", n)
    assert PF.heads_n() == 0 and not PF.heads_sub(0)
    q, k, w, start, ratio, ref, topk = case()
    assert np.array_equal(np.array(PF.plain_topk(q, k, w, start, ratio, topk, 32)), ref)


@pytest.mark.parametrize("n,m", [(32, 0), (32, 64), (32, 2048), (16, 2048)])
def test_all_heads_is_the_exact_pass(monkeypatch, n, m):
    """N = all heads in the first pass, whatever the candidate width (and a subset whose candidate set covers every block): the same top-k as the exact
    chain, because the block maxima are exact and the wider candidate set only adds blocks the exact pass already ranked below the top-k."""
    monkeypatch.setattr(PF, "HEADS", n)
    monkeypatch.setattr(PF, "HEADS_M", m)
    q, k, w, start, ratio, ref, topk = case(T=32, Tc=3000, topk=16, ratio=4, seed=1)
    got = PF.plain_topk(q, k, w, start, ratio, topk, 32)
    mx.eval(got)
    assert np.array_equal(np.array(got), ref)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_every_block_kept_is_exact_whatever_the_subset(monkeypatch, n):
    """A subset only narrows the candidate set: with every block kept and the survivors rescored on all 32 heads the answer is the exact top-k."""
    q, k, w, start, ratio, ref, topk = case(T=32, Tc=3000, topk=16, ratio=4, seed=2)
    monkeypatch.setattr(PF, "HEADS", n)
    monkeypatch.setattr(PF, "HEADS_M", nblk_of(3000))
    got = PF.plain_topk(q, k, w, start, ratio, topk, 32)
    mx.eval(got)
    assert np.array_equal(np.array(got), ref)


def test_kept_scores_are_exact(monkeypatch):
    """With a narrow candidate set the answer differs from the exact top-k, but it is the exact top-k of the keys the subset kept: the selection ranks
    the surviving keys on their full-width 32-head scores, so no kept key that scores higher is left out."""
    T, Tc, topk, ratio = 24, 3000, 16, 4
    q, k, w, start, ratio, ref, topk = case(T=T, Tc=Tc, topk=topk, ratio=ratio, seed=3)
    m = 20                                                                    # above topk: pf_idx_heads_m widens, it never narrows
    monkeypatch.setattr(PF, "HEADS", 16)
    monkeypatch.setattr(PF, "HEADS_M", m)
    got = np.array(PF.plain_topk(q, k, w, start, ratio, topk, T))
    s = np.array(scores(q, k, w, start, ratio))
    assert not np.array_equal(got, ref)                                        # the candidate set is approximate
    sel = np.array(PF.head_sel(q, w, 16))
    bm = PF.bmax_scores(q, k, w, start, ratio, 0, T, sel)
    ids = np.array(PF.pick_blocks(bm, (start + mx.arange(T) + 1) // ratio, m, False))
    for t in range(T):
        clen = (start + t + 1) // ratio
        cand = np.concatenate([b * BS + np.arange(BS) for b in ids[t] if b >= 0]).astype(np.int64)
        cand = cand[cand < min(clen, Tc)]
        want = np.sort(cand[np.argsort(-s[t, cand], kind="stable")[:topk]])
        assert got[t].tolist() == want.tolist(), t


# ----------------------------------------------------------------------------------------------------------------- recall counter
def _overlap(got, ref):
    g, r = np.array(got), np.array(ref)
    return int((g[:, :, None] == r[:, None, :]).any(-1).sum()), int(r.size)


def test_recall_counter_matches_the_overlap(monkeypatch):
    """pf_idx_heads_check runs the exact pass on the same chunk and counts what the served indices kept, per layer and summed."""
    T, Tc, topk, ratio = 24, 6000, 16, 4
    q, k, w, start, ratio, ref, topk = case(T=T, Tc=Tc, topk=topk, ratio=ratio, seed=4)
    monkeypatch.setattr(PF, "HEADS", 16)
    monkeypatch.setattr(PF, "HEADS_M", 32)
    monkeypatch.setattr(PF, "HEADS_CHECK", True)
    before = {k_: v for k_, v in PF.HITS.items() if isinstance(v, (int, float))}
    got = PF.plain_topk(q, k, w, start, ratio, topk, T, key=7)
    hit, tot = _overlap(got, ref)
    assert PF.HITS["heads_recall_hit"] == before["heads_recall_hit"] + hit
    assert PF.HITS["heads_recall_total"] == before["heads_recall_total"] + tot
    assert PF.HITS["heads_hit_l7"] == hit and PF.HITS["heads_total_l7"] == tot
    assert abs(PF.heads_recall_pct() - 100.0 * PF.HITS["heads_recall_hit"] / PF.HITS["heads_recall_total"]) < 1e-9
    assert 0 < hit <= tot


def test_recall_is_full_when_the_candidate_set_covers_every_block(monkeypatch):
    T, Tc, topk, ratio = 16, 3000, 16, 4
    q, k, w, start, ratio, ref, topk = case(T=T, Tc=Tc, topk=topk, ratio=ratio, seed=5)
    monkeypatch.setattr(PF, "HEADS", 16)
    monkeypatch.setattr(PF, "HEADS_M", nblk_of(Tc))
    monkeypatch.setattr(PF, "HEADS_CHECK", True)
    before = dict(PF.HITS)
    got = PF.plain_topk(q, k, w, start, ratio, topk, T, key=0)
    assert np.array_equal(np.array(got), ref)
    assert PF.HITS["heads_recall_hit"] - before["heads_recall_hit"] == got.size
    assert PF.HITS["heads_total_l0"] - before.get("heads_total_l0", 0) == got.size
    assert PF.HITS["heads_hit_l0"] - before.get("heads_hit_l0", 0) == got.size


def test_the_check_is_off_by_default(monkeypatch):
    q, k, w, start, ratio, ref, topk = case()
    monkeypatch.setattr(PF, "HEADS", 16)
    before = dict(PF.HITS)
    PF.plain_topk(q, k, w, start, ratio, topk, 32)
    assert PF.HITS["heads_recall_total"] == before["heads_recall_total"]
    assert PF.heads_recall_pct() == (100.0 * PF.HITS["heads_recall_hit"] / PF.HITS["heads_recall_total"] if PF.HITS["heads_recall_total"] else 0.0)


# ----------------------------------------------------------------------------------------------------------------- selection and packing
def test_head_sel_picks_the_largest_weighted_query_norm(monkeypatch):
    T, Tc = 20, 512
    q, k, w = make(T, Tc, seed=6)
    for nh in (4, 8, 16):
        got = np.array(PF.head_sel(q, w, nh))
        met = np.abs(np.array(w[0]).astype(np.float32)) * np.linalg.norm(np.array(q[0]).astype(np.float32), axis=-1)
        want = np.argsort(-met, axis=-1)[:, :nh]
        assert np.array_equal(np.sort(got, 1), np.sort(want, 1))


@pytest.mark.parametrize("T,nh", [(8, 16), (7, 16), (8, 8), (5, 8), (1, 4), (9, 4)])
def test_head_rows_packs_rows_in_tile_order(monkeypatch, T, nh):
    """32 / nh consecutive rows of qs / ws fill one 32-slot head tile; a tail group repeats the last row so a row block of any length can be launched."""
    Tc = 256
    q, k, w = make(T, Tc, seed=7)
    rng = np.random.default_rng(7)
    qi = mx.array(rng.integers(-120, 120, (T * NH, D)).astype(np.int8))
    w8 = mx.array(rng.random((T, NH)).astype(np.float32))
    sel = mx.array(np.stack([rng.permutation(NH)[:nh] for _ in range(T)]).astype(np.int32))
    qs, ws = PF.head_rows(qi, w8, sel, nh)
    mx.eval(qs, ws)
    K = 32 // nh
    groups = -(-T // K)
    assert qs.shape == (groups * K * nh, D) and ws.shape == (groups * K * nh,)
    a, b = np.array(qs).reshape(groups, K, nh, D), np.array(ws).reshape(groups, K, nh)
    Q, W, S = np.array(qi).reshape(T, NH, D), np.array(w8), np.array(sel)
    for g in range(groups):
        for sl in range(K):
            r = min(g * K + sl, T - 1)                                         # a tail group repeats the last row
            assert np.array_equal(a[g, sl], Q[r][S[r]])
            assert np.array_equal(b[g, sl], W[r][S[r]])


def test_heads_pick_widens_but_never_narrows(monkeypatch):
    monkeypatch.setattr(PF, "HEADS_M", 0)
    assert PF.heads_pick(544) == 544 and PF.heads_pick(16) == 16
    monkeypatch.setattr(PF, "HEADS_M", 2048)
    assert PF.heads_pick(544) == 2048 and PF.heads_pick(4096) == 4096


def test_pair_kernel_source_keeps_the_tile_and_adds_one_sum_per_row():
    """The packed kernel is the int8 block-max kernel with the 32 head slots read as 32 / nh rows: the same 32-slot matmul2d tile, one head sum and one
    causal length per row, one block-max write per row."""
    base = PF.i8_source()
    for nh, K in ((16, 2), (8, 4)):
        s = PF.i8_rows_source(nh)
        for keep in ("matmul2d_descriptor(32, 32, 128", "threadgroup int8_t Qs[32 * 128];", "tensor<threadgroup int8_t",
                     "= float(cT[i]);", "scr[sg][uint(ix[1]) * 33 + uint(ix[0])] = float(cT[i]);"):
            assert keep in s and keep in base, keep
        assert f"float v[{K}];" in s and f"for (uint sl = 0; sl < {K}u; ++sl)" in s
        assert f"for (uint h = 0; h < {nh}u; ++h)" in s and f"sl * {nh}u + h" in s
        assert f"(start + row0 + sl + 1) / ratio" in s and f"row0 = t0 + t * {K}u" in s
        assert f"ulong(t * {K}u + sl)" in s                                       # the output row is launch-relative, row0 stays absolute
        assert "const uint clen" not in s
        assert s.count("float v[") == 1 and s.count("S[ulong(") == 1


# ----------------------------------------------------------------------------------------------------------------- source layer
def test_source_layer_every_block_kept_is_exact(monkeypatch):
    """The candidate source on a subset with every block kept: the same kept blocks and the same top-k as the all-head pass."""
    T, Tc, topk, ratio, cand = 24, 3000, 16, 4, 64
    q, k, w = make(T, Tc, seed=8)
    start = ratio * (Tc - T)
    a, ia = PF.source_topk(q, k, w, start, ratio, topk, T, cand)
    monkeypatch.setattr(PF, "HEADS", 16)
    monkeypatch.setattr(PF, "HEADS_M", nblk_of(Tc))
    before = dict(PF.HITS)
    b, ib = PF.source_topk(q, k, w, start, ratio, topk, T, cand)
    mx.eval(a, ia, b, ib)
    assert PF.HITS["heads_source"] == before["heads_source"] + 1
    assert np.array_equal(np.array(a), np.array(b)) and np.array_equal(np.array(ia), np.array(ib))


def test_source_layer_recall_counter(monkeypatch):
    T, Tc, topk, ratio, cand = 16, 6000, 16, 4, 64
    q, k, w = make(T, Tc, seed=9)
    start = ratio * (Tc - T)
    ref = np.array(IT.topk(scores(q, k, w, start, ratio), topk))
    monkeypatch.setattr(PF, "HEADS", 16)
    monkeypatch.setattr(PF, "HEADS_M", 32)
    monkeypatch.setattr(PF, "HEADS_CHECK", True)
    before = dict(PF.HITS)
    got = PF.source_topk(q, k, w, start, ratio, topk, T, cand, key=4)[0]
    hit, tot = _overlap(got, ref)
    assert PF.HITS["heads_hit_l4"] - before.get("heads_hit_l4", 0) == hit
    assert PF.HITS["heads_total_l4"] - before.get("heads_total_l4", 0) == tot
    assert hit > 0


@metal
def test_indexer_dispatch_reaches_the_lever(monkeypatch):
    """`ds_pf_idx._impl_indexer` takes the subset through the flags and hands the layer index to the recall counter: all heads is the exact run, a
    subset stays inside the stock contract (ascending absolute positions inside the causal range)."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    import test_idx_topk as TI
    I = TI.I

    def scores_range(q, keys, w, start, ratio, t0, n, max_elems=1 << 26, n0=0, kq=None):
        return mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0=t0, n0=n0))

    monkeypatch.setattr(I, "scores_range", scores_range)
    monkeypatch.setattr(I, "fq4_rot", lambda q, cos, sin, rd: q)
    monkeypatch.setattr(I, "ENABLED", True)
    monkeypatch.setattr(PF, "METAL_OPT", True)
    monkeypatch.setattr(PF, "I8", True)
    monkeypatch.setattr(PF, "_I8_GRID", True)
    idx, cfg = TI.make_indexer(2, -1)                                       # a plain layer: no candidate source, no consumers
    idx.candidate_block_size = BS                                           # the block-max path's block size
    fused = I._impl_indexer(TI.L.Indexer.__call__)
    rng = np.random.default_rng(9)
    T, start = 96, 400
    n = (start + T) // 2
    x = mx.array(rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)).astype(mx.float16)
    qr = mx.array(rng.standard_normal((1, T, cfg.q_lora_rank)).astype(np.float32)).astype(mx.float16)
    keys = mx.array(_fp4_like(rng, (n, D)))[None]
    monkeypatch.setenv(IT.QBLOCK_ENV, "32")
    got, before = {}, dict(PF.HITS)
    for tag, heads, m in (("off", 0, 0), ("all", 32, 0), ("sub", 16, 8)):
        monkeypatch.setattr(PF, "HEADS", heads)
        monkeypatch.setattr(PF, "HEADS_M", m)
        cache = TI.SimpleNamespace(index_k=keys, candidates=None)
        got[tag] = np.array(fused(idx, x, qr, None, start, 0, cache))[0]
    assert np.array_equal(got["all"], got["off"])                           # all heads: the exact run, bit for bit
    assert PF.HITS["heads_plain"] - before["heads_plain"] == 1              # only the subset arm packed a head subset
    topk = min(TOPK_LAYER, n)
    assert got["sub"].shape == (T, topk) and np.all(np.diff(got["sub"], axis=1) >= 0)
    clen = (start + np.arange(T) + 1) // 2
    assert (got["sub"] < clen[:, None]).all()
    assert _overlap(got["sub"], got["off"])[0] / got["off"].size >= 0.5


# ----------------------------------------------------------------------------------------------------------------- flags
def test_flags_are_runtime_and_map_onto_pf_flat():
    assert FL.FLAGS["pf_idx_heads"].default == "8" and FL.FLAGS["pf_idx_heads"].kind == "int" and FL.FLAGS["pf_idx_heads"].runtime
    assert FL.FLAGS["pf_idx_heads"].attrs == (("pf_flat", "HEADS", int),)
    assert FL.FLAGS["pf_idx_heads_m"].default == "1024" and FL.FLAGS["pf_idx_heads_m"].attrs == (("pf_flat", "HEADS_M", int),)
    assert FL.FLAGS["pf_idx_heads_check"].default == "0" and FL.FLAGS["pf_idx_heads_check"].attrs == (("pf_flat", "HEADS_CHECK", FL._on),)
    h, hm, hc = PF.HEADS, PF.HEADS_M, PF.HEADS_CHECK
    try:
        FL.set("pf_idx_heads", 16)
        FL.set("pf_idx_heads_m", 1024)
        FL.set("pf_idx_heads_check", 1)
        assert PF.HEADS == 16 and PF.HEADS_M == 1024 and PF.HEADS_CHECK is True
        assert PF.heads_n() == 16 and PF.heads_sub(PF.heads_n())
        FL.set("pf_idx_heads", 0)
        assert PF.HEADS == 0 and PF.heads_n() == 0
    finally:
        FL.set("pf_idx_heads", h)
        FL.set("pf_idx_heads_m", hm)
        FL.set("pf_idx_heads_check", int(hc))


# ----------------------------------------------------------------------------------------------------------------- Metal
def _grid_case(T, Tc, seed):
    rng = np.random.default_rng(seed)
    q = mx.array(_fp4_like(rng, (T, NH, D)))[None]
    k = mx.array(_fp4_like(rng, (Tc, D)))[None]
    w = mx.array((rng.standard_normal((T, NH)) * .05).astype(np.float16))[None]
    return q, k, w


@metal
@pytest.mark.parametrize("nh", [8, 16])
def test_i8_pair_kernel_is_the_block_max_of_the_head_subset(monkeypatch, nh):
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no matmul2d")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    T, Tc = 48, 4096 + 37                                                   # a tail tile off the block grid
    q, k, w = _grid_case(T, Tc, 10)
    start, ratio = 2 * Tc, 2
    sel = PF.head_sel(q, w, nh)
    monkeypatch.setattr(PF, "METAL_OPT", False)                              # bmax_scores takes the subset on the numpy mirror only
    ref = PF.bmax_scores(q, k, w, start, ratio, 0, T, np.array(sel))
    monkeypatch.setattr(PF, "METAL_OPT", True)
    ki, ke = PF.i8pack(k[0])
    qi, qe = PF.i8pack(q[0].reshape(-1, D))
    w8 = w[0].astype(mx.float32) * qe.reshape(T, NH)
    qs, ws = PF.head_rows(qi, w8, sel, nh)
    mx.eval(ref, qs, ws)
    for t0, n in ((0, T), (5, T - 5), (T - 1, 1)):
        got = PF.bmax_i8_rows(qs, ki, ws, ke, start, ratio, t0, n, nh)
        g, r = np.array(got), np.array(ref)[t0:t0 + n]
        assert np.array_equal(np.isinf(g), np.isinf(r))
        f = np.isfinite(r)
        assert np.allclose(g[f], r[f], rtol=3e-2, atol=3e-2)
        for a, b in zip(r, g):                                              # the best block of a row survives the int8 rounding
            assert np.argmax(a) in np.argsort(-b)[:8]


@metal
@pytest.mark.parametrize("nh", [8, 16])
def test_i8_heads_plain_topk_matches_the_exact_pass(monkeypatch, nh):
    """On the int8/FP4 grid, every block kept: the packed kernel + the exact rescore give the exact all-head top-k (and the check counter reads 100 %)."""
    from kyojin_mlx import metal_nax as NX
    if not NX.nax_available():
        pytest.skip("no matmul2d")
    monkeypatch.setattr(PF, "METAL_OPT", True)
    monkeypatch.setattr(PF, "I8", True)
    monkeypatch.setattr(PF, "_I8_GRID", True)
    T, Tc, topk = 1024, 16384, 64
    q, k, w = _grid_case(T, Tc, 11)
    start, ratio = Tc - T, 1
    ref = np.array(IT.topk(scores(q, k, w, start, ratio), topk))
    monkeypatch.setattr(PF, "HEADS", 0)
    assert np.array_equal(np.array(PF.plain_topk(q, k, w, start, ratio, topk, T)), ref)
    monkeypatch.setattr(PF, "HEADS", nh)
    monkeypatch.setattr(PF, "HEADS_M", nblk_of(Tc))
    before = dict(PF.HITS)
    got = np.array(PF.plain_topk(q, k, w, start, ratio, topk, T, key=0))
    assert PF.HITS["heads_plain"] == before["heads_plain"] + 1                # the packed kernel ran
    assert np.array_equal(got, ref)
    monkeypatch.setattr(PF, "HEADS_CHECK", True)
    before = dict(PF.HITS)
    PF.plain_topk(q, k, w, start, ratio, topk, T, key=0)
    hit = PF.HITS["heads_recall_hit"] - before["heads_recall_hit"]
    tot = PF.HITS["heads_recall_total"] - before["heads_recall_total"]
    assert tot == T * topk and hit == tot                                      # every block kept: nothing to miss

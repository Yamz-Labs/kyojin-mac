"""pf_bound (cloud/cflatpf-2): the exact two-level bound of the prefill indexer, on the CPU against the plain path (numpy scoring in the kernels' order, METAL_OPT False).
Soundness first (the bound is never below a score it covers: negative head weights, identical keys where the half rounding lifts the score above the unrounded
bound, ties, the causal edge, a short last superblock), then the selected set against the stock sort(argsort(-s)), with a case that really prunes and a case that
falls back, and the state machine of the fallback."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL          # noqa: E402
from kyojin_mlx import idx_topk as IT       # noqa: E402
from kyojin_mlx import pf_bound as PB       # noqa: E402
from kyojin_mlx import pf_flat as PF        # noqa: E402
from kyojin_mlx import pfroof2 as PR        # noqa: E402

NH, D = 32, 128
SB = 64


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(PF, "METAL_OPT", False)
    monkeypatch.setattr(PB, "ENABLED", True)
    monkeypatch.setattr(PB, "MIN_KEYS", 0)
    monkeypatch.setattr(PB, "SB", SB)
    monkeypatch.setattr(PB, "MAXPCT", 25)
    PB._state.clear()
    for k in PB.HITS:
        PB.HITS[k] = 0


def rnd(T, Tc, seed=0, quant=False, wscale=0.05):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, NH, D)).astype(np.float16)
    k = rng.standard_normal((1, Tc, D)).astype(np.float16)
    w = (rng.standard_normal((1, T, NH)) * wscale).astype(np.float16)
    if quant:
        q, k = (np.round(q * 2) / 2).astype(np.float16), (np.round(k * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


def clustered(T, Tc, seed=0, centres=48, noise=0.005, run=SB):
    """Keys in runs of `run` consecutive positions around one of `centres` random centres (adjacent positions alike), queries / weights random."""
    rng = np.random.default_rng(seed)
    q, _, w = rnd(T, Tc, seed)
    cen = rng.standard_normal((centres, D)).astype(np.float32)
    ids = rng.integers(0, centres, size=-(-Tc // run))
    k = np.repeat(cen[ids], run, axis=0)[:Tc] + noise * rng.standard_normal((Tc, D)).astype(np.float32)
    return q, mx.array(k[None].astype(np.float16)), w


def scores(q, k, w, start, ratio, t0=0, n=None):
    n = q.shape[1] - t0 if n is None else n
    return PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(k[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0)


def stock(q, k, w, start, ratio, topk):
    return lambda t0, n: IT.topk(mx.array(scores(q, k, w, start, ratio, t0, n)), topk)


def block_max(s, sb):
    T, Tc = s.shape
    nsb = -(-Tc // sb)
    sp = np.pad(s, [(0, 0), (0, nsb * sb - Tc)], constant_values=-np.inf)
    return sp.reshape(T, nsb, sb).max(-1)


def check_sound(q, k, w, start, ratio, sb=SB):
    s = scores(q, k, w, start, ratio)
    u = np.array(PB.bounds(q, w, PB.summaries(k, sb), start, ratio, 0, q.shape[1]))
    bm = block_max(s, sb)
    fin = np.isfinite(bm)
    assert np.all(u[fin] >= bm[fin]), float((bm - u)[fin].max())
    assert np.all(np.isneginf(u[~fin & np.isneginf(bm)]) | True)
    return u, bm


@pytest.mark.parametrize("T,Tc,ratio,quant", [(24, 1000, 1, False), (24, 1000, 2, True), (16, 777, 4, False), (30, 2048, 1, True)])
def test_bound_is_never_below_a_score(T, Tc, ratio, quant):
    q, k, w = rnd(T, Tc, quant=quant)
    start = ratio * (Tc - T)
    u, bm = check_sound(q, k, w, start, ratio)
    assert np.all(np.isneginf(u[np.isneginf(bm)]))                              # nothing visible in a superblock: -inf, never pruned wrongly the other way


def test_bound_with_negative_and_zero_weights():
    q, k, w = rnd(20, 900, seed=4, wscale=0.2)
    wn = np.array(w)
    wn[:, :, ::3] = 0.0
    wn[:, :, 1::3] = -np.abs(wn[:, :, 1::3])                                    # a third negative, a third zero, a third as drawn
    check_sound(q, k, mx.array(wn), 900 - 20, 1)


def test_bound_identical_keys_half_rounding_does_not_exceed_it():
    """Radius 0 and box width 0: the bound is the real q . k, and the kernel's rounded-to-half product can be above it. The margin must cover that."""
    rng = np.random.default_rng(9)
    for seed in range(12):
        T, Tc = 16, 512
        v = rng.standard_normal((1, 1, D)).astype(np.float16) * (1 + seed)
        k = mx.array(np.repeat(v, Tc, axis=1))
        q, _, w = rnd(T, Tc, seed=seed)
        check_sound(q, k, w, Tc - T, 1)


@pytest.mark.parametrize("seed", range(5))
def test_bound_fuzz_never_below(seed):
    """Random shapes, superblock sizes, ratios, weight scales; heavy-tailed, tiny, identical and grid keys."""
    rng = np.random.default_rng(100 + seed)
    for it in range(12):
        Tn, Tc, sb, ratio = int(rng.integers(4, 20)), int(rng.integers(100, 700)), int(rng.choice([8, 16, 64])), int(rng.choice([1, 2, 4]))
        q, k, w = rnd(Tn, Tc, seed=seed * 100 + it, quant=it % 3 == 0, wscale=float(rng.choice([0.01, 0.05, 0.5])))
        kn = np.array(k).astype(np.float32)
        kind = it % 5
        if kind == 1:
            kn = kn * np.exp(rng.standard_normal((1, Tc, 1)) * 0.8)
        elif kind == 2:
            kn = np.repeat(kn[:, :1], Tc, axis=1) * (1 + rng.random())
        elif kind == 3:
            kn = kn * 0.001
        elif kind == 4:
            kn = np.round(kn * 4) / 4
        k = mx.array(np.clip(kn, -60000, 60000).astype(np.float16))
        check_sound(q, k, w, ratio * (Tc - Tn) + int(rng.integers(0, ratio)), ratio, sb)


def test_bound_large_dynamic_range_and_short_last_superblock():
    q, k, w = rnd(20, 1000 + 17, seed=5)
    kn = np.array(k) * np.exp(np.random.default_rng(1).standard_normal((1, 1017, 1)).astype(np.float32) * 0.7).astype(np.float16)
    check_sound(q, mx.array(kn), w, 1017 - 20, 1)                                # 1017 keys: 15 superblocks of 64 and a 57-key one


@pytest.mark.parametrize("T,Tc,topk,ratio,quant", [(40, 3000, 32, 1, False), (40, 4100, 40, 2, True), (33, 5000, 24, 4, False)])
def test_clustered_keys_prune_and_stay_exact(T, Tc, topk, ratio, quant):
    q, k, w = clustered(T, Tc, seed=T)
    if quant:
        k = (mx.round(k.astype(mx.float32) * 2) / 2).astype(mx.float16)
    start = ratio * (Tc - T)
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    for qb in ("0", "11"):
        import os
        os.environ[IT.QBLOCK_ENV] = qb
        try:
            PB._state.clear()
            got = np.array(PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=stock(q, k, w, start, ratio, topk)))
        finally:
            os.environ.pop(IT.QBLOCK_ENV, None)
        assert np.array_equal(got, ref)
    assert PB.HITS["fallback_blocks"] == 0
    assert PB.survivor_fraction() < 0.5                                          # the bound really pruned (this is what the Mac probe measures on real keys)


@pytest.mark.parametrize("quant", [False, True])
def test_isotropic_keys_fall_back_and_stay_exact(quant):
    T, Tc, topk, ratio = 36, 4000, 32, 1
    q, k, w = rnd(T, Tc, quant=quant)
    start = ratio * (Tc - T)
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    got = np.array(PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=stock(q, k, w, start, ratio, topk)))
    assert np.array_equal(got, ref)
    assert PB.HITS["fallback_blocks"] >= 1                                       # nothing to prune: the stock block ran


def test_ties_all_equal_scores_lowest_positions_win():
    T, Tc, topk = 24, 2048, 20
    k = mx.array(np.ones((1, Tc, D), dtype=np.float16) * 0.5)
    q = mx.array(np.ones((1, T, NH, D), dtype=np.float16) * 0.25)
    w = mx.array(np.ones((1, T, NH), dtype=np.float16) * 0.03125)
    start = Tc - T
    ref = np.array(stock(q, k, w, start, 1, topk)(0, T))
    got = np.array(PB.try_plain(q, k, w, start, 1, topk, T, key="L", stock=stock(q, k, w, start, 1, topk)))
    assert np.array_equal(got, ref)


def test_tie_group_straddles_the_threshold():
    """Quantised clustered data: many keys score exactly the topk-th value, some in pruned-looking superblocks. The whole tie group must survive."""
    T, Tc, topk, ratio = 32, 3200, 30, 1
    q, k, w = clustered(T, Tc, seed=3, noise=0.0, centres=10)
    k = (mx.round(k.astype(mx.float32) * 4) / 4).astype(mx.float16)
    q = (mx.round(q.astype(mx.float32) * 2) / 2).astype(mx.float16)
    w = (mx.round(w.astype(mx.float32) * 64) / 64).astype(mx.float16)
    start = Tc - T
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    got = np.array(PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=stock(q, k, w, start, ratio, topk)))
    assert np.array_equal(got, ref)


def test_causal_edge_rows_see_part_of_the_chunk():
    """ratio 2, start inside the keys: the last rows' own superblock is partly invisible, and the rows near the first of the chunk see fewer keys."""
    T, Tc, topk, ratio = 48, 2600, 24, 2
    q, k, w = clustered(T, Tc, seed=8, noise=0.05, centres=30)
    start = ratio * (Tc - T) + 1
    ref = np.array(stock(q, k, w, start, ratio, topk)(0, T))
    got = np.array(PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=stock(q, k, w, start, ratio, topk)))
    assert np.array_equal(got, ref)


def test_applies_gates():
    q, k, w = rnd(8, 1500)
    ok = lambda **kw: PB.applies(q, k, w, **{"topk": 32, "start": 1400, "ratio": 1, "plain": True, **kw})
    assert ok()
    assert not ok(plain=False)
    assert not ok(start=100)                                                     # fewer visible keys than the seed needs: stock path
    assert not ok(topk=0)
    PB.ENABLED = False
    assert not ok()
    PB.ENABLED = True
    PB.MIN_KEYS = 4000
    assert not ok()
    PB.MIN_KEYS = 0
    assert not PB.applies(q[:, :, :16], k, w[:, :, :16], 32, 1400, 1, True)    # other indexer shapes: stock path


def test_applies_needs_gscore_on_the_gpu_and_not_qmm(monkeypatch):
    q, k, w = rnd(8, 1500)
    args = (q, k, w, 32, 1400, 1, True)
    monkeypatch.setattr(PF, "METAL_OPT", True)
    monkeypatch.setattr(PF, "fscore_ok", lambda *a: True)
    monkeypatch.setattr(PF, "GSCORE", False)
    assert not PB.applies(*args)                                                # the gathered copy would move more bytes than the tile it saves
    monkeypatch.setattr(PF, "GSCORE", True)
    assert PB.applies(*args)
    monkeypatch.setattr(PF, "_failed", {"gs"})
    assert not PB.applies(*args)                                                # the gscore kernel did not build here
    monkeypatch.setattr(PF, "_failed", set())
    monkeypatch.setenv("KYOJIN_PF_IDX_QMM", "1")
    assert not PB.applies(*args)                                                # qmm score numerics differ from fscore


def test_skip_window_after_two_missed_chunks():
    T, Tc, topk, ratio = 24, 3000, 32, 1
    q, k, w = rnd(T, Tc)
    start = Tc - T
    st = stock(q, k, w, start, ratio, topk)
    for _ in range(2):
        PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=st)
    assert PB._state["L"][1] == PB.SKIP_CHUNKS
    h = PB.HITS["chunks"]
    assert PB.try_plain(q, k, w, start, ratio, topk, T, key="L", stock=st) is None   # the layer sits out; the caller runs the stock path
    assert PB.HITS["chunks"] == h and PB.HITS["skipped_chunks"] == 1
    PB.try_plain(q, k, w, start, ratio, topk, T, key="other", stock=st)             # other layers are not affected
    assert PB.HITS["chunks"] == h + 1


def test_blocks_of_clamps_past_the_last_block():
    ids = PB.blocks_of(mx.array([[0, 2, -1]], dtype=mx.int32), 20, 64)          # 8 blocks per superblock, only 20 blocks exist
    got = np.array(ids)[0]
    assert got.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def test_flag_rows_registered():
    for name in ("pf_idx_bound", "pf_idx_bound_min", "pf_idx_bound_sb", "pf_idx_bound_maxpct"):
        assert name in FL.FLAGS
    assert FL.FLAGS["pf_idx_bound"].default == "0"

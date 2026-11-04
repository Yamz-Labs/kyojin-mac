"""Decode attention core: the indexer score / fused-select levers (perf/decattn).

  * idxfuse.fused == index_scores -> index_topk_rt [-> candidate_mask_rt] bit for bit, including the 4096-key
    case the lever was parked on, the source -> consumer candidate chain and repeated calls.
  * idxfuse.scores_fast == index_scores bit for bit (same FMA and head order) and feeds the same exact select.
  * dsv4_core.core with idx_fuse / idx_score_fast is array_equal to the chain on the synthetic served-shape model.
"""
import importlib
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

from kyojin_mlx import dsv4_core as C  # noqa: E402
from kyojin_mlx import idxfuse as IF  # noqa: E402

NH, DI, K = 32, 128, 512


def bits(a):
    a = np.array(a.astype(mx.float32))
    return a.view(np.uint32)


def make(rng, N, Tm, dt, quant=True):
    q = rng.standard_normal((N, NH, DI)).astype(np.float32)
    k = (rng.standard_normal((N, Tm, DI)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((N, NH)) * 0.05).astype(np.float32)
    if quant:                                   # coarse values: ties, exact zeros
        q = np.round(q * 2) / 2
        k = np.round(k * 2) / 2
        k[:, 3::7] = 0.0
        w = np.round(w * 20) / 20
    return mx.array(q).astype(dt), mx.array(k).astype(dt), mx.array(w).astype(dt)


# ----------------------------------------------------------------------------------------------------------------- fused select
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("Tm,lens,kb", [
    (4096, [4096], 0), (4096, [4096], 2048), (4096, [4095, 3000], 2048),
    (2048, [2048, 1], 0), (8192, [8192, 4096], 2048), (300, [5, 300], 0),
])
def test_fused_matches_rt_chain(dt, Tm, lens, kb):
    """The one-launch score+select+mask equals the rt chain bit for bit (the 4096-key case the lever was parked on)."""
    N = len(lens)
    rng = np.random.default_rng(4096 + Tm + N)
    q, keys, w = make(rng, N, Tm, dt)
    ln = mx.array(lens, dtype=mx.int32)
    for r in range(3):                              # the counter buffer resets between calls
        idx, mask, sc = IF.fused(q, keys, w, ln, Tm, K, None, None, kb, 8, lmax=Tm)
        ref_sc = C.index_scores(q, keys, w, ln, Tm, rt=True)
        ref_idx = C.index_topk_rt(ref_sc, ln, K)
        mx.eval(idx, sc, ref_sc, ref_idx)
        assert np.array_equal(np.array(idx), np.array(ref_idx)), r
        for u, L in enumerate(lens):
            assert np.array_equal(bits(sc[u, :L]), bits(ref_sc[u, :L]))
        if kb:
            ref_m = C.candidate_mask_rt(ref_sc, ln, kb, 8)
            mx.eval(mask, ref_m)
            for u, L in enumerate(lens):
                assert np.array_equal(np.array(mask[u, :L]), np.array(ref_m[u, :L]))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_fused_source_then_consumer(dt):
    """Layer 20 (candidate source) -> layer 24 (consumer): the fused mask and the consumer's top-k equal the rt chain."""
    rng = np.random.default_rng(24)
    N, Tm = 2, 4096
    ln = mx.array([Tm, Tm - 1], dtype=mx.int32)
    q, keys, w = make(rng, N, Tm, dt)
    idx, mask, _ = IF.fused(q, keys, w, ln, Tm, K, None, None, 2048, 8, lmax=Tm)
    ref_sc = C.index_scores(q, keys, w, ln, Tm, rt=True)
    ref_idx = C.index_topk_rt(ref_sc, ln, K)
    ref_m = C.candidate_mask_rt(ref_sc, ln, 2048, 8)
    mx.eval(idx, mask, ref_idx, ref_m)
    assert np.array_equal(np.array(idx), np.array(ref_idx))
    for u, L in enumerate([Tm, Tm - 1]):
        assert np.array_equal(np.array(mask[u, :L]), np.array(ref_m[u, :L]))
    cq, ck, cw = make(np.random.default_rng(25), N, Tm, dt)
    got, _, _ = IF.fused(cq, ck, cw, ln, Tm, K, mask, None, 0, 8, lmax=Tm)
    ref = C.index_topk_rt(C.index_scores(cq, ck, cw, ln, Tm, ref_m, rt=True), ln, K)
    mx.eval(got, ref)
    assert np.array_equal(np.array(got), np.array(ref))


# ----------------------------------------------------------------------------------------------------------------- score-only
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("Tm,cand", [(2048, None), (4096, None), (4096, 0.3), (8192, 0.125), (16384, 0.125), (65536, 0.125)])
def test_scores_fast_bit_equal(dt, Tm, cand):
    """Lane-per-head scores are bit-identical to index_scores (same fp32 FMA chain per head, same head sum order)."""
    rng = np.random.default_rng(Tm + 7)
    q, keys, w = make(rng, 1, Tm, dt)
    ln = mx.array([Tm], dtype=mx.int32)
    if cand is None:
        cd = None
    else:                                       # block-constant mask like candidate blocks (8 keys)
        cd = mx.array(np.repeat(rng.random((1, Tm // 8)) < cand, 8, axis=1))
    assert IF.scores_supported(q.shape[1], q.shape[2], q.dtype)
    got = IF.scores_fast(q, keys, w, ln, Tm, cd)
    ref = C.index_scores(q, keys, w, ln, Tm, cd, rt=True)
    mx.eval(got, ref)
    assert np.array_equal(bits(got), bits(ref))
    # and it feeds the same exact selection
    got_idx = C.index_topk_rt(got, ln, K)
    ref_idx = C.index_topk_rt(ref, ln, K)
    mx.eval(got_idx, ref_idx)
    assert np.array_equal(np.array(got_idx), np.array(ref_idx))


def test_scores_fast_masked_skips_are_still_identical():
    """A sparse candidate mask: skipped keys are -inf in both, the kept scores keep their bits."""
    rng = np.random.default_rng(11)
    Tm = 4096
    q, keys, w = make(rng, 1, Tm, mx.float16)
    ln = mx.array([3000], dtype=mx.int32)
    cd = mx.array(np.repeat(rng.random((1, Tm // 8)) < 0.125, 8, axis=1))
    got = IF.scores_fast(q, keys, w, ln, Tm, cd)
    ref = C.index_scores(q, keys, w, ln, Tm, cd, rt=True)
    mx.eval(got, ref)
    assert np.array_equal(bits(got[0, :3000]), bits(ref[0, :3000]))
    assert np.any(np.isneginf(np.array(got))) and np.any(np.isfinite(np.array(got)))


# ----------------------------------------------------------------------------------------------------------------- core wiring
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decstage_fixture as FX                                    # noqa: E402  (after the tests dir is on the path; FX.toks keeps ids in vocab)


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    from kyojin_mlx import archs, rows as RW
    m, _ = FX.build(**FX.REAL)
    RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
    return m


@pytest.mark.parametrize("n", [1, 2, 4])
def test_core_levers_bit_equal(model, n, monkeypatch):
    """dc.core with idx_fuse / idx_score_fast on equals the chain (the levers never change a logit bit)."""
    from kyojin_mlx import rows as RW
    x = mx.array((np.random.default_rng(3).standard_normal((n, 1, 512)) * 0.2).astype(np.float16))
    attn = model.language_model.layers[1].attn

    def run(fuse, fast):
        monkeypatch.setattr(C, "IDX_FUSE", fuse)
        monkeypatch.setattr(C, "IDX_SCORE_FAST", fast)
        s = C.stack_rows(RW.batch([model.make_cache()])[0])
        s.begin([n])
        o = C.core(attn, x, s)
        mx.eval(o)
        return np.array(o)

    ref = run(False, False)
    assert np.array_equal(ref, run(True, False)), "idx_fuse"
    assert np.array_equal(ref, run(False, True)), "idx_score_fast"
    assert np.array_equal(ref, run(True, True)), "both"


@pytest.mark.parametrize("n", [1, 2, 4])
def test_core_runs_with_iscore_mma(model, n, monkeypatch):
    """iscore_mma through dc.core on a filled pool: the served crash was `from . import dsa_decode as D` rebinding core's D (head width), std::bad_cast at q.reshape."""
    from kyojin_mlx import longctx as LC, rows as RW
    monkeypatch.setattr(C, "IDX_FUSE", False)
    monkeypatch.setattr(C, "IDX_SCORE_FAST", False)
    monkeypatch.setattr(C, "ISCORE_MMA_MIN", 0)
    monkeypatch.setattr(C, "RT_SELECT", True)

    def run(mma):
        monkeypatch.setattr(C, "ISCORE_MMA", mma)
        pc = model.make_cache()
        mx.eval(model(mx.array([FX.toks(300, 5)]), cache=pc).logits)
        p = RW.verify(model, RW.batch([pc]), [FX.toks(n, 400)])
        mx.eval(p.logits)
        return np.array(p.logits)

    ref = run(False)
    before = LC.HITS["dec_score_mma"]
    on = run(True)
    assert LC.HITS["dec_score_mma"] > before, "the mma branch did not run"
    assert np.isfinite(on).all() and on.shape == ref.shape                       # values: test_iscore_mma (the mask / scores columns past lens are not read)


@pytest.mark.parametrize("n", [1, 4])
def test_verify_logits_bit_equal(model, n, monkeypatch):
    """A whole rows pass (plain n=1, verify n=4) keeps the chain's logits with both indexer levers on (greedy equal)."""
    from kyojin_mlx import rows as RW

    def run(fuse, fast):
        monkeypatch.setattr(C, "IDX_FUSE", fuse)
        monkeypatch.setattr(C, "IDX_SCORE_FAST", fast)
        cache = RW.batch([model.make_cache()])
        p = RW.verify(model, cache, [FX.toks(n, 5)])
        lg = p.logits
        mx.eval(lg)
        out = np.array(lg)
        p.commit([n - 1])
        return out

    ref = run(False, False)
    assert np.array_equal(ref, run(True, False)), "idx_fuse"
    assert np.array_equal(ref, run(False, True)), "idx_score_fast"
    assert np.array_equal(ref, run(True, True)), "both"


def test_core_still_writes_through_the_ablit_writer(model, monkeypatch):
    """The decensor hook projects after wo_b: the indexer levers must not bypass that writer."""
    from kyojin_mlx import rows as RW
    attn = model.language_model.layers[1].attn
    calls = []
    orig = attn.wo_b
    monkeypatch.setattr(attn, "wo_b", lambda x: (calls.append(1), orig(x))[1])
    monkeypatch.setattr(C, "IDX_FUSE", True)
    monkeypatch.setattr(C, "IDX_SCORE_FAST", True)
    s = C.stack_rows(RW.batch([model.make_cache()])[0])
    s.begin([2])
    x = mx.array(np.zeros((2, 1, 512), np.float16))
    o = C.core(attn, x, s)
    mx.eval(o)
    assert calls, "wo_b (the abliteration writer) was skipped"

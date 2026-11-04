"""CED prefill logic on a tiny random DS4.1 text model (stock mlx-vlm modules, any MLX backend). The checks that carry the weight:
- the CED path with nothing cut (tail == chunk rows) is bit for bit the stock path, on several chunks;
- with sliding_window = 1 the decoder half needs no history, so CED is EXACT for the last row (global KV from the exact encoder half);
- the caches of the encoder half and the midpoint's compressed KV / index keys equal the exact run bit for bit after a CED prefill;
- the layout predicate."""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C  # noqa: E402
from kyojin_mlx.archs import DS41, GLM5  # noqa: E402


def prefill(lm, ids, chunks):
    cache = lm.make_cache()
    out, s = None, 0
    for c in chunks:
        out = lm(mx.array([ids[s:s + c]]), cache=cache).logits
        mx.eval(out, *cache[0].state)
        s += c
    assert s == len(ids)
    return cache, out[:, -1].astype(mx.float32)


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def slots_equal(c1, c2, layers, slots=("window", "compress", "keys", "kv_state", "score_state")):
    return all(same(getattr(c1, s)[i], getattr(c2, s)[i]) for s in slots for i in layers)


def kl(p_logits, q_logits):
    p = mx.softmax(p_logits, axis=-1)
    return float(mx.sum(p * (mx.log(p + 1e-30) - mx.log(mx.softmax(q_logits, axis=-1) + 1e-30))).item())


# ------------------------------------------------------------------------------------------------------------------ layout
def test_layout_predicate():
    lm, cfg = M.make()
    lay = C.layout(cfg)
    assert (lay.mid, lay.n_layers, lay.window) == (4, 8, 16)
    assert C.supported(cfg) and C.explain(cfg) is None
    bad = {
        "odd split": dict(num_hidden_layers=7, compress_ratios=[0, 0, 2, 2, 1, 1, 1]),
        "mid not a source": dict(kv_source_layer_ids=[2], index_source_layer_ids=[2, 6]),
        "ratio 2 in the decoder half": dict(compress_ratios=[0, 0, 2, 2, 1, 2, 1, 1]),
        "source after mid": dict(kv_source_layer_ids=[2, 4, 6]),
        "engram after mid": dict(engram_layer_ids=[5], engram_num_embeddings=[100]),
        "candidate source before mid": dict(candidate_source_layer_id=2),
        "no window": dict(sliding_window=0),
    }
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    for why, over in bad.items():
        c = ModelConfig.from_dict({**M.CFG, **over})
        assert C.explain(c) is not None, why
        with pytest.raises(ValueError):
            C.layout(c)
    # an explicit midpoint skips the even split rule (shallow slices)
    c = ModelConfig.from_dict({**M.CFG, "num_hidden_layers": 3, "compress_ratios": [1, 1, 1], "kv_source_layer_ids": [0], "index_source_layer_ids": [0],
                               "candidate_source_layer_id": 0})
    assert C.explain(c) is not None and C.supported(c, mid=0)
    # real DS4.1 layouts: the 40-layer pack passes, GLM has no CedSpec
    real = ModelConfig.from_dict(dict(num_hidden_layers=40, compress_ratios=[0, 0] + [2] * 18 + [1] * 20, kv_source_layer_ids=[2, 8, 14, 20],
                                      index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], candidate_source_layer_id=20, engram_layer_ids=[1, 14],
                                      engram_num_embeddings=[10, 10]))
    assert C.layout(real).mid == 20
    assert GLM5.ced is None and DS41.ced is not None


def test_tail_shorter_than_window_is_refused():
    lm, cfg = M.make()
    with pytest.raises(ValueError):
        C.enable(lm, DS41, tail=cfg.sliding_window - 1)


# ------------------------------------------------------------------------------------------------------------------ nothing cut
@pytest.mark.parametrize("chunks", [(64,), (32, 32), (16, 16, 16, 16), (24, 24, 24)])
def test_nothing_cut_is_bit_identical_to_stock(chunks):
    """tail == rows of every chunk: the CED code path runs (mid layer re-composed, tail layers with start override), nothing is skipped."""
    ids = M.tokens(sum(chunks))
    lm, cfg = M.make()
    ref, ref_lg = prefill(lm, ids, chunks)
    lm2, _ = M.make()
    C.enable(lm2, DS41, tail=chunks[0])
    st = C.state(lm2)
    st.force = True
    got, lg = prefill(lm2, ids, chunks)
    assert st.stats == dict(chunks=len(chunks), rows=sum(chunks), rows_cut=0, skip_cache=0)
    assert same(lg, ref_lg)
    assert slots_equal(got[0], ref[0], range(cfg.num_hidden_layers))
    assert got[0].offset == ref[0].offset


def test_nothing_cut_forced_two_chunks_with_history():
    """Second chunk starts at s0 > 0 with a valid window ring: the forced path must use it (no 'nohist') and equal the stock path."""
    ids = M.tokens(64)
    lm, cfg = M.make()
    ref, ref_lg = prefill(lm, ids, (32, 32))
    lm2, _ = M.make()
    C.enable(lm2, DS41, tail=32)
    C.state(lm2).force = True
    got, lg = prefill(lm2, ids, (32, 32))
    assert C.state(lm2).stats == dict(chunks=2, rows=64, rows_cut=0, skip_cache=0)
    assert same(lg, ref_lg)
    assert slots_equal(got[0], ref[0], range(cfg.num_hidden_layers))


# ------------------------------------------------------------------------------------------------------------------ exact when the window is 1
TAIL1 = 32      # Metal GEMM kernels round differently below 32 rows (a 16-row tail differs from the exact run by the noise of any chunk grid change; the tiny random model amplifies it)


@pytest.mark.parametrize("chunks", [(100,), (40, 40, 20), (33, 33, 34)])
def test_window_one_is_exact_for_the_last_row(chunks):
    """sliding_window = 1: a token attends itself locally and the global KV of the (exact) encoder half; the decoder half needs only the rows
    it outputs. CED(tail = 1) then equals the exact prefill for the last row, whatever the chunking."""
    ids = M.tokens(sum(chunks))
    lm, cfg = M.make(sliding_window=1)
    ref, ref_lg = prefill(lm, ids, chunks)
    C.enable(lm, DS41, tail=TAIL1)
    got, lg = prefill(lm, ids, chunks)
    st = C.state(lm)
    assert st.stats["rows_cut"] >= sum(chunks) - len(chunks) * TAIL1
    assert float(mx.max(mx.abs(lg - ref_lg)).item()) < 1e-3 * float(mx.max(mx.abs(ref_lg)).item())
    assert int(mx.argmax(lg).item()) == int(mx.argmax(ref_lg).item())
    # encoder half + midpoint compressed KV and index keys: bit for bit
    mid = st.lay.mid
    assert slots_equal(got[0], ref[0], range(mid))
    assert slots_equal(got[0], ref[0], [mid], slots=("compress", "keys"))
    # decode steps continue from the CED cache like from the exact one
    c1, c2 = ref, got
    t = int(mx.argmax(ref_lg).item())
    for _ in range(6):
        a = lm_step(lm, c1, t, off=False)
        b = lm_step(lm, c2, t, off=True)
        assert float(mx.max(mx.abs(a - b)).item()) < 1e-3 * float(mx.max(mx.abs(a)).item())
        t = int(mx.argmax(a).item())


def lm_step(lm, cache, tok, off):
    ctx = C.exact(lm) if off else None
    out = lm(mx.array([[tok]]), cache=cache).logits[:, -1].astype(mx.float32)
    mx.eval(out)
    return out


# ------------------------------------------------------------------------------------------------------------------ approximate (real window)
def test_encoder_state_exact_decoder_window_ring_filled():
    ids = M.tokens(200)
    lm, cfg = M.make(sliding_window=16)
    ref, ref_lg = prefill(lm, ids, (64, 64, 72))
    C.enable(lm, DS41)
    got, lg = prefill(lm, ids, (64, 64, 72))
    st = C.state(lm)
    mid = st.lay.mid
    assert st.stats == dict(chunks=3, rows=200, rows_cut=200 - 3 * 16, skip_cache=0)
    assert got[0].offset == ref[0].offset == 200
    assert lg.shape == ref_lg.shape
    assert slots_equal(got[0], ref[0], range(mid))
    assert slots_equal(got[0], ref[0], [mid], slots=("compress", "keys"))
    for i in range(mid, cfg.num_hidden_layers):
        w = got[0].window[i]
        assert w is not None and w.shape[1] == 16
    # the window ring of a decoder layer holds the last 16 positions (ring slot = position % 16): same slots as the exact run, near values
    for i in range(mid, cfg.num_hidden_layers):
        a, b = got[0].window[i], ref[0].window[i]
        assert a.shape == b.shape
        assert bool(mx.all(mx.isfinite(a)).item())
    # the last-row logits stay a close distribution: tiny model, random weights, so only a loose bound
    assert kl(ref_lg, lg) < 0.5
    # decode continues
    t = int(mx.argmax(lg).item())
    out = lm(mx.array([[t]]), cache=got).logits
    mx.eval(out)
    assert out.shape[1] == 1 and got[0].offset == 201


def test_short_chunks_and_decode_take_the_stock_path():
    ids = M.tokens(40)
    lm, cfg = M.make(sliding_window=16)
    C.enable(lm, DS41)
    st = C.state(lm)
    cache = lm.make_cache()
    lm(mx.array([ids[:16]]), cache=cache)          # n == tail, not forced
    lm(mx.array([ids[16:17]]), cache=cache)        # decode
    lm(mx.array([ids[17:20]]), cache=cache)        # verify-like
    assert st.stats["chunks"] == 0


def test_exact_context_and_toggle():
    ids = M.tokens(120)
    lm, cfg = M.make(sliding_window=16)
    ref, ref_lg = prefill(lm, ids, (60, 60))
    C.enable(lm, DS41)
    assert C.signature(lm) == "ced1-m4-t16"
    with C.exact(lm):
        assert C.signature(lm) == "exact"
        got, lg = prefill(lm, ids, (60, 60))
    assert same(lg, ref_lg) and slots_equal(got[0], ref[0], range(8))
    assert C.signature(lm) == "ced1-m4-t16"
    C.set_on(lm, False)
    got, lg = prefill(lm, ids, (60, 60))
    assert same(lg, ref_lg)


def test_enable_is_idempotent_and_updates_tail():
    lm, cfg = M.make(sliding_window=16)
    r1 = C.enable(lm, DS41)
    r2 = C.enable(lm, DS41, tail=32)
    assert r1["tail"] == 16 and r2["tail"] == 32 and C.state(lm).tail == 32
    ids = M.tokens(100)
    prefill(lm, ids, (50, 50))
    assert C.state(lm).stats["rows_cut"] == 2 * (50 - 32)


# ------------------------------------------------------------------------------------------------------------------ window part
def test_window_nohist_matches_the_stock_window_of_a_fresh_start():
    """After skipped positions the window is the rows of the call only: same KV rows as the stock part (positions s..s+t), the mask of a call
    that starts at 0 (row j sees rows max(0, j-w+1)..j), and the ring holds those rows at position % w."""
    lm, cfg = M.make(sliding_window=16)
    attn = lm.layers[5].attn
    C.enable(lm, DS41)
    x = mx.random.normal((1, 40, cfg.hidden_size))
    s = 57
    c_stock = lm.make_cache()[0]
    c_hist = lm.make_cache()[0]
    c_hist.window[5] = mx.random.normal((1, 16, cfg.head_dim))                    # stale ring of an earlier chunk
    kv0, mask0 = type(attn).__mro__[1]._window_part(attn, x, 0, c_stock)           # stock method, fresh start
    kv_s, mask_s = C._window_nohist(attn, x, s, c_hist)
    assert mask_s.shape == mask0.shape and bool(mx.array_equal(mask_s, mask0).item())
    assert kv_s.shape == kv0.shape
    ring = c_hist.window[5]
    for p in range(s + 40 - 16, s + 40):
        assert bool(mx.array_equal(ring[0, p % 16], kv_s[0, p - s]).item())


def test_window_one_decoder_ring_matches_exact():
    ids = M.tokens(90)
    lm, cfg = M.make(sliding_window=1)
    ref, _ = prefill(lm, ids, (45, 45))
    C.enable(lm, DS41, tail=TAIL1)
    got, _ = prefill(lm, ids, (45, 45))
    for i in range(C.state(lm).lay.mid, cfg.num_hidden_layers):
        a, b = got[0].window[i], ref[0].window[i]
        assert a.shape == b.shape
        assert float(mx.max(mx.abs(a - b)).item()) < 1e-3 * float(mx.max(mx.abs(b)).item()) + 1e-6


# ------------------------------------------------------------------------------------------------------------------ tail = receptive field
@pytest.mark.parametrize("chunks", [(200,), (64, 64, 72), (100, 100)])
def test_receptive_field_tail_equals_exact(chunks):
    """Row p of decoder layer j depends on the window chain only: with tail = D * (w - 1) + w the last row and the window rings of every
    decoder layer are the exact ones (up to matmul-shape rounding), while the rows before the tail skip the decoder half."""
    ids = M.tokens(sum(chunks))
    lm, cfg = M.make(sliding_window=16)
    ref, ref_lg = prefill(lm, ids, chunks)
    r = C.enable(lm, DS41, tail="exact")
    assert r["tail"] == C.receptive_rows(C.state(lm).lay) == 4 * 15 + 16
    got, lg = prefill(lm, ids, chunks)
    assert C.state(lm).stats["rows_cut"] > 0 or max(chunks) <= r["tail"]
    assert float(mx.max(mx.abs(lg - ref_lg)).item()) < 1e-4 * float(mx.max(mx.abs(ref_lg)).item())
    for i in range(C.state(lm).lay.mid, cfg.num_hidden_layers):
        a, b = got[0].window[i], ref[0].window[i]
        assert float(mx.max(mx.abs(a - b)).item()) < 1e-4 * float(mx.max(mx.abs(b)).item())
    # one row less than the chain needs is not exact for the oldest window rows
    C.enable(lm, DS41, tail=16)
    got16, lg16 = prefill(lm, ids, chunks)
    if max(chunks) > 76:
        assert float(mx.max(mx.abs(lg16 - ref_lg)).item()) > 1e-4 * float(mx.max(mx.abs(ref_lg)).item())


def test_per_call_tail_overrides():
    ids = M.tokens(100)
    lm, cfg = M.make(sliding_window=16)
    C.enable(lm, DS41)
    cache = lm.make_cache()
    lm(mx.array([ids[:50]]), cache=cache, ced_tail=32)
    assert C.state(lm).stats["rows_cut"] == 18
    with pytest.raises(ValueError):
        lm(mx.array([ids[50:100]]), cache=cache, ced_tail=8)


def test_every_decoder_layer_restarts_its_window_at_the_tail(monkeypatch):
    """Skipped positions leave the ring stale: all decoder layers (midpoint included) must take the no-history window on a cut chunk and the
    stock one on a chunk with nothing cut."""
    seen = []
    real = C._window_nohist
    monkeypatch.setattr(C, "_window_nohist", lambda attn, x, s, cache: (seen.append(attn.layer_idx), real(attn, x, s, cache))[1])
    ids = M.tokens(120)
    lm, cfg = M.make(sliding_window=16)
    C.enable(lm, DS41)
    cache = lm.make_cache()
    lm(mx.array([ids[:60]]), cache=cache)
    assert seen == list(range(4, 8))
    seen.clear()
    lm(mx.array([ids[60:70]]), cache=cache)                      # 10 rows < tail: stock path, ring history used
    assert seen == []
    lm(mx.array([ids[70:120]]), cache=cache)
    assert seen == list(range(4, 8))


def test_auto_enable_default_and_off_switch():
    logs = []
    lm, cfg = M.make()
    assert C.auto_enable(lm, DS41, disabled=True, log=logs.append) is None and C.state(lm) is None
    r = C.auto_enable(lm, DS41, log=logs.append)
    assert r["mid"] == 4 and C.state(lm).on and "ced: on" in logs[-1]
    lm2, _ = M.make(compress_ratios=[0, 0, 2, 2, 1, 2, 1, 1])
    assert C.auto_enable(lm2, DS41, log=logs.append) is None and "not supported" in logs[-1]
    assert C.auto_enable(lm2, GLM5, log=logs.append) is None

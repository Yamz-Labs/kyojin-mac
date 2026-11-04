"""bound_dec (KYOJIN_BOUND_DEC): the layers a CED chunk runs on its tail get an eval group and a YIELD point too, and nothing moves.

The trap this pins: `bound_layers.install` wrapped only `type(layers[0])`, while CED (`ced._install_dsv4`) and the drafter taps give instances their own
subclass with its own `__call__`. Wrapping every class with a loop that reads `cls.__call__` into a variable the wrappers share makes every class call the
LAST class's `__call__`; an encoder layer then enters `_block_call` with the chunk's full `h` and the tail-sliced `image_mask`, and the MoE gate fails with
`broadcast_shapes (1,n,experts)` against `(1,tail,experts)`. The evals are innocent: it reproduces with no eval at all. Second trap: the decoder half carries
only `tail` rows, so the row test alone reads it as a decode step. The chunk's rows come from the CED context instead.
"""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import bound_layers as BL, ced as C  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402


def prefill(lm, ids, sizes, cache=None):
    """Run the chunks, return (logits, whole cache state, offset) with everything evaluated."""
    cache = lm.make_cache() if cache is None else cache
    logits, s = None, 0
    for n in sizes:
        logits = lm(mx.array([ids[s:s + n]]), cache=cache).logits
        mx.eval(logits, *cache[0].state)
        s += n
    return np.array(logits), [np.array(a) for a in cache[0].state], cache[0].offset


def same(a, b):
    return a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and a[2] == b[2] and len(a[1]) == len(b[1]) and all(
        x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a[1], b[1]))


def test_decoder_half_gets_its_bounds_yields_and_the_same_bits():
    lm, cfg = M.make()
    C.enable(lm, DS41)                                               # mid 4 of 8, tail 16
    assert C.layout(cfg).mid == 4 and C.state(lm).tail == 16
    ids = M.tokens(128)
    ref = prefill(lm, ids, (64, 64))                                 # CED, nothing wrapped
    assert BL.install(lm) == len(lm.layers)
    assert BL.install(lm) == 0                                       # idempotent
    old_every, old_min, old_dec = BL.EVERY, BL.MIN_ROWS, BL.DEC
    BL.EVERY, BL.MIN_ROWS = 4, 64
    yields = []
    BL.YIELD[0] = lambda: yields.append(1)
    try:
        BL.reset()
        BL.DEC = True                                                # bound every class the layers use
        got = prefill(lm, ids, (64, 64))
        # 4 encoder layers and 4 decoder layers per chunk, one bound each at EVERY=4: 2 per chunk (1 today)
        assert BL.stats()["bounds"] == 4, BL.stats()
        assert len(yields) == 4
        assert same(got, ref), "the decoder-half bounds moved a value"
        BL.reset()
        yields.clear()
        BL.DEC = False                                               # today: the decoder classes are pass-throughs
        today = prefill(lm, ids, (64, 64))
        assert BL.stats()["bounds"] == 2, BL.stats()
        assert len(yields) == 2
        assert same(today, ref)
    finally:
        BL.YIELD[0] = None
        BL.DEC, BL.EVERY, BL.MIN_ROWS = old_dec, old_every, old_min
        BL.uninstall()
    assert BL.install(lm) == len(lm.layers)                          # uninstall leaves the classes re-installable
    BL.uninstall()


def test_served_install_order_bounds_first_then_ced_arms():
    """The served order is `patch_model` (this bound) -> `enable_ced` -> the drafter taps, the reverse of the probe. CED then wraps the base wrapper, the midpoint
    bypasses the block call, and only the outermost wrapper counts: today's 1 bound with the flag off, 2 with it on, same bits."""
    lm, cfg = M.make()
    assert BL.install(lm) == len(lm.layers)                          # patch_model, before the backend arms anything
    C.enable(lm, DS41)                                               # rewrap() adds the decoder classes
    ids = M.tokens(128)
    old = (BL.EVERY, BL.MIN_ROWS, BL.DEC, BL.ENABLED)
    BL.EVERY, BL.MIN_ROWS = 4, 64
    yields = []
    BL.YIELD[0] = lambda: yields.append(1)
    try:
        BL.ENABLED, BL.DEC = False, False
        ref = prefill(lm, ids, (64, 64))                             # no bound
        BL.ENABLED, BL.DEC = True, False
        BL.reset()
        today = prefill(lm, ids, (64, 64))
        assert BL.stats()["bounds"] == 2, BL.stats()                 # 1 per chunk: the base wrapper inside _block_call sees tail rows as a decode step
        assert len(yields) == 2 and same(today, ref)
        BL.reset()
        yields.clear()
        BL.DEC = True
        got = prefill(lm, ids, (64, 64))
        assert BL.stats()["bounds"] == 4, BL.stats()                 # 2 per chunk: encoder half + midpoint and after, before `advance`
        assert len(yields) == 4 and same(got, ref)
        BL.EVERY = 5                                                 # a bound lands on the midpoint itself
        BL.reset()
        mid = prefill(lm, ids, (64, 64))
        assert BL.stats()["bounds"] == 3 and same(mid, ref)          # the first one is the midpoint of chunk 1 (call 5 of the 8)
    finally:
        BL.YIELD[0] = None
        BL.EVERY, BL.MIN_ROWS, BL.DEC, BL.ENABLED = old
        BL.uninstall()


def test_a_chunk_that_skips_its_decoder_half_is_bounded_and_keeps_the_bits():
    """ced_skip_dec chunks are the ones the engine slices; with bound_dec on they offer the decoder-half bound and yield too (the layers after the midpoint
    return their input unchanged, so only the midpoint has work there)."""
    lm, cfg = M.make()
    assert BL.install(lm) == len(lm.layers)
    C.enable(lm, DS41)
    old = (BL.EVERY, BL.MIN_ROWS, BL.DEC, BL.ENABLED)
    BL.EVERY, BL.MIN_ROWS, BL.ENABLED = 4, 64, True
    ids = M.tokens(64)

    def skipped():
        cache = lm.make_cache()
        with C.skip_decoder(lm):
            logits = lm(mx.array([ids]), cache=cache).logits
            mx.eval(logits, *cache[0].state)
        return np.array(logits), [np.array(a) for a in cache[0].state], cache[0].offset
    try:
        BL.DEC = False
        BL.reset()
        ref = skipped()
        assert BL.stats()["bounds"] == 1, BL.stats()
        BL.DEC = True
        BL.reset()
        solo = skipped()                                             # no hook armed: a solo chunk pays exactly what it pays with the flag off
        assert BL.stats()["bounds"] == 1, BL.stats()
        assert same(solo, ref)
        BL.YIELD[0] = lambda: None
        BL.reset()
        got = skipped()
        assert BL.stats()["bounds"] == 2, BL.stats()
        assert same(got, ref)
        assert C.state(lm).stats["skip_dec"] == 3
    finally:
        BL.YIELD[0] = None
        BL.EVERY, BL.MIN_ROWS, BL.DEC, BL.ENABLED = old
        BL.uninstall()


def test_the_chunk_rows_come_from_the_ced_context_not_the_tail():
    """The direct cause of the 1-bound probe: `_rows` of a decoder-half call is `tail`, the chunk is n."""
    lm, cfg = M.make()
    C.enable(lm, DS41)
    cache = lm.make_cache()
    entry = cache[0]
    h_tail = mx.zeros((1, 16, 2, 64))
    pre_tail = mx.zeros((1, 16, 2))
    assert BL._rows((h_tail, pre_tail, None, cache), {}) == 16
    ctx = C._Ctx(n=64, s0=0, tail=16)
    entry._ced = ctx
    try:
        assert BL._chunk_rows((h_tail, pre_tail, None, cache), {}) == 64
    finally:
        entry._ced = None
    assert BL._chunk_rows((h_tail, pre_tail, None, cache), {}) == 16


@pytest.mark.parametrize("install_first", [False, True])
def test_a_decode_round_inside_a_decoder_half_yield_cannot_touch_the_chunk_ced_state(install_first):
    """YIELD runs the other sessions' decode rounds. They carry their own cache entry (no `_ced`), so `_block_call` never sees this chunk's context, and the
    model-level CED state (on/skip/tail) is only read by a call that takes the CED path (n > tail). Checked from inside the yield, on the second (decoder-half)
    bound, and by the exact bits of the chunk and of every cache buffer. Both install orders: the probe's (CED first) and the served one (`patch_model` first)."""
    lm, cfg = M.make()
    if install_first:
        assert BL.install(lm) == len(lm.layers)
    C.enable(lm, DS41)
    st = C.state(lm)
    if not install_first:
        assert BL.install(lm) == len(lm.layers)
    old_every, old_min, old_dec = BL.EVERY, BL.MIN_ROWS, BL.DEC
    BL.EVERY, BL.MIN_ROWS, BL.DEC = 4, 64, True
    ids = M.tokens(64)
    try:
        other = lm.make_cache()                                      # a second session, prefilled (a CED chunk of its own) then decoding
        lm(mx.array([ids[:20]]), cache=other)
        mx.eval(*other[0].state)
        assert other[0].offset == 20 and other[0]._ced is None
        cache = lm.make_cache()
        seen = {"yields": 0, "ctx_before": None, "ctx_same": None, "state_same": None, "other_logits": 0}

        def pf_yield():
            seen["yields"] += 1
            if seen["yields"] != 2:                                  # the decoder-half bound of a 2-half chunk
                return
            entry, ctx = cache[0], cache[0]._ced
            assert ctx is not None and entry.offset == 0             # inside the chunk, before `advance`
            before = (ctx.n, ctx.s0, ctx.tail, ctx.skip, st.on, st.skip, st.tail, entry.offset)
            seen["ctx_before"] = before
            logits = lm(mx.array([[ids[20]]]), cache=other).logits   # another session's decode round, inside the yield
            mx.eval(logits, *other[0].state)
            seen["other_logits"] = int(logits.shape[-1])
            seen["ctx_same"] = cache[0]._ced is ctx
            seen["state_same"] = before == (ctx.n, ctx.s0, ctx.tail, ctx.skip, st.on, st.skip, st.tail, entry.offset)

        BL.YIELD[0] = pf_yield
        BL.reset()
        st.stats.clear(); st.stats.update(dict(chunks=0, rows=0, rows_cut=0, skip_cache=0))
        got = prefill(lm, ids, (64,), cache)
        assert seen["yields"] == 2 and BL.stats()["bounds"] == 2
        assert seen["ctx_before"] == (64, 0, 16, False, True, False, 16, 0)
        assert seen["ctx_same"] is True and seen["state_same"] is True
        assert seen["other_logits"] == cfg.vocab_size
        assert st.stats["chunks"] == 1                               # the decode round did not take the CED path
        BL.YIELD[0] = lambda: None
        BL.reset()
        ref = prefill(lm, ids, (64,))                                # same chunk, no decode round in between
        assert BL.stats()["bounds"] == 2
        assert same(got, ref)
    finally:
        BL.YIELD[0] = None
        BL.DEC, BL.EVERY, BL.MIN_ROWS = old_dec, old_every, old_min
        BL.uninstall()

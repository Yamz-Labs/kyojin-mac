"""ced_skip_dec: a prompt chunk whose NEXT chunk is a CED chunk skips its decoder half. On a tiny random DS4.1 text model (stock mlx-vlm modules, any MLX backend):
- the final logits and every cache buffer the decode reads are bit for bit those of the full CED prefill, on several chunk grids;
- the skipped chunk still writes the encoder half and the midpoint's compressed KV and index keys, and advances the cache;
- the hint is ignored by a chunk that does not take the CED path, and the lever never fires ahead of a short chunk;
- the engine gate (`Engine._skip_dec`): default off, never on the last chunk, mode 1 needs no snapshot at the boundary, mode 2 takes none."""
import os
import sys
import types

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C, flags as FL  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402
from test_ced import same, slots_equal  # noqa: E402

TAIL = 16                                  # the mini model's sliding window


def run(ids, chunks, skip, lm=None):
    """Prefill `ids` in `chunks` with CED; chunk i skips its decoder half when skip[i]. Returns (cache, last-row logits or None, model)."""
    if lm is None:
        lm, _ = M.make(sliding_window=TAIL)
        C.enable(lm, DS41)
    cache = lm.make_cache()
    lg, s = None, 0
    for c, sk in zip(chunks, skip):
        if sk:
            with C.skip_decoder(lm):
                lm(mx.array([ids[s:s + c]]), cache=cache)
            mx.eval(*cache[0].state)
        else:
            out = lm(mx.array([ids[s:s + c]]), cache=cache).logits
            mx.eval(out, *cache[0].state)
            lg = out[:, -1].astype(mx.float32)
        s += c
    assert s == len(ids)
    return cache, lg, lm


def all_equal(a, b, n_layers):
    return (slots_equal(a[0], b[0], range(n_layers), slots=("window", "compress", "keys", "kv_state", "score_state"))
            and a[0].offset == b[0].offset)


@pytest.mark.parametrize("chunks,skip", [
    ((64, 64), (True, False)),
    ((48, 40, 72), (True, True, False)),
    ((40, 40, 40, 40), (True, False, True, False)),
    ((33, 17, 64), (True, True, False)),            # a 17-row chunk (> tail) after a skipped one is itself skipped: its next chunk has 64 rows
])
def test_final_state_is_bit_identical_to_the_full_prefill(chunks, skip):
    ids = M.tokens(sum(chunks))
    full, lg_full, lm = run(ids, chunks, (False,) * len(chunks))
    n_layers = lm.config.num_hidden_layers
    got, lg, lm2 = run(ids, chunks, skip)
    assert same(lg, lg_full)
    assert all_equal(got, full, n_layers)
    assert C.state(lm2).stats["skip_dec"] == sum(skip)
    # decode continues from either cache with the same logits
    t = int(mx.argmax(lg_full).item())
    a = lm(mx.array([[t]]), cache=full).logits
    b = lm2(mx.array([[t]]), cache=got).logits
    mx.eval(a, b)
    assert same(a, b)


def test_skipped_chunk_writes_what_later_chunks_read():
    """After one skipped chunk: the encoder half and the midpoint's compressed KV and index keys equal the full chunk's, the decoder rings are untouched."""
    ids = M.tokens(64)
    full, _, lm = run(ids[:48], (48,), (False,))
    skipped, _, lm2 = run(ids[:48], (48,), (True,))
    st = C.state(lm)
    mid = st.lay.mid
    assert slots_equal(skipped[0], full[0], range(mid))
    assert slots_equal(skipped[0], full[0], [mid], slots=("compress", "keys"))
    assert skipped[0].offset == full[0].offset == 48
    for i in range(mid, st.lay.n_layers):
        assert skipped[0].window[i] is None             # no decoder layer ran


def test_flag_off_is_the_old_path_and_stats_are_unchanged():
    ids = M.tokens(120)
    lm, _ = M.make(sliding_window=TAIL)
    C.enable(lm, DS41)
    cache = lm.make_cache()
    lm(mx.array([ids[:60]]), cache=cache)
    lm(mx.array([ids[60:]]), cache=cache)
    assert C.state(lm).stats == dict(chunks=2, rows=120, rows_cut=120 - 2 * TAIL, skip_cache=0)
    assert not C.SKIP_DEC and not C.state(lm).skip
    assert FL.FLAGS["ced_skip_dec"].default == "3" and FL.FLAGS["ced_skip_dec"].kind == "int"


def test_hint_is_ignored_by_a_chunk_that_does_not_take_ced():
    """A decode step or a chunk of at most `tail` rows inside skip_decoder runs whole (stock path)."""
    ids = M.tokens(40)
    lm, _ = M.make(sliding_window=TAIL)
    C.enable(lm, DS41)
    ref, _ = M.make(sliding_window=TAIL)
    c1, c2 = lm.make_cache(), ref.make_cache()
    with C.skip_decoder(lm):
        o1 = lm(mx.array([ids[:TAIL]]), cache=c1).logits
    o2 = ref(mx.array([ids[:TAIL]]), cache=c2).logits
    mx.eval(o1, o2)
    assert same(o1, o2) and C.state(lm).stats["chunks"] == 0
    assert not C.state(lm).skip                           # reset on exit


def test_rewrites_decoder_predicate():
    lm, _ = M.make(sliding_window=TAIL)
    C.enable(lm, DS41)
    cache = lm.make_cache()
    assert not C.rewrites_decoder(lm, cache, TAIL)        # n == tail: stock path
    assert not C.rewrites_decoder(lm, cache, 1)
    assert C.rewrites_decoder(lm, cache, TAIL + 1)
    C.state(lm).force = True
    assert C.next_chunk_is_ced(lm, cache, TAIL) and not C.rewrites_decoder(lm, cache, TAIL)     # forced n == tail runs CED but reads the ring
    C.state(lm).force = False
    C.set_on(lm, False)
    assert not C.rewrites_decoder(lm, cache, 10 * TAIL)
    assert not C.rewrites_decoder(lm, [object()], 10 * TAIL)                  # not a DS4.1 cache


# ------------------------------------------------------------------------------------------------------------------ the engine gate
class _Backend:
    def __init__(self, mode, tail=TAIL):
        self.mode, self.tail = mode, tail

    def ced_can_skip(self, cache, n):
        return self.mode if self.mode and n > self.tail else 0


def _engine(mode, chunk=64, block=64, disk=None, budget=1):
    from kyojin_mlx.serve.engine import Engine
    e = Engine.__new__(Engine)
    e.chunk, e.block, e.big_chunk, e.big_from = chunk, block, 0, 0
    e.backend = _Backend(mode)
    e.prefix = types.SimpleNamespace(budget=budget, disk=disk)
    return e


def _sess(cuts=(), resume=False):
    return types.SimpleNamespace(cuts=list(cuts), resume=resume, cache=object())


def test_engine_gate_default_off_and_last_chunk():
    assert not _engine(0)._skip_dec(_sess(), 64, 256)
    e = _engine(2)
    assert e._skip_dec(_sess(), 64, 256)
    assert not e._skip_dec(_sess(), 256, 256)            # the last chunk runs whole
    assert not e._skip_dec(_sess(resume=True), 64, 256)
    assert not e._skip_dec(_sess(), 64, 64 + TAIL)       # the next chunk has `tail` rows: stock path, reads the rings
    assert e._skip_dec(_sess(), 64, 64 + TAIL + 1)
    e.backend = object()                                 # a backend without the method (Q35, fakes)
    assert not e._skip_dec(_sess(), 64, 256)


def test_engine_gate_follows_the_chunk_grid_and_cuts():
    e = _engine(2)
    assert e._chunk_end(_sess(), 0, 200) == 64 and e._chunk_end(_sess(), 192, 200) == 200
    assert e._chunk_end(_sess([40]), 0, 200) == 40 and e._chunk_end(_sess([40]), 40, 200) == 64
    # a cut leaves a short next chunk of 4 rows: this chunk must run whole
    assert not e._skip_dec(_sess([68]), 64, 200)
    assert e._skip_dec(_sess([100]), 64, 200)


def test_engine_gate_mode_1_needs_no_snapshot_at_the_boundary():
    e = _engine(1, chunk=64, block=64)
    assert not e._skip_dec(_sess(), 64, 256)             # a memory snapshot is taken at 64
    e = _engine(1, chunk=64, block=128)
    assert e._skip_dec(_sess(), 64, 256) and not e._skip_dec(_sess(), 128, 256)
    assert not e._skip_dec(_sess([64]), 64, 256)         # a cut stores
    e = _engine(1, chunk=64, block=128, disk=types.SimpleNamespace(block=64))
    assert not e._skip_dec(_sess(), 64, 256)             # a disk checkpoint is taken at 64
    e = _engine(1, chunk=64, block=64, budget=0)
    assert e._skip_dec(_sess(), 64, 256)                 # prefix cache off: nothing is stored
    assert _engine(2, chunk=64, block=64)._skip_dec(_sess(), 64, 256)    # mode 2 skips anyway; _prefill_step stores nothing there


def test_a_skip_ahead_of_a_short_chunk_would_change_the_output():
    """Why the engine gate exists: skip the chunk before a chunk of at most `tail` rows and the stock path reads decoder rings that were never written."""
    ids = M.tokens(40 + 8)
    full, lg_full, _ = run(ids, (40, 8), (False, False))
    bad, lg_bad, _ = run(ids, (40, 8), (True, False))
    assert not same(lg_bad, lg_full)


# ------------------------------------------------------------------------------------------------------------------ the engine loop
def test_engine_skips_and_stores_no_snapshot_at_a_skipped_boundary():
    import time
    from test_serve_engine import Fake, collect, nxt
    from kyojin_mlx.serve.engine import Engine

    calls = []

    class SkipFake(Fake):
        def __init__(self, mode):
            self.mode = mode

        def ced_can_skip(self, cache, n):
            return self.mode if n > 4 else 0

        def prefill_chunk(self, cache, ids, draft=None, skip_dec=False):
            calls.append((len(cache["h"]), len(ids), skip_dec))
            lg = super().prefill_chunk(cache, ids)
            return None if skip_dec else lg

    ids = [(7 * i + 3) % 90 for i in range(8 * 4 + 3)]            # chunks 8, 8, 8, 8, 3
    outs = {}
    for mode in (0, 2):
        calls.clear()
        eng = Engine(SkipFake(mode), sessions=1, ctx=4096, chunk=8, block=8, prefix_bytes=1 << 20)
        eng.start()
        try:
            outs[mode] = collect(eng.submit(list(ids), max_new=6))
            stored = sorted(k[0] for k in eng.prefix.entries)
            stats = dict(eng.stats)
        finally:
            eng.stop()
        if mode == 0:
            assert not any(sk for _, _, sk in calls) and stored == [8, 16, 24, 32]
        else:
            # chunks 0, 1, 2 are followed by an 8-row chunk (> 4): skipped; chunk 3 is followed by 3 rows: whole; the last runs whole
            assert [sk for _, _, sk in calls] == [True, True, True, False, False]
            assert stored == [32] and stats["pf_skip_dec"] == 3
    assert outs[0][0] == outs[2][0] and len(outs[0][0]) == 6       # same tokens

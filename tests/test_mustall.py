"""Flag mu_keep (KYOJIN_MU_KEEP): a member joining or leaving the batched decode copies nothing of the members that stay.

What the lever is: the stacked session axis of a batched decode reserves spare slots and each session owns one, so a leave writes back that one
row and drops its slot from the map, a join writes the joining session's rows into a free slot, and the rows of the members that stay are never
extracted, re-merged, re-stacked or re-packed. `Plan.sess` holds the slot of a row (`Plan.NS` the slot count), so every kernel and scatter of the
core addresses a slot; `Plan.S` stays the number of sessions.

Checks:
(1) a stack / join / leave / unstack sequence on small tensors gives the logits and every cache array of flag 0, bit for bit (the tiny DS4.1
    model with the served core: head_dim 512, Metal);
(2) the kept members' stacked buffers are the same array objects and their bytes are not rewritten by a leave or a join (`HITS`: keep_bytes
    stays 0, `copy_bytes` only carries the row that changes);
(3) flag 0 is untouched: no slot axis, one stacked row per session, the same arrays (CPU);
(4) the engine's `_recast` keeps the epoch and every session's tokens are exact (fake slot backend, no model);
(5) small units: `Plan` slots, the slot stack helpers, `sync_row` against `sync_rows`.
"""
import gc
import os
import sys
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402

from kyojin_mlx import dsv4_core as DC, fp4pool as FP, rows as R  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402
from kyojin_mlx.serve.backends import Backend  # noqa: E402
from kyojin_mlx.serve.engine import Engine  # noqa: E402

METAL = pytest.mark.skipif(not mx.metal.is_available(), reason="the batched decode core is Metal only")

# The served shape the core takes (dsv4_core.attend wants D % 128 == 0, the kv chain and the row core want the pack's 512 / 64): the tiny model's
# weights, a 512-wide head and a 128-wide hidden so the whole pass is a few MB.
OVER = dict(head_dim=512, hidden_size=128, num_attention_heads=4, index_head_dim=128, q_lora_rank=64, o_lora_rank=64, hc_mult=4)


class Wrap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


_LM = None


@pytest.fixture(scope="module", autouse=True)
def _release():
    """Drop the test model and the allocator's residue when the file is done: the next file's byte readings (`mx.get_active_memory`) start clean."""
    yield
    global _LM
    _LM = None
    gc.collect()
    mx.eval()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def model():
    """One model for the whole file: rows.enable installs per-instance classes, and two enabled models in one process do not agree bit for bit
    (the second enable re-wraps classes the first one already wrapped)."""
    global _LM
    if _LM is None:
        lm, _ = M.make(seed=0, **OVER)
        R.enable(Wrap(lm), DS41)
        _LM = lm
    return _LM


def sess(n: int, seed: int):
    lm = model()
    c = FP.wrap_cache(lm.make_cache())
    mx.eval(lm(mx.array([M.tokens(n, seed=seed)]), cache=c).logits, *c[0].state)
    return c


TOK0 = 7          # the first token fed to a step: inside the vocabulary (an offset as a token id reads past the embedding table)


def steps(bc, toks: list, k: int):
    """k decode steps on the batched cache; returns (the logits of every step, the tokens of the next step)."""
    lm = model()
    out = []
    for _ in range(k):
        y = R.step(Wrap(lm), bc, mx.array(toks, dtype=mx.int32))
        mx.eval(y)
        lg = np.array(y)
        out.append(lg)
        toks = [int(x) for x in np.argmax(lg, axis=-1)]
    return out, toks


def cache_arrays(c):
    """Every state array of one per-session cache (a cache list of one entry), dense, plus the offset."""
    row = c[0].rows[0] if hasattr(c[0], "rows") else c[0]
    out = {}
    for name in ("compress", "keys", "window", "kv_state", "score_state"):
        for l, v in enumerate(getattr(row, name)):
            out[f"{name}[{l}]"] = None if v is None else np.array(v)
    out["offset"] = np.array([row.offset])
    return out


def same_cache(a, b):
    assert set(a) == set(b)
    for k in a:
        x, y = a[k], b[k]
        assert (x is None) == (y is None), k
        if x is not None:
            assert x.shape == y.shape and bool(np.array_equal(x, y)), k


def sequence(flag: str, split: bool, long_n: int = 192, short_n: int = 160):
    """stack the long session, decode, join the short one, decode, let it leave, decode, unstack. `split` = flag 0: no slot axis, so every
    membership change dissolves the batch and rebuilds it (the reference)."""
    os.environ["KYOJIN_MU_KEEP"] = flag
    long_, short = sess(long_n, 1), sess(short_n, 2)
    bc = R.batch([long_])
    st = DC.stack_rows(bc[0])
    lg, toks = steps(bc, [TOK0], 2)
    joiner = short
    if split:                                                   # flag 0: dissolve, merge, stack again
        bc = R.batch([[bc[0].extract(0)], joiner])
        st = DC.stack_rows(bc[0])
    else:                                                       # mu_keep: the joining session's rows go into a free slot
        st.adopt(joiner[0])
    lg2, toks2 = steps(bc, toks + [TOK0 + 1], 3)
    if split:
        leaver = [bc[0].extract(1)]
        bc = R.batch([[bc[0].extract(0)]])
        st = DC.stack_rows(bc[0])
    else:
        leaver = [st.release_rows([1])[1]]
        st.keep_only([0])
    lg3, toks3 = steps(bc, toks2[:1], 2)
    st.unstack()
    return lg + lg2 + lg3, cache_arrays(bc), cache_arrays(leaver), toks3


@METAL
def test_join_and_leave_are_bit_identical_to_a_rebuild():
    """The whole sequence: every logit of every step and every cache array of both sessions equal flag 0."""
    ref, ref_cache, ref_leaver, ref_tok = sequence("0", split=True)
    got, got_cache, got_leaver, got_tok = sequence("1", split=False)
    assert len(ref) == len(got) == 7
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape and bool(np.array_equal(a, b)), f"step {i}"
    assert ref_tok == got_tok
    same_cache(ref_cache, got_cache)
    same_cache(ref_leaver, got_leaver)


@METAL
def test_a_join_longer_than_the_batch_buffer_grows_it():
    """Served 01/11: a joining session with more rows than the stack holds raised a broadcast error and failed the requests of the pass."""
    ref, ref_cache, ref_leaver, ref_tok = sequence("0", split=True, long_n=64, short_n=600)
    got, got_cache, got_leaver, got_tok = sequence("1", split=False, long_n=64, short_n=600)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape and bool(np.array_equal(a, b)), f"step {i}"
    assert ref_tok == got_tok
    same_cache(ref_cache, got_cache)
    same_cache(ref_leaver, got_leaver)


@METAL
def test_the_kept_rows_are_not_rewritten():
    """A leave and a join touch only the row that changes: the stacked buffers of the saved slots keep their object and their bytes."""
    os.environ["KYOJIN_MU_KEEP"] = "1"
    a, b, c = sess(192, 1), sess(160, 2), sess(160, 3)
    bc = R.batch([a, b])
    st = DC.stack_rows(bc[0])
    steps(bc, [TOK0, TOK0 + 1], 1)
    assert st.slots == [0, 1] and st.S_cap == 3                  # one spare slot (mu_spare)
    names = ("win", "cmp", "keys", "kvs", "scs")
    ids = {n: [None if x is None else id(x) for x in getattr(st, n)] for n in names}
    vals = {n: [None if x is None else np.array(x) for x in getattr(st, n)] for n in names}

    def kept(tag):
        for n in names:
            for l, x in enumerate(getattr(st, n)):
                if ids[n][l] is None:
                    assert x is None, (tag, n, l)
                    continue
                assert id(x) == ids[n][l], f"{tag}: {n}[{l}] is a new buffer"          # the kept slot's buffer was not rewritten
                assert bool(np.array_equal(np.array(x)[0], vals[n][l][0])), f"{tag}: {n}[{l}] slot 0 changed"
        assert DC.HITS["keep_bytes"] == 0, tag

    for k in DC.HITS:
        DC.HITS[k] = 0
    st.adopt(c[0])                                             # a join: only the joining session's rows are written
    kept("adopt")
    assert DC.HITS["adopt"] == 1 and DC.HITS["copy_bytes"] > 0
    assert st.slots == [0, 1, 2]
    adopted = DC.HITS["copy_bytes"]
    for k in DC.HITS:
        DC.HITS[k] = 0
    FP.HITS["write_bytes"] = 0
    leaver = st.release_rows([1])[1]                            # a leave: one row written back, the slot dropped
    st.keep_only([0])
    kept("keep_only")
    assert DC.HITS["keep_bytes"] == 0 and st.slots == [0]
    assert len(st.rows) == 1
    # the row that leaves carries its own state only: the window rings + the compressor states, no pool / key layer (sync_tail packs the new rows)
    assert 0 < DC.HITS["copy_bytes"] < adopted
    assert FP.HITS["write_bytes"] == 0, "the leaver must not re-pack a whole pool / key layer"
    # and the state it carries is the stack's: the same offset the batch had for it (one decode step after its prefill)
    assert int(cache_arrays([leaver])["offset"][0]) == 161      # 160 prompt tokens + the step


@METAL
def test_flag_0_has_no_slot_axis():
    """Off: one stacked row per session, no `slots`, and the arrays are exactly the n sessions (the layout of every other lever)."""
    os.environ["KYOJIN_MU_KEEP"] = "0"
    a, b = sess(160, 1), sess(160, 2)
    bc = R.batch([a, b])
    st = DC.stack_rows(bc[0])
    assert st.slots is None and st.S_cap == 2
    plan = DC.plan_of(st)
    assert plan.slots == [0, 1] and plan.NS == 2 and plan.S == 2 and plan.sess == [0, 1]
    for n in ("win", "cmp", "keys", "kvs", "scs"):
        for x in getattr(st, n):
            if x is not None:
                assert x.shape[0] == 2
    st.unstack()
    assert not hasattr(bc[0], "slots")


def test_plan_slots_are_the_buffer_address_and_sessions_stay_ordered():
    """A pass over 2 sessions living in slots 1 and 3: rows address the slot, the count / first arrays stay per session."""
    plan = DC.Plan([10, 11, 12, 20, 21], [3, 2], slots=[1, 3], nslots=4)
    assert plan.S == 2 and plan.NS == 4 and plan.slots == [1, 3]
    assert plan.sess == [1, 1, 1, 3, 3] and plan.first == [0, 3] and plan.p0 == [10, 20]
    assert plan.plain is False and plan.n == 5
    assert [int(x) for x in plan.sess_arr.tolist()] == [1, 1, 1, 3, 3]
    assert [int(x) for x in plan.slot_arr.tolist()] == [1, 3]
    # the slot axis is part of the cached-plan key: a join changes it, the same membership does not
    assert DC.Plan([1, 2], None, [0, 1], 2).key == DC.Plan([1, 2], None, [0, 1], 2).key
    assert DC.Plan([1, 2], None, [0, 1], 3).key != DC.Plan([1, 2], None, [0, 1], 2).key


def test_the_slot_stack_equals_the_plain_stack_on_the_live_slots():
    """`_stack_ragged` / `_stack_plain` / `_stack_state` with slots: the same rows as the plain layout, the spare slot zero / -inf."""
    mx.random.seed(3)
    a = mx.random.normal((1, 5, 8)).astype(mx.float16)
    b = mx.random.normal((1, 3, 8)).astype(mx.float16)
    mx.eval(a, b)
    buf, cnt = DC._stack_ragged([a, b], 8, [0, 2], 4)
    ref, ref_cnt = DC._stack_ragged([a, b], 8)
    assert cnt == [5, 0, 3, 0] and ref_cnt == [5, 3] and buf.shape[0] == 4
    assert bool(mx.array_equal(buf[0, :5], ref[0, :5]).item())
    assert bool(mx.array_equal(buf[2, :3], ref[1, :3]).item())
    assert float(mx.abs(buf[1]).max().item()) == 0.0 and float(mx.abs(buf[3]).max().item()) == 0.0
    w = [mx.ones((1, 2, 4), dtype=mx.float16), None]
    win = DC._stack_plain(w, [1, 0], 3)
    assert win.shape == (3, 2, 4) and float(mx.abs(win[1]).min().item()) == 1.0 and float(mx.abs(win[0]).max().item()) == 0.0
    st = DC._stack_state([None, mx.ones((1, 2, 4), dtype=mx.float32)], -mx.inf, [0, 2], 4)
    assert st.shape == (4, 2, 4) and float(st[2].min().item()) == 1.0 and float(st[0].max().item()) == -mx.inf


@METAL
def test_sync_row_writes_the_same_bits_as_sync_rows():
    """The refactor of the write-back: one row at a time is the whole write-back, for that row."""
    os.environ["KYOJIN_MU_KEEP"] = "0"
    a, b = sess(160, 1), sess(160, 2)
    bc = R.batch([a, b])
    st = DC.stack_rows(bc[0])
    steps(bc, [TOK0, TOK0 + 1], 2)
    snap = [None if x is None else np.array(x) for x in st.win[3]]
    st.sync_row(1)
    one = [None if x is None else np.array(x) for x in st.rows[1].window]
    st.sync_rows()
    both = [None if x is None else np.array(x) for x in st.rows[1].window]
    for x, y in zip(one, both):
        assert (x is None) == (y is None) and (x is None or bool(np.array_equal(x, y)))
    assert snap[1] is not None and bool(np.array_equal(snap[1][None], one[3]))


# ------------------------------------------------------------------------------------------------ the engine's recast (no model)
V = 97


def nxt(hist):
    return (sum((i + 3) * t for i, t in enumerate(hist)) * 31 + len(hist)) % V


def onehot(rows):
    out = [[0.0] * V for _ in rows]
    for r, t in zip(out, rows):
        r[t] = 10.0
    return mx.array(out, dtype=mx.float32)


class SlotBatch:
    """A batched cache that looks like a DS4.1 slot stack: the rows (one per session), the slot of each row, a spare slot, no pass open."""

    def __init__(self, rows, spare=1):
        self.rows = list(rows)
        self.S_cap = len(self.rows) + spare
        self.slots = list(range(len(self.rows)))
        self._counts = None


class SlotFake(Backend):
    """The fake backend of tests/test_serve_engine.py on a slot batch cache: `recast` drops and adopts rows the way dsv4_core does (the caches
    are lists of one entry, like the DS4.1 cache list)."""
    kind = "slotfake"
    max_rows = 8

    def __init__(self):
        self.calls = []

    def make_cache(self):
        return [{"h": []}]

    def fork(self, cache):
        return [{"h": list(copy[0]["h"])} for copy in cache]

    def snapshot(self, cache):
        return self.fork(cache)

    def prefill_chunk(self, cache, ids):
        cache[0]["h"].extend(ids)
        return onehot([nxt(cache[0]["h"])])

    def batch(self, caches):
        return [SlotBatch([list(c) for c in caches])]

    def unbatch(self, bc, n):
        return [list(r) for r in bc[0].rows]

    def recast(self, bc, drop, park, add):
        self.calls.append((list(drop), list(park), len(add)))
        b = bc[0]
        gone = set(drop) | set(park)
        out = {i: list(b.rows[i]) for i in park}
        b.rows = [r for i, r in enumerate(b.rows) if i not in gone]
        b.slots = list(range(len(b.rows)))
        for c in add:
            b.rows.append(list(c))
            b.slots.append(len(b.rows) - 1)
        return out

    def step(self, bc, tokens):
        b = bc[0]
        for c, t in zip(b.rows, tokens.tolist()):
            c[0]["h"].append(int(t))
        return onehot([nxt(c[0]["h"]) for c in b.rows])

    def state(self, cache):
        return []

    def nbytes(self, cache):
        b = cache[0] if isinstance(cache, list) else cache
        rows = b.rows if isinstance(b, SlotBatch) else [b]
        return sum(8 * len((r[0] if isinstance(r, list) else r)["h"]) for r in rows)


def collect(h, timeout=60):
    toks = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = h.events.get(timeout=timeout)
        if ev[0] == "tok":
            toks.append(ev[1])
        else:
            return toks
    raise TimeoutError


def solo(ids, **kw):
    eng = Engine(SlotFake(), sessions=1, ctx=4096, chunk=4, block=8, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, **kw))
    finally:
        eng.stop()


def both(flag, n=3):
    """A long session decodes alone, a short one joins and finishes, the long one is cancelled: every token of both must be the solo stream.
    The join is not raced: the short request is submitted after the long one's first token, which puts it in the decode loop's `ready`."""
    os.environ["KYOJIN_MU_KEEP"] = flag
    be = SlotFake()
    eng = Engine(be, sessions=n, ctx=4096, chunk=4, block=8, prefix_bytes=0)
    eng.start()
    try:
        ids_a = M.tokens(40, seed=11)
        ids_b = M.tokens(28, seed=12)
        ha = eng.submit(ids_a, max_new=4000)
        ev = ha.events.get(timeout=60)
        assert ev[0] == "tok"
        hb = eng.submit(ids_b, max_new=6)
        b = collect(hb)
        ha.cancel()
        a = [ev[1]] + collect(ha)
    finally:
        eng.stop()
    return a, b, be.calls, eng.stats["mu_recast"]


def test_the_engine_recasts_instead_of_dissolving():
    """flag mu_keep: a join and a leave of the short session update the batch in place, and both replies are the solo ones."""
    os.environ["KYOJIN_MU_KEEP"] = "1"
    try:
        a, b, calls, recasts = both("1")
        os.environ["KYOJIN_MU_KEEP"] = "0"
        a0, b0, calls0, recasts0 = both("0")
    finally:
        os.environ.pop("KYOJIN_MU_KEEP", None)
    assert b == b0                                      # the short session's reply: the solo stream, flag on or off
    assert a[:min(len(a), len(a0))] == a0[:min(len(a), len(a0))] and len(a) > 2
    assert recasts >= 2 and calls                        # a join and a leave went through recast
    assert any(add for _drop, _park, add in calls)       # the join was an adopt, not a rebuild
    assert recasts0 == 0 and not calls0                  # the flag off: the dissolve and rebuild of today


def test_a_parked_session_gets_its_cache_back_and_is_flipped_back_in():
    """A yield is the one leave that needs the row: the parked session gets its own cache (only its row is written), then rejoins the same batch
    by a adopt. Its reply is the solo stream, and the long member keeps decoding across both changes."""
    os.environ["KYOJIN_MU_KEEP"] = "1"
    be = SlotFake()
    eng = Engine(be, sessions=2, ctx=4096, chunk=4, block=8, prefix_bytes=0)
    eng.start()
    ids_a, ids_b = M.tokens(40, seed=11), M.tokens(28, seed=12)
    try:
        ha = eng.submit(ids_a, max_new=4000)
        assert ha.events.get(timeout=60)[0] == "tok"
        hb = eng.submit(ids_b, max_new=12)
        t0 = time.time()
        while len(eng.members) < 2 and time.time() - t0 < 20:              # both on the batched cache
            time.sleep(0.001)
        assert len(eng.members) == 2
        eng.members[1].park_now = True                                     # flag kv_admit: the short session yields its slot
        b = collect(hb)
        ha.cancel()
        a = collect(ha)
    finally:
        eng.stop()
        os.environ.pop("KYOJIN_MU_KEEP", None)
    assert eng.stats["yields"] >= 1
    assert any(park for _drop, park, _add in be.calls), be.calls         # the park extracted that row when it left
    assert any(add for _drop, _park, add in be.calls), be.calls          # and the same batch took it back (no rebuild)
    assert b == solo(ids_b, max_new=12)[:len(b)] and len(b) > 2
    assert len(a) > 2


def test_flag_default_and_listed():
    from kyojin_mlx import flags
    f = flags.FLAGS["mu_keep"]
    assert f.default == "1" and f.env == "KYOJIN_MU_KEEP"
    assert "KYOJIN_MU_KEEP" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "cloud", "LEVERS.md")).read()
    os.environ.pop("KYOJIN_MU_KEEP_SPARE", None)
    assert DC.mu_spare() == 1
    os.environ["KYOJIN_MU_KEEP_SPARE"] = "0"
    try:
        assert DC.mu_spare() == 0
    finally:
        os.environ.pop("KYOJIN_MU_KEEP_SPARE", None)
    os.environ["KYOJIN_VERIFY_REPLAY"] = "1"
    os.environ["KYOJIN_MU_KEEP"] = "1"
    try:
        assert DC.mu_keep() is False                    # the recorded command stream is shape-stable: no slot axis under verify_replay
    finally:
        os.environ.pop("KYOJIN_VERIFY_REPLAY", None)
        os.environ.pop("KYOJIN_MU_KEEP", None)

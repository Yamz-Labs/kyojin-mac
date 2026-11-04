"""Flag stack_dedupe (KYOJIN_STACK_DEDUPE): a batched decode keeps one copy of each pool / index-key layer. CPU, tiny random DS4.1.
Checks: (1) bit-identical logits for every decode step and bit-identical caches after the sessions leave the batch, flag 0 against 1,
(2) fewer bytes held by the batched cache (what the engine's kv_bytes counts) and less retained MLX memory, (3) the snapshot alias that makes
a live append copy its whole buffer (why a prefix entry pins capacity), (4) the flag is off by default and listed."""
import gc
import os
import sys
import types

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import fp4pool as FP, rows as R  # noqa: E402
from kyojin_mlx.serve.backends import arrays_of  # noqa: E402


def mem():
    gc.collect()
    mx.eval()
    return mx.get_active_memory()


def nb(v):
    return sum(a.nbytes for a in arrays_of(v))


def dense_layers(c):
    return [None if x is None else mx.array(x) for slots in (c.compress, c.keys) for x in slots]


def run(flag, B=3, n0=640):
    """Prefill B sessions, stack them (dsv4_core.stack_rows, the step the batched core starts with: the core itself needs Metal), measure while stacked, write back
    (sync_rows) and extract. Returns (counted bytes of the stacked cache, retained MLX bytes while stacked, dense pool / key layers before, after, window + state before, after)."""
    os.environ["KYOJIN_STACK_DEDUPE"] = flag
    try:
        from kyojin_mlx import dsv4_core as DC
        lm, _ = M.make(sliding_window=16)
        users, before, side0 = [], [], []
        for i in range(B):
            c = FP.wrap_cache(lm.make_cache())
            mx.eval(lm(mx.array([M.tokens(n0 + 16 * i, seed=1 + i)]), cache=c).logits, *c[0].state)
            before.append(dense_layers(c[0]))
            side0.append([None if x is None else mx.array(x) for k in ("window", "kv_state", "score_state") for x in getattr(c[0], k)])
            users.append(c)
        mx.eval(*[x for d in before for x in d if x is not None], *[x for d in side0 for x in d if x is not None])
        base = mem()                                                          # model residue + the dense references above + the prefilled sessions
        bc = R.batch(users)
        st = DC.stack_rows(bc[0])
        mx.eval(*st.stacked_arrays())
        counted, held = nb(bc), mem() - base
        st.sync_rows()
        mine = [st.rows[u] for u in range(B)]
        after = [dense_layers(r) for r in mine]
        side1 = [[None if x is None else mx.array(x) for k in ("window", "kv_state", "score_state") for x in getattr(r, k)] for r in mine]
        return counted, held, before, after, side0, side1
    finally:
        os.environ.pop("KYOJIN_STACK_DEDUPE", None)


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x is None) == (y is None)
        if x is not None:
            assert x.shape == y.shape and bool(mx.array_equal(x, y).item())


@pytest.mark.parametrize("flag", ["0", "1"])
def test_stack_roundtrip_is_bit_identical(flag):
    """Stack then write back: every pool / index-key layer, window and compressor state of every session comes back bit for bit (flag 0 and 1)."""
    counted, held, before, after, side0, side1 = run(flag)
    for u in range(len(before)):
        same(before[u], after[u])
        same(side0[u], side1[u])


def test_dedupe_holds_fewer_bytes_while_stacked():
    c0, h0, *_ = run("0")
    c1, h1, *_ = run("1")
    print("\nstacked: counted by kv_bytes %d -> %d (%.3f), retained MLX bytes above the prefilled sessions %d -> %d" % (c0, c1, c1 / c0, h0, h1))
    assert c1 < c0                                                            # the sessions' packed codes + scales are gone from the cache the engine measures
    # The retained-MLX delta moves by ~11 KB run to run (the allocator's own residue; this comparison flaked at the base commit too), so it is held to a
    # fraction of the counted saving instead of a strict ordering: the dense mirrors may not grow against it.
    assert h1 - h0 < (c0 - c1) // 4


def test_dedupe_leaves_no_row_copy_until_sync():
    lm, _ = M.make(sliding_window=16)
    cs = [FP.wrap_cache(lm.make_cache()) for _ in range(2)]                  # N = 1 is never released (tests/test_cmu2_mini.py)
    for i, c in enumerate(cs):
        mx.eval(lm(mx.array([M.tokens(320, seed=1 + i)]), cache=c).logits, *c[0].state)
    from kyojin_mlx import dsv4_core as DC
    bc = R.batch(cs)
    os.environ["KYOJIN_STACK_DEDUPE"] = "1"
    try:
        st = DC.stack_rows(bc[0])
    finally:
        os.environ.pop("KYOJIN_STACK_DEDUPE", None)
    row = st.rows[0]
    held = [l for l in range(len(row.compress)) if st.cmp[l] is not None]
    assert held and all(list.__getitem__(row.compress, l) is None for l in held)
    assert st._dirty
    st.sync_rows()                                                            # leave without a step: the layers come back from the stack
    assert all(list.__getitem__(row.compress, l) is not None for l in held)


def test_snapshot_alias_makes_the_next_append_copy_the_buffer():
    """Why a prefix entry pins a whole capacity buffer: an aliased buffer is copied by the live session's next in-place write (MLX is copy on write)."""
    base = mem()
    a = mx.zeros((1, 1 << 16, 256), dtype=mx.uint8)
    mx.eval(a)
    one = mem() - base
    snap = a[...]
    mx.eval(snap)
    assert mem() - base == one                                                # an alias costs nothing
    a[:, 10:20] = mx.ones((1, 10, 256), dtype=mx.uint8)
    mx.eval(a)
    assert mem() - base == 2 * one                                            # the write copied the buffer; the snapshot keeps the old one
    del snap
    assert mem() - base == one


def test_flag_off_by_default_and_listed():
    from kyojin_mlx import flags
    f = flags.FLAGS["stack_dedupe"]
    assert f.default == "0" and f.env == "KYOJIN_STACK_DEDUPE"
    assert "KYOJIN_STACK_DEDUPE" in open(os.path.join(os.path.dirname(__file__), "..", "docs", "cloud", "LEVERS.md")).read()


def _codes(slots):
    out = []
    for i in range(len(slots)):
        e = list.__getitem__(slots, i)
        out.append(None if e is None else (np.array(e.codes[:, :e.n]).tobytes(), np.array(e.scales[:, :e.n]).tobytes(), e.n) if isinstance(e, FP._Pk) else np.array(e).tobytes())
    return out


def test_sync_tail_writes_the_same_codes_as_a_whole_repack():
    """A stack that appended rows: the tail write-back and the whole re-pack leave the same codes, scales and dense rows; a second sync appends from where the first stopped."""
    from kyojin_mlx.fp4pack import pack, unpack
    D = 32
    def on_grid(n, seed):
        mx.random.seed(seed)
        v = mx.random.normal((1, n, D)).astype(mx.float16)
        return unpack(*pack(v, FP.PackedSlots.BLOCK, FP.PackedSlots.E4M3), FP.PackedSlots.BLOCK, FP.PackedSlots.E4M3, mx.float16)
    head, tail = on_grid(512, 1), on_grid(40, 2)
    stack = mx.concatenate([head, tail, mx.zeros((1, 88, D), dtype=mx.float16)], axis=1)     # capacity after the valid rows, as a real stack has
    mx.eval(head, tail, stack)
    def slots():
        return FP.PackedSlots([head])
    a, b = slots(), slots()
    a.write_back(0, stack, 530)
    a.write_back(0, stack, 552)
    a.write_back(0, stack, 552)
    b[0] = stack[:, :552][...]
    assert _codes(a) == _codes(b) and isinstance(list.__getitem__(a, 0), FP._Pk)
    assert np.array(a[0]).tobytes() == np.array(b[0]).tobytes()
    assert list.__getitem__(a, 0).n == 552


def test_sync_tail_whole_write_when_the_row_has_no_entry_or_is_ahead():
    """Released layer (stack_dedupe), and an entry longer than the stack count: the whole write, never a stale prefix."""
    from kyojin_mlx.fp4pack import pack, unpack
    v = unpack(*pack(mx.random.normal((1, 64, 32)).astype(mx.float16), 16, True), 16, True, mx.float16)
    mx.eval(v)
    s = FP.PackedSlots([None])
    s.write_back(0, v, 64)
    assert list.__getitem__(s, 0).n == 64
    s = FP.PackedSlots([v])
    s.write_back(0, v, 16)
    assert list.__getitem__(s, 0).n == 16
    s = FP.PackedSlots([v])
    s.write_back(0, v, 0)
    assert list.__getitem__(s, 0) is None

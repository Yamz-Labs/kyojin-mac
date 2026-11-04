"""KYOJIN_DEC_SEL_EVERY (kyojin_mlx.decsel): the decode index selection reused between rounds.

Pins (1) flag 0 is today's path (no entry, no extra node) and EVERY = 1 is a full scan every round, (2) the keys appended since the last full
scan are candidates of the next round, (3) a consumer layer drops the appended keys whose block the source mask dropped and keeps the entries
of the previous selection, (4) the cache is published from the row the COMMIT keeps, so a rejected draft leaves nothing behind and a commit
that moved the count below the entry forces a full scan, (5) a session switch (a row the batch does not hold an entry for) forces a full scan,
(6) the candidate selection equals the full scan bit for bit when the candidate set holds the full answer, (7) the overlap counter of
`dec_sel_check` counts the common keys, (8) a tiny DS4.1 model decoded through the batched core gives the same logits at flag 0 and EVERY = 1,
engages the lever above the width gate, and publishes the kept row's selection after a partial commit.
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.dirname(__file__))

from types import SimpleNamespace                                                       # noqa: E402

from kyojin_mlx import decsel as DS                                                     # noqa: E402
from kyojin_mlx import dsv4_core as C                                                   # noqa: E402
from kyojin_mlx import flags as FL                                                      # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

NH, DI, K, CAP = 4, 32, 8, 512           # index geometry of the unit world: the thread-per-key chain serves it, so no 32 x 128 gate
T = CAP                                  # t_max (>= 4 K, so the layer clears the width gate)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """Every test starts from the flag's default and puts the module back: the lever is read per call."""
    monkeypatch.setattr(DS, "EVERY", 0)
    monkeypatch.setattr(DS, "CHECK", False)
    DS.HITS.clear()
    DS.STATS.clear()
    yield
    DS.HITS.clear()
    DS.STATS.clear()


def world(rows=2, S=1, seed=7, cap=CAP, dtype=mx.float16):
    """Queries, head weights, dense keys of S sessions, per-row key counts and the pass (one session of `rows` rows, or two of one)."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((S, cap, DI)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    q = rng.standard_normal((rows, NH, DI)).astype(np.float32)
    w = rng.standard_normal((rows, NH)).astype(np.float32)
    counts = [rows] + [0] * (S - 1) if S == 1 else [1] * S
    counts = [c for c in counts if c]
    pos = []
    for s, c in enumerate(counts):
        pos += [cap - 40 + i for i in range(c)]                                        # the rows of one session are consecutive
    plan = C.Plan(pos, counts)
    out = [mx.array(x).astype(dtype) for x in (q, w, keys)]
    mx.eval(*out)
    return out


def emit(lens, k=K):
    plan = C.Plan([int(l) for l in lens], [len(lens)])
    em = SimpleNamespace(lens=[int(l) for l in lens], lens_arr=mx.array([int(l) for l in lens], dtype=mx.int32), tmax=max(lens))
    idxr = SimpleNamespace(index_topk=k, is_candidate_source=False, uses_candidates=False, n_heads=NH, index_head_dim=DI)
    return plan, em, idxr


def put(st, l, row, sel, pos, full=True, age=None):
    e = DS.Entry(pos, sel, 0 if age is None else age, row)
    DS.store_of(st)[(id(row), l)] = e
    return e


# --------------------------------------------------------------------------------------------------------------------- flag behaviour
def test_flag_0_is_todays_path():
    assert DS.EVERY == 0 and not DS.active()
    q, w, keys = world()
    plan, em, idxr = emit([CAP - 39, CAP - 38])
    st = SimpleNamespace(rows=[object()])
    assert DS.layer(st, 3, idxr, keys, None, q, w, em, T, plan) is None
    assert DS.HITS.get("reuse", 0) == 0 and DS.HITS.get("gate", 0) == 0             # not even the gate ran: the core keeps the stock chain
    assert not hasattr(st, "dec_sel") or st.dec_sel == {}


def test_every_1_is_a_full_scan_every_round():
    """EVERY = 1 must be bit-identical to 0: the layer NEVER takes the candidate path, whether an entry exists or not."""
    DS.EVERY = 1
    assert DS.active()
    q, w, keys = world()
    plan, em, idxr = emit([CAP - 39, CAP - 38])
    st = SimpleNamespace(rows=[row := object()])
    assert DS.layer(st, 3, idxr, keys, None, q, w, em, T, plan) is None            # no entry: full scan
    put(st, 3, row, mx.zeros((K,), dtype=mx.int32), CAP - 39)                       # a fresh entry (age 0)
    assert DS.layer(st, 3, idxr, keys, None, q, w, em, T, plan) is None            # age + 1 >= 1: still a full scan
    assert DS.HITS.get("reuse", 0) == 0 and DS.HITS["full"] == 2


def test_width_gate_keeps_the_full_scan():
    """Below 4 x index_topk the full width is already cheap: the lever does not engage, and a candidate set that wide would not pay."""
    DS.EVERY = 3
    q, w, keys = world(cap=64)
    plan, em, idxr = emit([40, 41])
    st = SimpleNamespace(rows=[object()])
    st.dec_sel = {}
    put(st, 3, st.rows[0], mx.arange(K, dtype=mx.int32), 40)                        # an entry at 40 keys
    assert DS.layer(st, 3, idxr, keys, None, q, w, em, 64, plan) is not None        # 64 >= 4 x 8 and 40 <= 40: the lever engages
    DS.HITS["gate"] = 0
    assert DS.layer(st, 5, idxr, keys, None, q, w, em, 31, plan) is None           # gate: t_max < 4 K
    assert DS.HITS["gate"] == 1


# --------------------------------------------------------------------------------------------------------------------- the candidate set
def test_appended_keys_are_candidates():
    """The candidate positions are the previous selection plus every key appended since it, ascending, real entries first."""
    cached = mx.array([[3, 7, 11, -1], [5, -1, -1, -1]], dtype=mx.int32)
    pos = mx.array([12, 6], dtype=mx.int32)
    lens = mx.array([16, 9], dtype=mx.int32)
    gp, real = DS.build(cached, pos, lens, 32, None, maxnew=4)
    gp, real = np.array(gp), np.array(real)
    for u, want in enumerate(([3, 7, 11, 12, 13, 14, 15], [5, 6, 7, 8])):
        got = gp[u][real[u]].tolist()
        assert got == want, (u, got, want)
        assert np.all(np.diff(got) > 0)                                            # ascending: the tie rule stays "lower index first"
        assert real[u][: len(want)].all() and not real[u][len(want):].any()        # a prefix: what the score cut and the select need
        assert np.all(gp[u][~real[u]] == 0)                                        # the padding slot: gathered at 0, never scored


def test_consumer_mask_drops_the_appended_keys_of_a_dropped_block():
    """A consumer layer only adds appended keys whose 8-key block the source kept this round; the previous selection is kept as it is."""
    cached = mx.array([[1, 2, -1, -1]], dtype=mx.int32)
    pos, lens = mx.array([8], dtype=mx.int32), mx.array([24], dtype=mx.int32)
    mask = np.ones((1, 32), dtype=bool)
    mask[0, 16:24] = False                                                          # the block of keys 16..23 is dropped
    gp, real = DS.build(cached, pos, lens, 32, mx.array(mask), maxnew=16)
    got = np.array(gp)[0][np.array(real)[0]].tolist()
    assert [k for k in range(16, 24) if k in got] == []
    assert [k for k in range(8, 16) if k in got] == list(range(8, 16))              # the kept block's appended keys are in


def test_candidates_hold_the_full_scan_answer():
    """When the true top-k is inside the candidate set the candidate selection IS the full scan's selection, bit for bit."""
    DS.EVERY = 3
    q, w, keys = world()
    plan, em, idxr = emit([CAP - 39, CAP - 38])
    st = SimpleNamespace(rows=[row := object()])
    base = DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, T, None, K)       # the selection of the previous round (same keys)
    put(st, 3, row, base[0], CAP - 39)                                              # session 0: the first row's selection
    put(st, 4, row, base[0], CAP - 39)
    got = DS.layer(st, 4, idxr, keys, None, q, w, em, T, plan)
    assert got is not None
    full = DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, T, None, K)
    gp, real = DS.build(mx.stack([base[0], base[0]]), mx.array([CAP - 39] * 2, dtype=mx.int32), em.lens_arr, T, None, 2)
    cand = set(np.array(gp)[0][np.array(real)[0]].tolist())
    fv = [i for i in np.array(full[0]).tolist() if i >= 0]
    ov = sum(1 for i in fv if i in cand)
    assert ov == len(fv) and np.array_equal(np.array(got[0]), np.array(full[0]))


def test_overlap_counts_the_common_keys():
    a = mx.array([[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1]], dtype=mx.int32)
    b = mx.array([[1, 2, 3, 9, 10, 11], [7, 8, -1, -1, -1, -1]], dtype=mx.int32)
    assert DS.overlap(a, b, 32) == [3, 0]
    assert DS.overlap(a, a, 32) == [6, 0]


def test_check_counter_reports_the_overlap_per_layer():
    """One served run of dec_sel_check gives the recall of the lever: ov<layer> / slots<layer> in engage."""
    DS.EVERY = 3
    DS.CHECK = True
    q, w, keys = world()
    plan, em, idxr = emit([CAP - 39, CAP - 38])
    st = SimpleNamespace(rows=[row := object()])
    base = DS.full_select(q, w, keys, mx.array([CAP - 39], dtype=mx.int32), mx.array([0], dtype=mx.int32), T, None, K)
    DS.store_of(st)[(id(row), 9)] = DS.Entry(CAP - 39, base[0], 0, row)              # the exact selection of the previous round
    got = DS.layer(st, 9, idxr, keys, None, q, w, em, T, plan)
    assert got is not None and DS.HITS["check"] == 1
    assert DS.STATS["slots9"] == 2 * K
    full = DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, T, None, K)        # the reference the check ran
    assert DS.STATS["ov9"] == sum(DS.overlap(got, full, T)) and DS.STATS["ov9"] >= K  # the entry's keys are all in the answer
    assert DS.STATS["reuse9"] == 2


# --------------------------------------------------------------------------------------------------------------------- rounds and rollback
def test_rounds_between_full_scans():
    """P = rounds between full scans: a full scan, P - 1 reused rounds, a full scan."""
    DS.EVERY = 3
    q, w, keys = world(rows=1)
    plan, em, idxr = emit([CAP - 1])
    st = SimpleNamespace(rows=[row := object()])
    full = DS.full_select(q, w, keys, em.lens_arr, plan.sess_idx, T, None, K)
    DS.publish(DS.store_of(st), 2, st.rows, (full, em.lens, True), None)            # round 1: the stock full scan, published at the commit
    rounds = []
    for _ in range(6):
        got = DS.layer(st, 2, idxr, keys, None, q, w, em, T, plan)
        rounds.append(got is None)
        if got is None:
            DS.publish(DS.store_of(st), 2, st.rows, (full, em.lens, True), None)
        else:
            DS.publish(DS.store_of(st), 2, st.rows, (got, em.lens, False), None)
    assert rounds == [False, False, True, False, False, True], rounds                 # full every 3rd round


def test_rollback_publishes_the_kept_row():
    """A commit that keeps row 0 of a 3-row pass publishes row 0's key count and selection: a rejected draft leaves nothing behind."""
    plan, em, idxr = emit([100, 101, 102])
    st = SimpleNamespace(rows=[row := object()])
    sel = mx.arange(3 * K, dtype=mx.int32).reshape(3, K)                            # row r holds r * K + i: which row was published is visible
    DS.publish(DS.store_of(st), 6, st.rows, (sel, em.lens, True), last=[0])         # the commit accepted row 0 only
    e = DS.store_of(st)[(id(row), 6)]
    assert e.pos == 100 and np.array_equal(np.array(e.idx), np.arange(K))


def test_a_rewind_forces_a_full_scan():
    """The committed count moved below the entry's: the keys it references are gone, so the layer scans."""
    DS.EVERY = 5
    q, w, keys = world(rows=1)
    plan, em, idxr = emit([CAP - 20])
    st = SimpleNamespace(rows=[row := object()])
    put(st, 7, row, mx.arange(K, dtype=mx.int32), CAP - 10)                         # the entry was taken at 502 keys
    em2 = SimpleNamespace(lens=[CAP - 21], lens_arr=mx.array([CAP - 21], dtype=mx.int32), tmax=CAP - 21)   # the row is back at 491
    assert DS.layer(st, 7, idxr, keys, None, q, w, em2, CAP, plan) is None
    assert DS.HITS["full"] == 1


def test_session_switch_forces_a_full_scan():
    """A row the cache holds no entry for (a session that joined, or the same slot reused by a new request) scans."""
    DS.EVERY = 5
    q, w, keys = world(rows=1)
    plan, em, idxr = emit([CAP - 20])
    a, b = object(), object()
    st = SimpleNamespace(rows=[a])
    put(st, 8, a, mx.arange(K, dtype=mx.int32), CAP - 21)
    assert DS.layer(st, 8, idxr, keys, None, q, w, em, T, plan) is not None        # the same row reuses
    st2 = SimpleNamespace(rows=[b], dec_sel=st.dec_sel)                             # the batch switched to another session
    assert DS.layer(st2, 8, idxr, keys, None, q, w, em, T, plan) is None
    assert DS.HITS["reuse"] == 1 and DS.HITS["full"] == 1


def test_a_session_without_an_entry_forces_the_whole_layer_to_scan():
    """Two sessions of one pass: one missing entry is enough (one dispatch, one path)."""
    DS.EVERY = 5
    q, w, keys = world(rows=2, S=2)
    plan = C.Plan([CAP - 20, CAP - 20], [1, 1])
    em = SimpleNamespace(lens=[CAP - 20, CAP - 20], lens_arr=mx.array([CAP - 20, CAP - 20], dtype=mx.int32), tmax=CAP - 20)
    idxr = SimpleNamespace(index_topk=K, is_candidate_source=False, uses_candidates=False, n_heads=NH, index_head_dim=DI)
    a, b = object(), object()
    st = SimpleNamespace(rows=[a, b])
    put(st, 11, a, mx.arange(K, dtype=mx.int32), CAP - 21)
    assert DS.layer(st, 11, idxr, keys, None, q, w, em, T, plan) is None
    put(st, 11, b, mx.arange(K, dtype=mx.int32), CAP - 21)
    assert DS.layer(st, 11, idxr, keys, None, q, w, em, T, plan) is not None


def test_the_candidate_source_keeps_its_full_scan():
    """The source layer's answer is the full-width block mask of the consumers: a candidate set cannot rebuild it."""
    DS.EVERY = 5
    q, w, keys = world(rows=1)
    plan, em, idxr = emit([CAP - 20])
    idxr.is_candidate_source = True
    st = SimpleNamespace(rows=[row := object()])
    put(st, 12, row, mx.arange(K, dtype=mx.int32), CAP - 21)
    assert DS.layer(st, 12, idxr, keys, None, q, w, em, T, plan) is None
    assert DS.HITS.get("reuse", 0) == 0


# --------------------------------------------------------------------------------------------------------------------- the flags
def test_flags_are_runtime_switchable():
    assert FL.FLAGS["dec_sel_every"].kind == "int" and FL.FLAGS["dec_sel_every"].default == "0"
    assert FL.FLAGS["dec_sel_check"].default == "0"
    was = (DS.EVERY, DS.CHECK)
    try:
        FL.set_many({"dec_sel_every": 3, "dec_sel_check": 1})
        assert DS.EVERY == 3 and DS.CHECK is True                                   # the module attribute the core reads per call
        FL.set_many({"dec_sel_every": 0, "dec_sel_check": 0})
        assert DS.EVERY == 0 and DS.CHECK is False
    finally:
        DS.EVERY, DS.CHECK = was


# --------------------------------------------------------------------------------------------------------------------- the batched core, end to end
def mini_model(seed=3):
    """A tiny DS4.1 language model on the batched core (head width 128: what `attend` is built for; the whole model in fp16: the index path's
    kernels take half scores, as a real pack does) and the (`rows.enable`) patch that routes a batched decode step through `dsv4_core.core`."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M                                                                      # noqa: E402
    from mlx.utils import tree_map                                                             # noqa: E402
    from kyojin_mlx.archs import DS41                                                         # noqa: E402
    from kyojin_mlx import rows as R                                                          # noqa: E402

    lm, cfg = M.make(head_dim=128, dtype=mx.float16, seed=seed)
    lm.update(tree_map(lambda a: a.astype(mx.float16), lm.parameters()))                       # the pack's indexer weights are half
    mx.eval(lm.parameters())

    class Wrap:                                                                               # rows.enable wants the model object, the tests hold the language model
        def __init__(self, lm):
            self.language_model = lm

        def __getattr__(self, n):
            return getattr(self.language_model, n)

    m = Wrap(lm)
    R.enable(m, DS41, grouped=False)
    return m, R, M


def decode(m, R, ids, toks, every, check=False, steps=20):
    """Prefill once, then `steps` plain batched steps, greedy. Returns (logits [steps, vocab], tokens, HITS, STATS)."""
    DS.EVERY, DS.CHECK = every, check
    DS.HITS.clear()
    DS.STATS.clear()
    cache, lg = R.prefill(m, ids)
    bc = R.batch([cache])
    tok = int(np.array(lg).argmax(-1)[0])
    out, got = [], []
    for _ in range(steps):
        l = R.step(m, bc, mx.array([tok]))
        tok = int(np.array(l.argmax(-1)[0]))
        out.append(np.array(l))
        got.append(tok)
    return np.concatenate(out, 0), got, dict(DS.HITS), dict(DS.STATS), bc


def test_mini_flag_0_and_every_1_are_identical():
    """EVERY = 1 runs the bookkeeping and takes the full scan every round: the logits are the flag-0 logits, bit for bit."""
    m, R, M = mini_model()
    was = C.RT_SELECT
    C.RT_SELECT = True                                                             # the runtime-width path (an internal switch, no arithmetic)
    try:
        ids = M.tokens(200)
        a, ta, ha, _, _ = decode(m, R, ids, None, 0)
        b, tb, hb, _, bc = decode(m, R, ids, None, 1)
    finally:
        C.RT_SELECT = was
        DS.EVERY = 0
    assert ha.get("reuse", 0) == 0 and hb.get("reuse", 0) == 0                     # neither arm takes the candidate path
    assert np.array_equal(a, b) and ta == tb, "EVERY = 1 moved the bits"
    assert len(bc[0].dec_sel) > 0                                                  # the bookkeeping ran (age 0 entries, all published)


def test_mini_reuse_engages_and_the_check_counts_the_overlap():
    """EVERY = 3: the plain and consumer layers reuse, the candidate source keeps its full scan, and dec_sel_check reports the overlap per layer."""
    m, R, M = mini_model()
    was = C.RT_SELECT
    C.RT_SELECT = True
    try:
        ids = M.tokens(200)
        _, _, h, s, _ = decode(m, R, ids, None, 3, check=True, steps=18)
    finally:
        C.RT_SELECT = was
        DS.EVERY, DS.CHECK = 0, False
    assert h["reuse"] > 0 and h["full"] > 0 and h["check"] > 0
    assert s["reuse6"] > 0 and s["full6"] > 0 and s["ov6"] <= s["slots6"]           # layer 6 is the consumer: reused and checked
    assert s.get("reuse4", 0) == 0 and s["full4"] > 0                               # layer 4 is the candidate source: never reused
    assert 0 <= s["ov2"] <= s["slots2"]


def test_mini_rollback_publishes_the_kept_row():
    """A verify pass whose second row is rejected publishes the FIRST row's key count: the cache never holds a selection of a dropped row."""
    m, R, M = mini_model()
    was = C.RT_SELECT
    C.RT_SELECT = True
    DS.EVERY, DS.CHECK = 4, False
    try:
        DS.HITS.clear()
        cache, lg = R.prefill(m, M.tokens(200))
        bc = R.batch([cache])
        tok = int(np.array(lg).argmax(-1)[0])
        p0 = bc[0].rows[0].offset
        p = R.verify(m, bc, [[tok, 11, 12, 13]])                                    # four rows, one session
        mx.eval(p.logits)
        p.commit([0])                                                               # only the pending token is kept: three drafts rejected
        assert bc[0].rows[0].offset == p0 + 1
        ents = {l: e for (_, l), e in bc[0].dec_sel.items()}
        assert 6 in ents and ents[6].pos == (p0 + 1) // 1                           # the kept row's count, not the last row's
        assert ents[6].age == 0                                                     # that round ran a full scan (no entry yet): published as such
    finally:
        C.RT_SELECT = was
        DS.EVERY = 0


def test_sessions_in_high_slots_read_their_own_entries():
    """mu_keep: the entries are per session of the pass, `keys` per buffer slot. A slot above the session count changes the buffer, not the answer."""
    DS.EVERY = 5
    q, w, keys = world(rows=2, S=2)
    em = SimpleNamespace(lens=[CAP - 20, CAP - 20], lens_arr=mx.array([CAP - 20, CAP - 20], dtype=mx.int32), tmax=CAP - 20)
    idxr = SimpleNamespace(index_topk=K, is_candidate_source=False, uses_candidates=False, n_heads=NH, index_head_dim=DI)
    a, b = object(), object()
    st = SimpleNamespace(rows=[a, b])
    put(st, 11, a, mx.arange(K, dtype=mx.int32), CAP - 21)
    put(st, 11, b, mx.arange(K, dtype=mx.int32) + K, CAP - 21)
    ref = DS.layer(st, 11, idxr, keys, None, q, w, em, T, C.Plan([CAP - 20, CAP - 20], [1, 1]))
    st = SimpleNamespace(rows=[a, b])
    put(st, 11, a, mx.arange(K, dtype=mx.int32), CAP - 21)
    put(st, 11, b, mx.arange(K, dtype=mx.int32) + K, CAP - 21)
    high = mx.concatenate([mx.zeros_like(keys[:1]), keys])
    got = DS.layer(st, 11, idxr, high, None, q, w, em, T, C.Plan([CAP - 20, CAP - 20], [1, 1], slots=[1, 2], nslots=3))
    assert ref is not None and np.array_equal(np.array(got), np.array(ref))

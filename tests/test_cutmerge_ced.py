"""Flag pf_cut_merge=3: the merged pass keeps the CED path (decoder half over tail + the rows past the cut). Mini DS4.1 model with CED on (tail 8, window 8, midpoint
layer 4 of 8): the cut state and the greedy continuation of the merged pass against the split path's two chunks (a CED chunk up to the cut, a short chunk after it)."""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_cutmerge import MiniBackend, TOL, diffs, greedy, no_fq, prefill, rel, run, split_and_merged, engine, CHUNK, WIN  # noqa: E402,F401
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C, kvdisk as K  # noqa: E402


@pytest.fixture(autouse=True)
def mode3(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "3")


def decoder_ring_diff(c1, c2, mid):
    return max(rel(c1[0].window[l], c2[0].window[l]) for l in range(mid, len(c1[0].window)))


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("cut,L,start", [(37, 40, 0),      # CED chunk to the cut (k > tail), cut inside a compressor group, 3-row tail
                                          (36, 41, 0),      # cut on a group edge
                                          (24, 29, 0),      # k > tail with k - tail = 16
                                          (50, 55, 33),     # pass starts at 33: history in ring and pool, k = 17
                                          (9, 14, 0),       # k = 9 just above tail (1 row skipped by the decoder)
                                          (6, 12, 0),       # k <= tail <= n: no CED pass (neither split chunk is one)
                                          (20, 21, 0)])     # one-row tail
def test_ced_cut_state_and_continuation_equal_the_split_path(packed, cut, L, start, monkeypatch):
    ids = M.tokens(L)
    (bs, cs, fs, lgs, _, _), (bm, cm, fm, lgm, _, _) = split_and_merged(dict(ced=True, packed=packed), ids, cut, start)
    n = bs.lm.config.num_hidden_layers
    d_cut, d_fin = diffs(cs, cm, n), diffs(fs, fm, n)
    mid = C.state(bs.lm).lay.mid
    print(f"packed={packed} cut={cut} L={L} start={start}: cut {d_cut} final {d_fin} dec ring cut {decoder_ring_diff(cs, cm, mid)} logits {rel(lgs, lgm)}")
    assert max(d_cut.values()) < TOL and max(d_fin.values()) < TOL and rel(lgs, lgm) < TOL
    assert int(cm[0].offset) == cut and int(fm[0].offset) == L
    assert int(mx.argmax(lgs, axis=-1).item()) == int(mx.argmax(lgm, axis=-1).item())
    more = M.tokens(6, seed=9)
    for b, c in ((bs, cs), (bm, cm)):
        b.last = b.prefill_chunk(b.restore(b.snapshot(c)), more)
    assert rel(bs.last, bm.last) < TOL
    assert greedy(bs, bs.restore(bs.snapshot(cs)), 5) == greedy(bm, bm.restore(bm.snapshot(cm)), 5)
    assert greedy(bs, fs, 5) == greedy(bm, fm, 5)


def test_ced_pass_is_a_ced_pass_with_the_decoder_rows_cut(monkeypatch):
    """The merged pass really skips the decoder half of the early rows (stats), and the decoder rows are tail + the rows past the cut."""
    ids = M.tokens(40)
    be = MiniBackend(ced=True)
    c = be.make_cache()
    st = C.state(be.lm)
    before = dict(st.stats)
    be.prefill_chunk_cut(c, ids, 37)
    assert st.stats["chunks"] == before["chunks"] + 1 and st.stats["rows_cut"] - before["rows_cut"] == 40 - (8 + 3)


def test_ced_snapshot_holds_no_row_past_the_cut():
    ids = M.tokens(40)
    other = ids[:37] + M.tokens(3, seed=77)
    (_, _, _, _, _, _), (b1, c1, *_) = split_and_merged(dict(ced=True), ids, 37)
    (_, _, _, _, _, _), (b2, c2, *_) = split_and_merged(dict(ced=True), other, 37)
    assert max(diffs(c1, c2, b1.lm.config.num_hidden_layers).values()) < TOL
    assert decoder_ring_diff(c1, c2, C.state(b1.lm).lay.mid) < TOL


def test_ced_drafter_ring_at_the_cut(monkeypatch):
    monkeypatch.setattr("test_cutmerge.KEEP", 8)                   # tail >= keep, as on the real model (the tap layers of both halves then hold the same last rows)
    ids = M.tokens(40)
    (bs, _, _, _, ds, dfs), (bm, _, _, _, dm, dfm) = split_and_merged(dict(ced=True, spec=True), ids, 37, spec=True)
    assert ds.rows[0].shape == dm.rows[0].shape and rel(ds.rows[0], dm.rows[0]) < TOL and rel(dfs.rows[0], dfm.rows[0]) < TOL


def test_ced_fallbacks(monkeypatch):
    be = MiniBackend(ced=True)
    cache = be.make_cache()
    assert be.cut_merge_ok(cache, 40)
    C.state(be.lm).force = True
    assert not be.cut_merge_ok(cache, 40)                          # n == tail would be a CED pass under force: split
    C.state(be.lm).force = False
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "1")
    assert not be.cut_merge_ok(cache, 40)                          # mode 1 still falls back on a CED pass
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "0")
    assert not be.cut_merge_ok(cache, 8)


def engine_run(ids, cuts, merge, monkeypatch, **kw):
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "3" if merge else "0")
    be = kw.pop("be", None) or MiniBackend(ced=True)
    eng = engine(be, disk=kw.pop("disk", None), block=kw.pop("block", CHUNK), chunk=kw.pop("chunk", CHUNK))
    s, lg = prefill(eng, ids, cuts)
    return be, eng, s, lg


def test_ced_engine_snapshots_and_next_turn_hit(monkeypatch):
    ids = M.tokens(40)
    be0, e0, s0, lg0 = engine_run(ids, [37], False, monkeypatch)
    be1, e1, s1, lg1 = engine_run(ids, [37], True, monkeypatch)
    assert e1.stats["pf_cut_merge"] == 1 and e1.stats["n_pf_chunk"] == 1 and e0.stats["n_pf_chunk"] == 2
    assert sorted(e1.prefix.entries) == sorted(e0.prefix.entries) and s1.path == s0.path == (37, 40) and rel(lg0, lg1) < TOL
    n = be0.lm.config.num_hidden_layers
    k37 = next(k for k in e0.prefix.entries if k[0] == 37)
    assert max(diffs(e0.prefix.entries[k37][0], e1.prefix.entries[k37][0], n).values()) < TOL
    nxt = ids[:37] + M.tokens(15, seed=4)
    got = []
    for eng in (e0, e1):
        s2, lg2 = prefill(eng, nxt, [37])
        assert s2.cached == 37
        got.append(lg2)
    assert rel(got[0], got[1]) < TOL
    assert int(mx.argmax(got[0], axis=-1).item()) == int(mx.argmax(got[1], axis=-1).item())


def test_ced_multi_cut_block_boundary_and_long_tail(monkeypatch):
    ids = M.tokens(57)                                              # (the tail after the cut is <= the CED tail of this model, 8; on DS4.1 it is <= 64 < 128)
    be0, e0, s0, lg0 = engine_run(ids, [20, 50], False, monkeypatch)
    be1, e1, s1, lg1 = engine_run(ids, [20, 50], True, monkeypatch)
    assert e1.stats["pf_cut_merge"] == 1 and sorted(e1.prefix.entries) == sorted(e0.prefix.entries) and s1.path == s0.path == (20, 50, 57)
    assert rel(lg0, lg1) < TOL
    ids = M.tokens(71)                                              # cut on the block boundary 64: not merged (pass boundary already); with chunk 128 it is
    _, e, _, _ = engine_run(ids, [64], True, monkeypatch)
    assert "pf_cut_merge" not in e.stats
    be0, e0, _, lg0 = engine_run(ids, [64], False, monkeypatch)
    be2, e2, _, lg2 = engine_run(ids, [64], True, monkeypatch, chunk=128, block=128)
    assert e2.stats["pf_cut_merge"] == 1 and 64 in {k[0] for k in e2.prefix.entries} and rel(lg0, lg2) < TOL
    _, e, s, _ = engine_run(M.tokens(150), [37], True, monkeypatch, chunk=256, block=256)      # tail 113 > 64
    assert "pf_cut_merge" not in e.stats


def test_ced_resume_from_the_disk_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")
    ids = M.tokens(40)

    def disk(be):
        return K.KvDisk(str(tmp_path), {"arch": "mini"}, be.make_cache, block=CHUNK)
    be0 = MiniBackend(ced=True)
    be0.disk_namespace = lambda: {"arch": "mini"}
    engine_run(ids, [37], True, monkeypatch, disk=disk(be0), be=be0)
    be1 = MiniBackend(ced=True)
    be1.lm.update(be0.lm.parameters())
    mx.eval(be1.lm.parameters())
    be1.disk_namespace = lambda: {"arch": "mini"}
    nxt = ids[:37] + M.tokens(15, seed=4)
    _, eng, s2, lg2 = engine_run(nxt, [37], False, monkeypatch, disk=disk(be1), be=be1)
    assert s2.cached == 37
    ref = MiniBackend(ced=True)
    ref.lm.update(be0.lm.parameters())
    mx.eval(ref.lm.parameters())
    _, _, _, lgr = engine_run(nxt, [37], False, monkeypatch, be=ref)
    assert rel(lg2, lgr) < TOL

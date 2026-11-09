"""Flag pf_cut_merge: one prefill pass across the last chat cut, cut state captured inside the pass. Tiny random DS4.1 text model (stock mlx-vlm modules, any MLX
backend), through the real Engine prefill loop and PrefixCache / KvDisk. The merged pass differs from the split path by the M of the GEMMs only (rounding), so the
states are compared to a relative tolerance and the greedy continuations of the two caches must agree."""
import os
import sys
import types

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ced as C, cutmerge as CM, fp4pool, kvdisk as K, rows as R, spec as SP  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402
from kyojin_mlx.serve.backends import Backend, VlmBackend  # noqa: E402
from kyojin_mlx.serve.engine import Engine  # noqa: E402

SLOTS = ("window", "compress", "keys", "kv_state", "score_state")
CHUNK = 64
WIN = 8                    # fake drafter ring rows
KEEP = 12                  # tap rows a plain pass keeps


class Ctx:
    def __init__(self):
        self.rows, self.lazy = [None], []


class FakeDrafter:
    """A ring of the last WIN tap rows (stage 0), like the DSpark head's per-stage kv ring: a row depends on its tap and position only."""
    tap_ids = [2, 5]
    window = WIN
    ctx = None

    def new_ctx(self):
        return Ctx()

    def append(self, taps, start):
        t = mx.concatenate([a.astype(mx.float32) for a in taps], axis=-1)
        r = self.ctx.rows[0]
        self.ctx.rows = [(t if r is None else mx.concatenate([r, t], axis=0))[-WIN:]]


class MiniBackend(Backend):
    """The Engine on the mini model with the real VlmBackend merge methods."""
    kind = "mini"
    cut_merge_ok = VlmBackend.cut_merge_ok
    prefill_chunk_cut = VlmBackend.prefill_chunk_cut
    _tap_keep = VlmBackend._tap_keep
    _ring = staticmethod(VlmBackend._ring)
    _free_handoff = VlmBackend._free_handoff
    draft_restore = VlmBackend.draft_restore

    def __init__(self, ced=False, packed=False, spec=False, **over):
        self.lm, _ = M.make(sliding_window=8, **over)
        if ced:
            C.enable(self.lm, DS41, tail=8)
        self.packed = packed
        self.spec = spec
        self.drafter = FakeDrafter()
        self.HL = __import__("kyojin_mlx.headlast", fromlist=["x"])
        if spec:
            SP.install_taps(types.SimpleNamespace(language_model=self.lm), FakeDrafter.tap_ids, keep=KEEP)

    def make_cache(self):
        c = self.lm.make_cache()
        return fp4pool.wrap_cache(c) if self.packed else c

    def fork(self, cache):
        return R.fork(cache)

    def snapshot(self, cache):
        return R.snapshot(cache)

    def restore(self, cache):
        return R.restore(cache)

    def _forward_kw(self, ids, vl):
        return {"input_ids": mx.array(ids)[None]}

    def _flush_lazy(self, ds):
        pass

    def _feed(self, d, taps, start):
        self.drafter.ctx = d
        self.drafter.append([taps[l] for l in self.drafter.tap_ids], start)

    def draft_snapshot(self, d):
        return None if d is None else list(d.rows)

    def new_draft(self):
        return self.drafter.new_ctx()

    def prefill_chunk(self, cache, ids, draft=None, skip_dec=False, vl=None):
        if draft is not None:
            cache[0].taps = {}
            p = cache[0].offset
        lg = self.lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32)
        if draft is not None:
            taps = cache[0].taps
            del cache[0].taps
            n = taps[next(iter(taps))].shape[1]
            self._feed(draft, {l: t[0] for l, t in taps.items()}, p + len(ids) - n)
        mx.eval(lg, *self.state(cache), *self._ring(draft))
        self.last = lg
        return lg

    def prefill_chunk_cut(self, cache, ids, k, draft=None):
        out = VlmBackend.prefill_chunk_cut(self, cache, ids, k, draft=draft)
        self.last = out[0]
        return out


def engine(be, disk=None, block=CHUNK, chunk=CHUNK):
    e = Engine(be, sessions=1, ctx=4096, chunk=chunk, block=block, prefix_bytes=1 << 28, disk=disk)
    return e


def prefill(eng, ids, cuts=()):
    eng.submit(list(ids), max_new=4, cuts=cuts)
    s = eng._inbox.pop()
    eng.prefilling.append(s)
    eng.backend.last = None
    while not s.prefilled:
        eng._prefill_step()
    return s, eng.backend.last


def rel(a, b, lenient=False):
    """Max relative difference (to the larger absolute value of the pair) of two arrays; inf on a shape mismatch / one-sided None (lenient: a one-sided None counts 0:
    the unsplit pass keeps no group state when the prompt ends on a group edge, the chunked one keeps a full one)."""
    if a is None or b is None:
        return 0.0 if (a is None and b is None) or lenient else float("inf")
    if a.shape != b.shape:
        return float("inf")
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.abs(a - b).max() / (mx.maximum(mx.abs(a), mx.abs(b)).max() + 1e-30)).item())


def diffs(c1, c2, n_layers, lenient=False):
    """{slot: max rel diff} of two DS4.1 caches, plus the Engram history and the offset."""
    e1, e2 = c1[0], c2[0]
    out = {s: max(rel(getattr(e1, s)[l], getattr(e2, s)[l], lenient) for l in range(n_layers)) for s in SLOTS}
    out["engram"] = rel(e1.engram, e2.engram)
    out["offset"] = 0.0 if int(e1.offset) == int(e2.offset) else float("inf")
    return out


TOL = 2e-2                  # fp32 mini model, fake quant off: the split path itself differs from the unsplit pass by ~3e-3 on the states (another M, the sorted MoE path, the router)


@pytest.fixture(autouse=True)
def no_fq(request):
    """The fp8 fake quant turns a last-bit difference (another M, the sorted MoE path from 64 routed rows) into a whole quantisation step: the split path itself differs
    from the unsplit pass by ~10 % on this model with it on. The comparisons to the split path run with it off; `test_merged_pass_is_the_unsplit_pass` keeps it on."""
    from kyojin_mlx import archs
    fq = archs.sub("fakequant")
    old = fq.DISABLE
    fq.DISABLE = not request.node.name.startswith("test_merged_pass_is_the_unsplit")
    yield
    fq.DISABLE = old


def greedy(be, cache, first, steps=8):
    toks, t = [], first
    for _ in range(steps):
        lg = be.lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
        t = int(mx.argmax(lg, axis=-1).item())
        toks.append(t)
    return toks


def split_and_merged(be_kw, ids, cut, start=0, spec=False):
    """(split: cut cache, final cache, logits, draft cut, draft final), (merged: same) on the same weights; optional first chunk [0, start) before both."""
    out, params = [], None
    for merged in (False, True):
        be = MiniBackend(**be_kw)
        if params is None:
            params = be.lm.parameters()
        else:
            be.lm.update(params)
            mx.eval(be.lm.parameters())
        cache = be.make_cache()
        draft = be.new_draft() if spec else None
        if start:
            be.prefill_chunk(cache, ids[:start], draft=draft)
        if merged:
            lg, cc, cd = be.prefill_chunk_cut(cache, ids[start:], cut - start, draft=draft)
        else:
            be.prefill_chunk(cache, ids[start:cut], draft=draft)
            cc = be.snapshot(cache)
            cd = None if draft is None else be.draft_restore(be.draft_snapshot(draft))
            lg = be.prefill_chunk(cache, ids[cut:], draft=draft)
        mx.eval(*be.state(cc))
        out.append((be, cc, cache, lg, cd, draft))
    return out


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
@pytest.mark.parametrize("cut,L,start", [(37, 60, 0),      # a cut inside a compressor group (odd) and a tail longer than the window
                                          (36, 60, 0),      # a cut on a group edge
                                          (64, 90, 0),      # a cut at a block boundary
                                          (50, 55, 33),     # the pass starts at 33: history in the ring and the pool
                                          (20, 21, 0)])     # a one-token tail
def test_cut_state_and_continuation_equal_the_split_path(packed, cut, L, start):
    ids = M.tokens(L)
    (bs, cs, fs, lgs, _, _), (bm, cm, fm, lgm, _, _) = split_and_merged(dict(packed=packed), ids, cut, start)
    n = bs.lm.config.num_hidden_layers
    # the unsplit pass (same rows, one chunk, no cut) is the noise floor of the model: another M moves the final state by that much
    bu = MiniBackend(packed=packed)
    bu.lm.update(bs.lm.parameters())
    mx.eval(bu.lm.parameters())
    uc = bu.make_cache()
    if start:
        bu.prefill_chunk(uc, ids[:start])
    lgu = bu.prefill_chunk(uc, ids[start:])
    d_cut, d_fin, d_floor = diffs(cs, cm, n), diffs(fs, fm, n), diffs(fs, uc, n, lenient=True)
    print(f"packed={packed} cut={cut} L={L} start={start}: cut state {d_cut} final {d_fin} floor {d_floor} logits {rel(lgs, lgm)} floor {rel(lgs, lgu)}")
    assert max(d_cut.values()) < TOL                                              # the state at the cut: the split path's own chunk-1 state, to rounding
    assert all(d_fin[k] <= (TOL if k in ("kv_state", "score_state") else 1.5 * d_floor[k] + 1e-5) for k in d_fin)       # the final state: as close to the split path's as the unsplit pass is
    assert rel(lgs, lgm) <= 1.5 * rel(lgs, lgu) + 1e-5
    nxt = int(mx.argmax(lgs, axis=-1).item())
    assert nxt == int(mx.argmax(lgm, axis=-1).item())
    # the next turn continues from the cut snapshot: greedy tokens of a continuation (new text after the cut) agree
    more = M.tokens(6, seed=9)
    for b, c in ((bs, cs), (bm, cm)):
        c2 = b.restore(b.snapshot(c))
        b.last = b.prefill_chunk(c2, more)
    assert rel(bs.last, bm.last) < TOL
    assert greedy(bs, bs.restore(bs.snapshot(cs)), 5) == greedy(bm, bm.restore(bm.snapshot(cm)), 5)


def test_the_cut_snapshot_holds_no_row_past_the_cut():
    """A poisoned tail: the cut state of a pass whose rows after the cut are arbitrary equals the one of another tail (only positions < cut count)."""
    ids = M.tokens(60)
    other = ids[:37] + M.tokens(23, seed=77)
    (_, _, _, _, _, _), (b1, c1, *_) = split_and_merged(dict(), ids, 37)
    (_, _, _, _, _, _), (b2, c2, *_) = split_and_merged(dict(), other, 37)
    d = diffs(c1, c2, b1.lm.config.num_hidden_layers)
    assert max(d.values()) < TOL, d


def test_drafter_ring_at_the_cut():
    ids = M.tokens(60)
    (bs, _, _, _, ds, dfs), (bm, _, _, _, dm, dfm) = split_and_merged(dict(spec=True), ids, 37, spec=True)
    assert ds.rows[0].shape == dm.rows[0].shape == (WIN, 2 * 64) and rel(ds.rows[0], dm.rows[0]) < TOL
    assert rel(dfs.rows[0], dfm.rows[0]) < TOL
    # a tail longer than the window keep: still the last WIN rows before the cut
    ids2 = M.tokens(100)
    (_, _, _, _, ds2, _), (_, _, _, _, dm2, _) = split_and_merged(dict(spec=True), ids2, 36, spec=True)
    assert rel(ds2.rows[0], dm2.rows[0]) < TOL


def test_ced_pass_runs_exact_in_mode_2_and_falls_back_in_mode_1(monkeypatch):
    ids = M.tokens(80)
    be = MiniBackend(ced=True)
    cache = be.make_cache()
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "1")
    assert not be.cut_merge_ok(cache, 80) and be.cut_merge_ok(cache, 8)       # a 80-row pass is a CED pass; 8 rows (== tail) is not
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "2")
    assert be.cut_merge_ok(cache, 80)
    lg, cc, _ = be.prefill_chunk_cut(cache, ids, 60)
    ref = MiniBackend(ced=False)
    ref.lm.update(be.lm.parameters())
    mx.eval(ref.lm.parameters())
    rc = ref.make_cache()
    ref.prefill_chunk(rc, ids[:60])
    assert max(diffs(cc, ref.snapshot(rc), 8).values()) < 5e-2
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "0")
    assert not be.cut_merge_ok(cache, 8)


# ----------------------------------------------------------------------------------------------------------------------------- the engine
def run(ids, cuts, merge, monkeypatch, disk=None, be=None, block=None, chunk=CHUNK, **kw):
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "1" if merge else "0")
    be = be or MiniBackend(**kw)
    eng = engine(be, disk=disk, block=block or chunk, chunk=chunk)
    s, lg = prefill(eng, ids, cuts)
    return be, eng, s, lg


def test_flag_off_is_the_split_path(monkeypatch):
    ids = M.tokens(60)
    be, eng, s, lg = run(ids, [37], False, monkeypatch)
    assert "pf_cut_merge" not in eng.stats and sorted(k[0] for k in eng.prefix.entries) == [37]


def unsplit_logits(ids, be):
    """Last-row logits of the one-pass prefill of `ids` on the weights of `be` (the noise floor of the model: another M moves them by that much)."""
    ref = MiniBackend(packed=be.packed)
    ref.lm.update(be.lm.parameters())
    mx.eval(ref.lm.parameters())
    return ref.prefill_chunk(ref.make_cache(), ids)


def close(be, ids, lg_split, lg_merged):
    return rel(lg_split, lg_merged) <= 1.5 * rel(lg_split, unsplit_logits(ids, be)) + 1e-5


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_engine_snapshots_and_next_turn_hit(packed, monkeypatch):
    ids = M.tokens(60)
    be0, e0, s0, lg0 = run(ids, [37], False, monkeypatch, packed=packed)
    be1, e1, s1, lg1 = run(ids, [37], True, monkeypatch, packed=packed)
    be1.lm.update(be0.lm.parameters())                                  # (same seed: the weights are equal already)
    assert e1.stats["pf_cut_merge"] == 1 and e1.stats["n_pf_chunk"] == 1 and e0.stats["n_pf_chunk"] == 2
    assert e1.stats["prefill_tokens"] == e0.stats["prefill_tokens"] == 60
    assert sorted(e1.prefix.entries) == sorted(e0.prefix.entries) and (37 in {k[0] for k in e1.prefix.entries})     # same keys (length, digest)
    assert s1.path == s0.path == (37, 60) and close(be0, ids, lg0, lg1)
    n = be0.lm.config.num_hidden_layers
    k37 = next(k for k in e0.prefix.entries if k[0] == 37)
    assert max(diffs(e0.prefix.entries[k37][0], e1.prefix.entries[k37][0], n).values()) < TOL
    # the next turn (same text up to the cut, new text after) hits the merged snapshot and continues like the split path's
    nxt = ids[:37] + M.tokens(15, seed=4)
    got = []
    for be, eng in ((be0, e0), (be1, e1)):
        s2, lg2 = prefill(eng, nxt, [37])
        assert s2.cached == 37
        got.append(lg2)
    assert close(be0, nxt, got[0], got[1])


def test_multi_cut_prompt_merges_the_last_cut_only(monkeypatch):
    ids = M.tokens(60)
    be0, e0, s0, lg0 = run(ids, [20, 50], False, monkeypatch)
    be1, e1, s1, lg1 = run(ids, [20, 50], True, monkeypatch)
    assert e1.stats["pf_cut_merge"] == 1 and sorted(e1.prefix.entries) == sorted(e0.prefix.entries) and s1.path == s0.path == (20, 50, 60)
    assert {20, 50} <= {k[0] for k in e1.prefix.entries} and close(be0, ids, lg0, lg1)


def test_cut_at_a_grid_point_is_not_merged(monkeypatch):
    """A cut on a chunk end (a block boundary is one) is a pass boundary already: the merged pass would cross a chunk end of the split path."""
    ids = M.tokens(80)
    be0, e0, _, lg0 = run(ids, [64], False, monkeypatch)
    be1, e1, _, lg1 = run(ids, [64], True, monkeypatch)
    assert "pf_cut_merge" not in e1.stats and sorted(e1.prefix.entries) == sorted(e0.prefix.entries) and rel(lg0, lg1) == 0.0
    # the same cut, with a bigger chunk: the cut is inside the chunk, the pass merges, and the snapshot lies on a block boundary of the engine's block (128)
    be2, e2, s2, lg2 = run(ids, [64], True, monkeypatch, chunk=128)
    assert e2.stats["pf_cut_merge"] == 1 and 64 in {k[0] for k in e2.prefix.entries} and close(be0, ids, lg0, lg2)


def test_long_tail_and_chunk_crossing_fall_back(monkeypatch):
    be, e, s, lg = run(M.tokens(150), [37], True, monkeypatch, chunk=256)                  # tail 113 > 64
    assert "pf_cut_merge" not in e.stats and s.path == (37, 150)
    be, e, s, lg = run(M.tokens(70), [37], True, monkeypatch)                              # [0, 70) crosses the chunk end at 64
    assert "pf_cut_merge" not in e.stats and s.path == (37, 64, 70)
    be, e, s, lg = run(M.tokens(60), [37], True, monkeypatch, chunk=32)                    # chunk ends 32, 37 (cut), 60: the pass [32, 60) holds the cut
    assert e.stats["pf_cut_merge"] == 1 and s.path == (32, 37, 60) and 37 in {k[0] for k in e.prefix.entries}
    be, e, s, lg = run(M.tokens(60), [37], True, monkeypatch)                              # control
    assert e.stats["pf_cut_merge"] == 1


def test_resume_from_the_disk_checkpoint_of_a_merged_cut(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")
    ids = M.tokens(60)

    def disk(be):
        return K.KvDisk(str(tmp_path), {"arch": "mini"}, be.make_cache, block=CHUNK)
    be0 = MiniBackend()
    be0.disk_namespace = lambda: {"arch": "mini"}
    d0 = disk(be0)
    run(ids, [37], True, monkeypatch, disk=d0, be=be0)
    assert sorted(b for b, _ in d0.index) == [37]
    # a new process: same directory, empty memory tier; the next turn restores the cut checkpoint
    be1 = MiniBackend()
    be1.lm.update(be0.lm.parameters())
    mx.eval(be1.lm.parameters())
    d1 = disk(be1)
    nxt = ids[:37] + M.tokens(15, seed=4)
    monkeypatch.setenv("KYOJIN_PF_CUT_MERGE", "0")
    eng = engine(be1, disk=d1)
    s2, lg2 = prefill(eng, nxt, [37])
    assert s2.cached == 37
    ref = MiniBackend()
    ref.lm.update(be0.lm.parameters())
    mx.eval(ref.lm.parameters())
    _, _, _, lgr = run(nxt, [37], False, monkeypatch, be=ref)
    assert rel(lg2, lgr) < TOL


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_merged_pass_is_the_unsplit_pass(packed):
    """With the fake quant ON (packed pool layers on the fp4 grid): the merged pass equals the one-pass prefill of the same rows (its final cache and logits), and its
    cut state has the pool / key rows of the first `cut` positions only."""
    ids = M.tokens(60)
    be = MiniBackend(packed=packed)
    full_c = be.make_cache()
    full = be.prefill_chunk(full_c, ids)
    c = be.make_cache()
    lg, cc, _ = be.prefill_chunk_cut(c, ids, 37)
    n = be.lm.config.num_hidden_layers
    assert rel(full, lg) < 1e-5 and max(diffs(full_c, c, n, lenient=True).values()) < 1e-5
    assert int(cc[0].offset) == 37 and int(c[0].offset) == 60
    for l in (2, 3, 4):
        assert (cc[0].compress[l] is None) or cc[0].compress[l].shape[1] == 37 // (2 if l < 4 else 1)
    assert cc[0].engram is None or cc[0].engram.shape[1] == 37

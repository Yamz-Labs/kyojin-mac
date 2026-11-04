"""Copy path v2 (flag copy_v2): matching on token lists, draft length from agreement, end of match, mismatch / spent match, the
copy > block > plain route and the lazy ring commit. No model and no GPU."""
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import copyv2 as CV
from kyojin_mlx import pld as PLD
from kyojin_mlx.serve.backends import BatchedPass, VlmBackend
from kyojin_mlx.serve.engine import Engine
from test_spec_lookup import FakeLookup, nxt8, repeated_prompt, run


def span_stream(n_lead=6, span=40, gap=5, tail=18):
    """lead + SPAN + gap + the SPAN's first `tail` tokens: the committed tail agrees `tail` tokens with the first SPAN."""
    lead = [900 + i for i in range(n_lead)]
    sp = [(i * 13 + 7) % 800 for i in range(span)]
    mid = [850 + i for i in range(gap)]
    return lead + sp + mid + sp[:tail], sp, tail


def index_of(ids):
    ix = PLD.Index(256)
    ix.extend(ids[:-1])
    return ix, ids[-1]


# ------------------------------------------------------------------------------------------------ matching
def test_match_on_own_tokens_agreement_and_draft():
    ids, sp, tail = span_stream()
    ix, t1 = index_of(ids)
    p = CV.plan(ix, t1, room=100, row_cap=24, block=5)
    assert p is not None and p.agree == tail                           # tokens agreeing backwards from the round's first token
    first = 6                                                           # the earlier occurrence starts after the 6 lead tokens
    assert p.draft == (sp + ids[6 + len(sp):])[tail:tail + p.k]        # what followed the earlier occurrence, past the span too


def test_below_the_agreement_floor_is_not_a_copy():
    ids, _, _ = span_stream(tail=10)
    ix, t1 = index_of(ids)
    assert ix.match_best(t1, 8).suffix < CV.MIN_AGREE
    assert CV.plan(ix, t1, 100, 24, 5) is None                       # the weaker lookup gate is bypassed: block / plain takes the round
    assert CV.plan(ix, t1, 100, 24, 5, floor=8) is not None          # the floor is a parameter (flag copy_v2_min)


def test_no_4gram_no_copy():
    ix, _ = index_of(list(range(50)))
    assert CV.plan(ix, 12345, 100, 24, 5) is None


def test_end_of_match_falls_back_in_the_same_round():
    """The source has no more than a block left (a tight loop: the earlier occurrence is 3 tokens behind the tail): not a copy, the block route
    takes the same round; with a one-row block the same source is a copy."""
    ids = [11, 12, 13] * 9
    ix, t1 = index_of(ids)
    m = ix.match_best(t1, 23)
    assert m is not None and m.suffix >= CV.MIN_AGREE and len(m.draft) <= 3
    assert CV.plan(ix, t1, 100, 24, block=5) is None
    assert CV.plan(ix, t1, 100, 24, block=1) is not None


def test_room_caps_the_draft():
    ids, _, _ = span_stream()
    ix, t1 = index_of(ids)
    p = CV.plan(ix, t1, room=3, row_cap=24, block=2)
    assert p is not None and p.k <= 3
    assert CV.plan(ix, t1, room=0, row_cap=24, block=2) is None


def test_max_len_flag(monkeypatch):
    assert CV.max_len(24) == 23
    monkeypatch.setenv("KYOJIN_COPY_V2_MAX", "15")
    assert CV.max_len(24) == 15 and CV.max_len(8) == 7
    monkeypatch.setenv("KYOJIN_COPY_V2_MIN", "20")
    assert CV.min_agree() == 20


# ------------------------------------------------------------------------------------------------ length from agreement
def test_survival_and_expected_accept():
    assert CV.survival(16, 16) == 0.5 and CV.survival(0, 1) == 0.0 and CV.survival(5, 0) == 1.0
    e = [CV.expected_accept(a, 15) for a in (16, 32, 64)]
    assert e[0] < e[1] < e[2] < 15                                   # more agreement, more expected accepts, never more than drafted
    assert CV.expected_accept(16, 1) == pytest.approx(16 / 17)


def test_length_grows_with_agreement_and_is_bounded():
    ks = [CV.choose_len(a, 23, 23) for a in (16, 32, 64)]
    assert ks == sorted(ks) and ks[-1] > ks[0] > 0
    assert CV.choose_len(64, 7, 23) <= 7 and CV.choose_len(64, 23, 5) <= 5
    assert CV.choose_len(64, 0, 23) == 0


def test_block_route_wins_when_it_pays_more_per_ms():
    # measured: a copy round at 9 drafts costs 400 ms against a 30 ms block round landing 4.6 tokens
    assert CV.choose_len(16, 23, 23, lookup_ms=lambda k: 400.0, mtp_ms=30.0, mtp_width=5, mtp_ema=3.6) == 0
    # a cheap copy round beats the same block round
    assert CV.choose_len(64, 23, 23, lookup_ms=lambda k: 30.0 + 2.0 * k, mtp_ms=45.0, mtp_width=5, mtp_ema=2.6) > 0


# ------------------------------------------------------------------------------------------------ spent match / mismatch
def test_spent_match_shortens_the_next_copy_and_a_full_landing_restores():
    s = 1.0
    s = CV.update_scale(s, 23, 1)                                    # landed 1 of 23: a spent match
    assert s == 0.5
    s = CV.update_scale(s, 23, 1)
    s = CV.update_scale(s, 23, 0)
    assert s == CV.SCALE_FLOOR                                        # bounded
    assert CV.update_scale(0.5, 10, 6) == 0.5                        # landed over half: unchanged
    assert CV.update_scale(0.25, 10, 10) == 1.0
    assert CV.update_scale(0.5, 0, 0) == 0.5
    ids, _, _ = span_stream(tail=24)
    ix, t1 = index_of(ids)
    full = CV.plan(ix, t1, 100, 24, 5, scale=1.0)
    spent = CV.plan(ix, t1, 100, 24, 5, scale=0.5)
    assert full is not None and (spent is None or spent.k <= full.k)
    low = CV.plan(ix, t1, 100, 24, 5, scale=CV.SCALE_FLOOR)
    assert low is None or low.k < full.k                             # the least confident scale drafts strictly shorter (or hands over)


def test_mismatch_breaks_the_match():
    ids, sp, tail = span_stream()
    ix, t1 = index_of(ids)
    assert CV.plan(ix, t1, 100, 24, 5) is not None                   # the clean stream copies
    # the round's first token is not the source's next token: the 4-gram does not occur, no copy and no stale span
    bad = ids[:-1] + [(ids[-1] + 1) % 800 + 1000]
    ixb, tb = index_of(bad)
    assert ixb.match_best(tb, 8) is None and CV.plan(ixb, tb, 100, 24, 5) is None
    # a mismatch inside the agreement window: the 4-gram still matches, the agreement stops at the mismatch and is under the floor
    mid = list(ids)
    mid[-9] = 5000
    ixm, tm = index_of(mid)
    m = ixm.match_best(tm, 8)
    assert m is not None and m.suffix == 8 and CV.plan(ixm, tm, 100, 24, 5) is None
    assert CV.plan(ixm, tm, 100, 24, 5, floor=8) is not None        # the floor is what refuses it


# ------------------------------------------------------------------------------------------------ engine route
def _eng(monkeypatch, **env):
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Engine(FakeLookup(), sessions=1, ctx=4096, chunk=4, block=8, prefix_bytes=0, mtp_depth=4)


def _sess(ids):
    return SimpleNamespace(ids=ids, out=[nxt8(ids)], n_prompt=len(ids), n_gen=1, max_new=100, last=nxt8(ids),
                           lk_ema=0.0, mt_ema=0.0, lk_streak=False, lookup_index=None)


def test_route_copy_then_block_then_plain(monkeypatch):
    eng = _eng(monkeypatch, KYOJIN_COPY_V2="1")
    ids = repeated_prompt()
    s = _sess(ids)
    depths, lookups = eng._lookup_plan([s], [4])                     # strong match, source long enough: copy
    assert lookups and depths[0] == lookups[0][0] > 4 and eng.stats["spec_copy_nomatch"] == 0
    other = _sess([(i * 11 + 5) % 997 for i in range(40)])           # no match: the planned block depth stays
    depths, lookups = eng._lookup_plan([other], [4])
    assert depths == [4] and not lookups and eng.stats["spec_copy_nomatch"] == 1
    depths, lookups = eng._lookup_plan([_sess([(i * 11 + 5) % 997 for i in range(40)])], [0])   # no block either: plain (depth 0)
    assert depths == [0] and not lookups


def test_flag_off_keeps_the_lookup_gate(monkeypatch):
    eng = _eng(monkeypatch)
    s = _sess(repeated_prompt())
    eng._lookup_plan([s], [4])
    assert eng.stats["spec_copy_nomatch"] == 0 and not hasattr(s, "cp_scale")


def test_copy_v2_run_matches_serial_and_traces_source(monkeypatch, tmp_path):
    ids = repeated_prompt()
    trace = tmp_path / "rl.jsonl"
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    monkeypatch.setenv("KYOJIN_COPY_V2", "1")
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "1")
    monkeypatch.setenv("KYOJIN_ROUND_LOG_PATH", str(trace))
    got, st, be = run({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    assert got["a"][1] == "length" and len(got["a"][0]) == 96
    assert st["spec_copy_rounds"] > 0 and st["spec_copy_drafted"] >= st["spec_copy_accepted"] > 0
    monkeypatch.setenv("KYOJIN_COPY_V2", "0")
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "0")
    plain, _, _ = run({"a": (ids, dict(max_new=96))}, mtp_depth=0)
    assert got["a"][0] == plain["a"][0]                               # greedy: the verify decides every token
    recs = [json.loads(x) for x in trace.read_text().splitlines()]
    srcs = {r["src"] for r in recs}
    assert "copy" in srcs and srcs <= {"copy", "block", "plain"}
    assert all(r["dlen"] == r["draft"] - r["cut"] for r in recs if r["src"] == "copy")


# ------------------------------------------------------------------------------------------------ lazy ring commit
class _Drafter:
    tap_ids = [3, 5]
    window = 4


def _rows(n, start):
    return {l: mx.array(np.full((n, 2), start * 10 + l, dtype=np.float32)) for l in _Drafter.tap_ids}


def _be():
    be = VlmBackend.__new__(VlmBackend)
    be.drafter = _Drafter()
    return be


def _flushed(be, d):
    """What `_flush_lazy` hands to the drafter for this state: (start, rows per tap) of the concatenated queue."""
    got = []
    be.drafter.append_many = lambda items: got.extend(items)
    be._flush_lazy([d])
    (dd, taps, start), = got
    return start, [np.array(t) for t in taps]


def test_defer_rows_equals_the_eager_feed_window():
    """Any mix of plain-step rows and copy-round chunks, flushed once, equals the last `window` rows an eager feed would have kept, at the
    right start position."""
    rng = np.random.default_rng(3)
    for sizes in ([2, 3, 4], [1, 1, 7, 2], [9], [3, 2, 2, 1, 1, 5, 4], [2, 6]):
        be, d = _be(), SimpleNamespace(lazy=[])
        pos, allrows = 100, []
        for i, n in enumerate(sizes):
            rows = {l: mx.array(rng.normal(size=(n, 2)).astype(np.float32)) for l in _Drafter.tap_ids}
            if n == 1 and i % 2:                                  # a plain decode step in between
                be._defer_feed(d, rows, pos)
            else:
                be._defer_feed_rows(d, rows, pos)
            allrows.append([np.array(rows[l]) for l in _Drafter.tap_ids])
            pos += n
        total = sum(sizes)
        start, got = _flushed(be, d)
        want = [np.concatenate([r[t] for r in allrows])[-_Drafter.window:] for t in range(2)]
        for t in range(2):
            assert got[t].shape[0] >= min(total, _Drafter.window)
            assert np.array_equal(got[t][-_Drafter.window:], want[t])      # the newest window rows, bit for bit
        assert start + got[0].shape[0] == 100 + total                       # contiguous up to the last committed position
        assert got[0].shape[0] <= _Drafter.window + 1 or sizes[-1] == 1     # bounded in rows (a plain step may add one past the bound)


def test_defer_rows_longer_than_the_window_drops_the_old_queue():
    be, d = _be(), SimpleNamespace(lazy=[])
    be._defer_feed_rows(d, _rows(2, 0), 10)
    be._defer_feed_rows(d, _rows(9, 2), 12)
    assert len(d.lazy) == 1 and d.lazy[0][0] == 12 + 9 - 4 and d.lazy[0][1][0].shape[0] == 4


def test_batched_commit_defers_only_the_flagged_sessions():
    calls = {"defer": [], "feed": []}
    be = SimpleNamespace(_defer_feed_rows=lambda d, t, p: calls["defer"].append((d, p)),
                         _feed_many=lambda items: calls["feed"].append([(d, p) for d, _, p in items]))
    ps = SimpleNamespace(commit=lambda acc: None)
    bp = BatchedPass.__new__(BatchedPass)
    bp.backend, bp.ps = be, ps
    bp.taps, bp.starts = {7: mx.zeros((8, 1, 2))}, [0, 4]
    bp.commit([1, 2], ["d0", "d1"], [100, 200], defer=[True, False])
    assert calls["defer"] == [("d0", 100)] and calls["feed"] == [[("d1", 200)]]
    calls["defer"].clear(); calls["feed"].clear()
    bp.commit([1, 2], ["d0", "d1"], [100, 200])                       # default: both fed eagerly, as before
    assert not calls["defer"] and calls["feed"] == [[("d0", 100), ("d1", 200)]]

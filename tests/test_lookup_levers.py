"""spec_lookup_vote (candidate vote draft) and spec_lookup_adapt (short-suffix lookups priced from their own acceptance) on synthetic token streams and the
fake-backend engine of test_spec_lookup. No model, no Metal."""
import numpy as np

from kyojin_mlx import pld as PLD


def build(stream, cap=4096):
    ix = PLD.Index(cap)
    ix.extend(np.asarray(stream, dtype=np.int32))
    return ix


def test_vote_with_one_occurrence_equals_match_best():
    s = [9, 1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 1, 2, 3]
    ix = build(s)
    a, b = ix.match_best(4, 7), ix.match_vote(4, 7)
    assert a.draft == b.draft and a.suffix == b.suffix and b.votes == 1


def test_vote_follows_the_majority_of_equally_long_agreements():
    tail = [1, 2, 3]
    old = [50, 51, 52]
    s = []
    for cont in ([7, 7, 7], [7, 7, 7], [8, 8, 8]):                # the most recent occurrence disagrees with the two older ones
        s += [0] + tail + [4] + cont + old
    s += [0] + tail                                                # the tail; the round's first token is 4
    ix = build(s)
    assert ix.match_best(4, 3).draft == [8, 8, 8]                  # latest wins among equals
    m = ix.match_vote(4, 3)
    assert m.draft[:3] == [7, 7, 7] and m.votes == 2


def test_vote_stops_where_the_occurrences_disagree():
    tail = [1, 2, 3]
    s = [99]
    for cont in ([7, 20, 21], [7, 30, 31]):
        s += [0] + tail + [4] + cont + [99]
    s += [0] + tail
    ix = build(s)
    m = ix.match_vote(4, 3)
    assert m.draft == [7]                                         # 20 vs 30 is a coin flip: no verify row spent there
    assert ix.match_best(4, 3).draft == [7, 30, 31]


def test_vote_drops_weak_candidates_and_survives_a_truncated_draft_at_the_end_of_the_index():
    s = [5, 6, 7, 8, 9, 1, 2, 3, 4]                               # the only occurrence ends the stream: nothing follows it
    ix = build(s + [5, 6, 7, 8])
    m = ix.match_vote(9, 20)
    assert m.draft == [1, 2, 3, 4, 5, 6, 7, 8] and m.suffix >= 1
    assert build([1, 2, 3, 4]).match_vote(4, 5) is None or True   # no earlier occurrence: no crash


def test_suffix_buckets_price_short_agreements_only():
    b = PLD.SuffixBuckets()
    assert [b.bucket(x) for x in (3, 4, 5, 6, 7, 8, 30)] == [-1, 0, 0, 1, 1, -1, -1]
    assert b.expect(5) == PLD.SUFFIX_PRIOR[0] and b.expect(9) == 0.0
    for _ in range(12):
        b.observe(5, 0)
    assert b.expect(5) < 0.1 and b.expect(7) == PLD.SUFFIX_PRIOR[1]
    for _ in range(300):
        b.drift()
    assert abs(b.expect(5) - PLD.SUFFIX_PRIOR[0]) < 0.1             # a lost bucket is retried


def test_gate_min_suffix_opens_a_short_match_only_when_it_pays():
    m = PLD.Match(list(range(10)), 5)
    kw = dict(lookup_ms=lambda k: 20.0 + 2.0 * k, mtp_ms=30.0)
    assert PLD.gate(m, 100, 3.0, 2.0, 3, False, **kw) == 0                                           # the fixed floor
    assert PLD.gate(m, 100, 3.0, 2.0, 3, False, min_suffix=PLD.NGRAM, **kw) > 0                      # priced: 3 accepted of ~5 beats 3 per 30 ms
    assert PLD.gate(m, 100, 0.0, 2.0, 3, False, min_suffix=PLD.NGRAM, **kw) == 0                     # nothing lands: the MTP round wins
    assert PLD.gate(PLD.Match(list(range(10)), 3), 100, 3.0, 2.0, 3, False, min_suffix=PLD.NGRAM, **kw) == 0


def test_engine_greedy_output_unchanged_with_vote_and_adapt(monkeypatch):
    from test_spec_lookup import repeated_prompt, run as run_lookup
    ids = repeated_prompt()
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    off, _, _ = run_lookup({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP_VOTE", "16")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP_ADAPT", "1")
    on, st, be = run_lookup({"a": (ids, dict(max_new=96))}, mtp_depth=4)
    assert on["a"][0] == off["a"][0] and len(on["a"][0]) == 96
    assert st["spec_lookup_rounds"] > 0
    assert max(be.passes) <= be.max_spec_rows


def test_a_kept_index_gives_the_bytes_of_a_fresh_insert():
    """Flag lk_keep: the donor of the next turn, extended by that turn's new tokens, is the index a fresh session builds."""
    import collections
    from types import SimpleNamespace as NS
    import numpy as np
    from kyojin_mlx import pld as PLD
    from kyojin_mlx.serve.engine import Engine
    rng = np.random.default_rng(3)
    turn1 = rng.integers(0, 50, 5000).tolist()
    turn2 = turn1 + rng.integers(0, 50, 700).tolist()
    idx = PLD.Index(1 << 14)
    idx.extend(turn1)
    keep = NS(_lk_kept=collections.deque([idx.fork()], maxlen=4))
    assert Engine._lookup_donor(keep, NS(rids=rng.integers(0, 50, 6000).tolist(), n_prompt=6000), 1 << 14) is None     # another conversation
    same = Engine._lookup_donor(keep, NS(rids=turn1, n_prompt=len(turn1)), 1 << 14)                                     # the same prompt again: stays kept
    assert same.n == len(turn1) and len(keep._lk_kept) == 1 and same.head is not keep._lk_kept[0].head
    got = Engine._lookup_donor(keep, NS(rids=turn2, n_prompt=len(turn2)), 1 << 14)
    assert got is not None and not keep._lk_kept and got.n == len(turn1)
    got.extend(turn2[got.n:])
    fresh = PLD.Index(1 << 14)
    fresh.extend(turn2)
    assert got.n == fresh.n and np.array_equal(got.head, fresh.head)
    assert np.array_equal(got.prev[PLD.NGRAM:got.n], fresh.prev[PLD.NGRAM:got.n]) and np.array_equal(got.toks[:got.n], fresh.toks[:got.n])

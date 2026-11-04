"""kyojin_mlx.serve.hostloop: batched detokenization, queue drain, GC / switch-interval / profile levers, flag table. No Metal, no mlx."""
import gc
import os
import queue
import random
import sys

import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx.serve import hostloop as HL

SERVE = ("serve_coalesce", "serve_nodelay", "serve_gc", "serve_switch", "serve_prof")


class ByteApp:
    """Byte-level tokenizer: token t = one byte (ids >= 256 = a 3-byte CJK character), decode with replacement, like a byte BPE."""
    def decode(self, ids):
        b = bytearray()
        for t in ids:
            b += bytes([t]) if t < 256 else ("世界漢"[t - 256]).encode()
        return b.decode("utf-8", errors="replace")


class SpmApp:
    """SentencePiece style: a leading space of the decoded window is stripped, tokens carry their own space."""
    PIECES = ["▁The", "▁cat", "▁sat", ",", "▁on", "▁the", "▁mat", ".", "ing", "▁and"]
    def decode(self, ids):
        s = "".join(self.PIECES[t] for t in ids).replace("▁", " ")
        return s[1:] if s.startswith(" ") else s


class CleanupApp:
    """HF clean_up_tokenization_spaces: ' ,' -> ',' depends on the window, so a batch window can differ from the per-token windows."""
    PIECES = [" a", " ,", " b", " .", "x", " c"]
    def decode(self, ids):
        s = "".join(self.PIECES[t] for t in ids)
        return s.replace(" ,", ",").replace(" .", ".")


@pytest.fixture(autouse=True)
def fresh():
    HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False
    saved = (HL.COALESCE, HL.NODELAY, HL.GC, HL.SWITCH_US, HL.PROF)
    yield
    HL.COALESCE, HL.NODELAY, HL.GC, HL.SWITCH_US, HL.PROF = saved
    HL.gc_sync(); HL.switch_sync(); HL.bind(None)


def per_token(app, toks):
    d = HL.Detok(app)
    return "".join(d.push(t) for t in toks) + d.flush()


def batched(app, toks, sizes):
    d, out, i = HL.Detok(app), [], 0
    for n in sizes:
        out.append(d.push_many(toks[i:i + n])); i += n
    out.append(d.push_many(toks[i:]))
    return "".join(out) + d.flush()


def test_byte_level_batches_equal_per_token_text_with_split_characters():
    rng = random.Random(1)
    app = ByteApp()
    for _ in range(40):
        toks = []
        for _ in range(rng.randint(1, 60)):
            if rng.random() < 0.2:                               # a CJK character cut into its three bytes, spread over tokens
                toks += list("世".encode())
            else:
                toks.append(rng.randint(32, 126))
        sizes = [rng.randint(1, 6) for _ in range(rng.randint(0, 12))]
        assert batched(app, toks, sizes) == per_token(app, toks)


def test_a_batch_that_ends_inside_a_character_is_held_then_released():
    app, d = ByteApp(), HL.Detok(ByteApp())
    e = "世".encode()
    assert d.push_many([65, 66, e[0], e[1]]) == "AB"             # per-token: A, B emitted, the two bytes held
    assert d.push_many([e[2], 67]) == "世C"


def test_sentencepiece_leading_space_window():
    app = SpmApp()
    toks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    for sizes in ([2, 3, 1], [1, 1, 1], [5], [3, 3, 3]):
        assert batched(app, toks, sizes) == per_token(app, toks)


def test_context_dependent_decode_keeps_the_per_token_text():
    app = CleanupApp()
    rng = random.Random(3)
    for _ in range(50):
        toks = [rng.randint(0, 5) for _ in range(rng.randint(1, 30))]
        sizes = [rng.randint(1, 4) for _ in range(6)]
        HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False
        assert batched(app, toks, sizes) == per_token(app, toks)


class WindowApp:
    """A decode whose text depends on the window length (a marker appended once the window holds 4+ ids): the batch text differs from the per-token text."""
    def decode(self, ids):
        return "".join(chr(t) for t in ids) + ("?" if len(ids) >= 4 else "")


def test_a_tokenizer_that_decodes_by_window_switches_batching_off_and_keeps_the_per_token_text():
    app = WindowApp()
    toks = list(b"abcdefghijkl")
    want = per_token(app, toks)
    assert batched(app, toks, [4, 4, 2]) == want
    assert HL.BATCH_BAD is True                                  # found out in the verified batches, per token from then on
    assert batched(app, toks, [6]) == want


def test_batch_helper_is_off_after_the_verify_window():
    HL.VERIFY_LEFT = 0
    app = ByteApp()
    toks = list(b"hello world, this is a test")
    assert batched(app, toks, [5, 5, 5]) == "hello world, this is a test"


class Raw:
    def feed(self, text):
        return [("content", text)] if text else []

    def finish(self):
        return []


def text_of(ev):
    return "".join(e[1] for e in ev if e[0] == "content")


def test_reply_push_many_matches_push_without_stops():
    app = ByteApp()
    toks = list(b"The quick brown fox jumps over the lazy dog")
    a, b = HL.Reply(app, Raw(), []), HL.Reply(app, Raw(), [])
    ea = [x for t in toks for x in a.push(t)] + a.finish()
    eb = b.push_many(toks[:7]) + b.push_many(toks[7:30]) + b.push_many(toks[30:]) + b.finish()
    assert text_of(ea) == text_of(eb) and a.n_tokens == b.n_tokens == len(toks)


def test_drain_events_keeps_one_frame_per_token():
    """serve_coalesce (30/10 regression): feeding the drained batch to push_many merged several tokens into one delta event, so
    an OpenAI client that counts delta frames read 299 tokens where 514 were sent (and looked 2x slower). The batch is still one
    write, but drain_events must yield one event per token, like the non-coalesced path."""
    app = ByteApp()
    toks = list(b"The quick brown fox jumps over the lazy dog")
    a, b = HL.Reply(app, Raw(), []), HL.Reply(app, Raw(), [])
    ea = [x for t in toks for x in a.push(t)] + a.finish()
    eb = HL.drain_events(b, toks) + b.finish()
    assert text_of(ea) == text_of(eb)
    assert len(eb) == len(toks) and b.n_tokens == len(toks)


def test_reply_stop_string_counts_tokens_like_push():
    app = ByteApp()
    toks = list(b"abc<END>garbage after")
    a, b = HL.Reply(app, Raw(), ["<END>"]), HL.Reply(app, Raw(), ["<END>"])
    ea = [x for t in toks for x in a.push(t)] + a.finish()
    eb = b.push_many(toks[:4]) + b.push_many(toks[4:]) + b.finish()
    assert a.stopped and b.stopped and a.n_tokens == b.n_tokens == 8
    assert text_of(ea) == text_of(eb) == "abc"


def test_drain_takes_the_queued_tokens_and_stops_at_done():
    q = queue.SimpleQueue()
    for t in (2, 3, 4):
        q.put(("tok", t))
    q.put(("done", "stop", {"x": 1}))
    q.put(("tok", 9))
    toks, tail = HL.drain(q, ("tok", 1), queue.Empty)
    assert toks == [1, 2, 3, 4] and tail == ("done", "stop", {"x": 1})
    assert q.get_nowait() == ("tok", 9)                          # nothing behind the done event was taken
    q.put(("tok", 9))
    toks, tail = HL.drain(q, ("tok", 5), queue.Empty)
    assert toks == [5, 9] and tail is None


def test_new_queue_follows_the_flag():
    HL.COALESCE = True
    assert isinstance(HL.new_queue(), type(queue.SimpleQueue()))
    HL.COALESCE = False
    assert isinstance(HL.new_queue(), queue.Queue)


def test_switch_interval_applies_and_zero_restores_the_start_value():
    start = sys.getswitchinterval()
    try:
        HL.SWITCH_US = 500
        HL.switch_sync()
        assert abs(sys.getswitchinterval() - 500e-6) < 1e-7
        HL.SWITCH_US = 0
        HL.switch_sync()
        assert sys.getswitchinterval() == start
    finally:
        sys.setswitchinterval(start)


def test_gc_lever_freezes_and_restores_thresholds():
    before = gc.get_threshold()
    try:
        HL.GC = True
        HL.gc_sync()
        assert gc.get_threshold() == (HL.GC_GEN0, HL.GC_GEN1, HL.GC_GEN2) and gc.get_freeze_count() > 0
        HL.idle()                                                # a collection, once per idle
        HL.GC = False
        HL.gc_sync()
        assert gc.get_threshold() == before and gc.get_freeze_count() == 0
    finally:
        gc.unfreeze()
        gc.set_threshold(*before)


def test_profile_marks_sections_and_gc_pauses():
    import collections
    st = collections.Counter()
    HL.PROF = True
    try:
        HL.bind(st)
        HL.reset()
        sum(range(1000)); HL.mark("a")
        sum(range(1000)); HL.mark("b")
        HL.mark("a")
        gc.collect()
        assert st["n_hp_a"] == 2 and st["n_hp_b"] == 1 and st["t_hp_a"] > 0
        assert st["n_hp_gc"] >= 1 and st["t_hp_gc"] > 0
        HL.idle()                                                # the next mark only starts a chain (the engine waited)
        HL.mark("c")
        assert st["n_hp_c"] == 0
    finally:
        HL.PROF = False
        HL.bind(st)
    HL.mark("a")                                                 # off: nothing is added
    assert st["n_hp_a"] == 2


def test_mark_is_free_when_off():
    import collections
    st = collections.Counter()
    HL.bind(st)
    HL.mark("x")
    assert not st


def test_flag_table_matches_the_module_literals_and_is_runtime_switchable():
    for n in SERVE:
        f = FL.FLAGS[n]
        assert f.default == "0" and f.runtime and f.env.startswith("KYOJIN_SERVE_")
        (mod, attr, fn), = f.attrs
        assert mod == "serve.hostloop" and getattr(HL, attr) == fn(f.default)
        assert f.doc
    src = open(HL.__file__).read()
    for n in SERVE:                                              # the module reads the same variable with the same fallback literal
        assert f'os.environ.get("{FL.FLAGS[n].env}", "0")' in src
    saved = {FL.FLAGS[n].env: os.environ.get(FL.FLAGS[n].env) for n in SERVE}
    try:
        FL.set("serve_coalesce", "1")
        assert HL.COALESCE is True
        FL.set("serve_switch", "250")
        assert HL.SWITCH_US == 250
        FL.set("serve_switch", "0")
        assert HL.SWITCH_US == 0
        FL.set("serve_coalesce", "0")
        assert HL.COALESCE is False
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

"""Loop guard of the served thinking block (kyojin_mlx.serve.loopguard, flag loop_guard): the three rules on synthetic text, the answer never watched, and the
real HTTP handler on a scripted engine (close of the think block, continuation request, a second event ends the reply). No model, no mlx op."""
import http.client
import json
import random
import re
import sys
import threading
import time
import types

import pytest

pytest.importorskip("jinja2")

try:
    import mlx.core  # noqa: F401
except Exception:
    m = types.ModuleType("mlx")
    c = types.ModuleType("mlx.core")
    c.array = object
    m.core = c
    sys.modules.setdefault("mlx", m)
    sys.modules.setdefault("mlx.core", c)

from kyojin_mlx import flags                                         # noqa: E402
from kyojin_mlx.serve import ds41_parse                              # noqa: E402
from kyojin_mlx.serve import hostloop as HL                          # noqa: E402
from kyojin_mlx.serve import loopguard as LG                         # noqa: E402
from kyojin_mlx.serve import server as SV                            # noqa: E402

# ------------------------------------------------------------------------------------------------ word tokenizer
VOC: dict = {}
INV: dict = {}


def wid(w):
    if w not in VOC:
        VOC[w] = len(VOC) + 10
        INV[VOC[w]] = w
    return VOC[w]


def words(text):
    return re.findall(r"</think>|\S+|\s+", text)


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [wid(w) for w in words(text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(INV[i] for i in ids)


def replay(text, guard=None):
    """Feed a text word by word like the server does; (first hit or None)."""
    g = guard or LG.ThinkGuard()
    for w in words(text):
        h = g.feed([wid(w)], w)
        if h:
            return h
    return None


def prose(n, seed=1):
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(400)]
    return "\n".join(" ".join(rnd.choice(vocab) for _ in range(rnd.randint(6, 18))) + "." for _ in range(n))


def longlines(n, seed=3):
    """n lines of 14-20 random words (always 40+ non-space chars), no shared template: the text rule cannot see a block made of them."""
    rnd = random.Random(seed)
    vocab = [f"v{i}" for i in range(2000)]
    return "".join(" ".join(rnd.choice(vocab) for _ in range(rnd.randint(14, 20))) + ".\n" for _ in range(n))


# ------------------------------------------------------------------------------------------------ the rules
def test_flag_registered_default_on(monkeypatch):
    f = flags.FLAGS["loop_guard"]
    assert f.env == "KYOJIN_LOOP_GUARD" and f.default == "1" and f.runtime
    monkeypatch.delenv("KYOJIN_LOOP_GUARD", raising=False)
    assert LG.enabled()
    monkeypatch.setenv("KYOJIN_LOOP_GUARD", "0")
    assert not LG.enabled()


def test_varied_thinking_never_fires():
    assert replay(prose(400)) is None


def test_period_rule():
    g = LG.ThinkGuard()
    ids = [5, 6, 7, 8, 9] * 40
    hit = None
    for t in ids:
        hit = g.feed([t], "") or hit
        if hit:
            break
    assert hit and hit["kind"] == "period" and hit["period"] == 5


def test_text_rule_catches_a_short_line_block_the_line_rule_misses():
    """30 short lines (under 40 chars each) repeated: no line is long enough, the text rule fires."""
    block = "".join(f'**"Item{i}"**: You tap. Hmm.\n\n' for i in range(30))
    hit = replay(prose(60) + "\n" + block * 8)
    assert hit and hit["kind"] in ("text", "period")


def test_line_rule_third_copy_and_offset():
    block = longlines(12)
    head = prose(40)
    text = head + "\n" + block * 4
    hit = replay(text)
    assert hit and hit["kind"] == "line"
    # three copies = the end of the third copy's first line is the earliest: the first line of the block
    first_end = len(head) + 1 + len(block.split("\n")[0]) + 1
    assert hit["at"] == first_end + 2 * len(block)


def test_line_rule_needs_40_non_space_chars_and_exact_match():
    short = "Hmm, let me think about this again.\n"                  # under 40 non-space chars
    assert replay(prose(30) + "\n" + short * 3 + prose(30, 2)) is None
    line = longlines(1, 8)
    assert replay(prose(30) + "\n" + line * 2 + prose(30, 2)) is None       # twice is not a loop
    assert replay(longlines(60, 9)) is None                                  # many long lines, none repeated
    assert replay(prose(30) + "\n" + line * 3)["kind"] == "line"


def test_answer_is_never_watched():
    """Repeated code lines in the answer: the guard only sees thinking text."""
    code = "".join("    ctx.fillRect(x + %d, y, 10, 10); ctx.strokeRect(x, y, 10, 10); // draw the cell\n" % 0 for _ in range(40))
    parser = ds41_parse.StreamParser(False, False, None)           # content mode
    lg = LG.ReplyGuard(parser)
    assert lg.on and lg.observe([1, 2, 3] * 100, [("content", code)], LG.in_think(parser)) is None
    assert len(lg.gen) == 300
    assert replay(code)["kind"] == "line"                           # ... while the same text inside thinking is a loop


def test_recovery_is_once():
    lg = LG.ReplyGuard(ds41_parse.StreamParser(True, False, None))
    assert lg.closes() is True and lg.closes() is False


def test_guard_off_by_flag(monkeypatch):
    monkeypatch.setenv("KYOJIN_LOOP_GUARD", "0")
    p = ds41_parse.StreamParser(True, False, None)
    lg = LG.ReplyGuard(p)
    assert not lg.on and lg.observe([1], [("reasoning", "x\n" * 5)], True) is None and not lg.eos_open("stop")


def test_eos_open_only_in_think_and_only_on_stop():
    p = ds41_parse.StreamParser(True, False, None)
    lg = LG.ReplyGuard(p)
    assert lg.eos_open("stop") and not lg.eos_open("length")
    p.mode = "content"
    assert not lg.eos_open("stop")


# ------------------------------------------------------------------------------------------------ the handler
LOOP = longlines(5, 4) * 4
ANSWER = "Final answer is here."


class Handle:
    def __init__(self):
        self.events = HL.new_queue()
        self.cancelled = threading.Event()
        self.session = None

    def cancel(self):
        self.cancelled.set()


class Engine:
    ctx = 100000
    max_step_gap = 0.0

    def __init__(self, scripts):
        self.scripts, self.calls = list(scripts), []

    def submit(self, ids, **kw):
        h = Handle()
        self.calls.append((list(ids), kw))
        text, reason = self.scripts.pop(0)
        threading.Thread(target=self._feed, args=(h, [wid(w) for w in words(text)], reason), daemon=True).start()
        return h

    @staticmethod
    def _feed(h, toks, reason):
        n = 0
        for t in toks:
            if h.cancelled.is_set():
                break
            h.events.put(("tok", t))
            n += 1
            if n % 8 == 0:
                time.sleep(0.002)
        h.events.put(("done", "cancelled" if h.cancelled.is_set() else reason,
                      {"prompt_tokens": 7, "completion_tokens": n, "cached_tokens": 0, "queue_s": 0.0, "ttft_s": 0.0, "total_s": 0.1}))


class Family:
    def __init__(self, parser_cls):
        self.parser_cls = parser_cls

    def render(self, body):
        return types.SimpleNamespace(prompt="Question please", images=None, cut_texts=[], thinking_first=True, parse_tools=False, tools=None)

    def parser(self, r):
        return self.parser_cls()


class StuckThinkParser:
    """Reports a think block that never closes (a second loop can happen)."""
    mode = "reasoning"

    def feed(self, text):
        return [("reasoning", text)] if text else []

    def finish(self):
        return []


def post(scripts, parser_cls, family=None, **body):
    eng = Engine(scripts)
    app = SV.App(eng, Tok(), (family or Family)(parser_cls), "m", {}, set(), log=lambda *a: None)
    srv = SV.Server(("127.0.0.1", 0), SV.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
        conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}], **body}), {"Content-Type": "application/json"})
        out = json.loads(conn.getresponse().read().decode())
    finally:
        srv.shutdown()
    return out, eng


def ds41_parser():
    return ds41_parse.StreamParser(True, False, None)


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv("KYOJIN_LOOP_GUARD", raising=False)
    HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False


@pytest.mark.parametrize("coalesce", [False, True])
def test_loop_in_thinking_closes_the_block_and_the_model_answers(coalesce):
    HL.COALESCE = coalesce
    prefix = "Let me think. " + prose(5)
    out, eng = post([(prefix + "\n" + LOOP * 2, "length"), (ANSWER, "stop")], ds41_parser)
    msg = out["choices"][0]["message"]
    assert len(eng.calls) == 2
    ids1, ids2 = eng.calls[0][0], eng.calls[1][0]
    close = Tok().encode("</think>")
    assert ids2[:len(ids1)] == ids1 and ids2[-len(close):] == close           # prompt + text so far + the template's close tag
    assert eng.calls[0][1]["max_new"] is None and eng.calls[1][1]["max_new"] is None
    assert msg["content"] == ANSWER and msg["reasoning_content"].startswith("Let me think.")
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == len(ids1)
    assert out["usage"]["completion_tokens"] == len(ids2) - len(ids1) + len(Tok().encode(ANSWER))


def test_loop_resume_respects_max_tokens():
    out, eng = post([(prose(5) + "\n" + LOOP * 2, "length"), (ANSWER, "stop")], ds41_parser, max_tokens=100000)
    assert len(eng.calls) == 2
    n_prev = len(eng.calls[1][0]) - len(eng.calls[0][0])
    assert eng.calls[1][1]["max_new"] == 100000 - n_prev


def test_eos_inside_an_open_think_block_forces_the_close_once():
    out, eng = post([("I am thinking and then the turn stops", "stop"), (ANSWER, "stop")], ds41_parser)
    msg = out["choices"][0]["message"]
    assert len(eng.calls) == 2 and msg["content"] == ANSWER
    assert msg["reasoning_content"] == "I am thinking and then the turn stops"


def test_eos_after_the_close_is_a_plain_stop():
    out, eng = post([("thought</think>The answer.", "stop")], ds41_parser)
    assert len(eng.calls) == 1 and out["choices"][0]["message"]["content"] == "The answer."


def test_second_loop_ends_the_reply_with_stop_and_the_text_so_far():
    loop = prose(5) + "\n" + LOOP * 2
    out, eng = post([(loop, "length"), (prose(5, 9) + "\n" + LOOP * 2, "length"), ("never used", "stop")], StuckThinkParser)
    msg = out["choices"][0]["message"]
    assert len(eng.calls) == 2                                      # one recovery, the second loop ends the reply
    assert out["choices"][0]["finish_reason"] == "stop" and "never used" not in json.dumps(out)
    assert msg["reasoning_content"].count(LOOP.split("\n")[0]) >= 4   # text of both parts is kept


def test_second_eos_in_think_ends_the_reply():
    out, eng = post([("first thoughts", "stop"), ("more thoughts", "stop")], StuckThinkParser)
    assert len(eng.calls) == 2 and out["choices"][0]["finish_reason"] == "stop"


def test_flag_off_leaves_a_looping_reply_alone(monkeypatch):
    monkeypatch.setenv("KYOJIN_LOOP_GUARD", "0")
    out, eng = post([(prose(5) + "\n" + LOOP * 2, "length")], ds41_parser)
    assert len(eng.calls) == 1 and out["choices"][0]["finish_reason"] == "length"


def test_line_rule_skips_fenced_code_in_thinking():
    from kyojin_mlx.serve.loopguard import ThinkGuard
    g = ThinkGuard()
    code = "ctx.globalCompositeOperation = 'source-over'; ctx.fillRect(0, 0, w, h);\n"
    text = "Let me sketch the draw loop.\n```js\n" + code * 3 + "```\n"
    assert g._lines(text) is None
    line = "I think I will go with the pulse mechanic for this game, final answer.\n"
    assert g._lines(line * 2) is None
    assert g._lines(line)["kind"] == "line"


# ------------------------------------------------------------------------------------------------ a prompt with images (the 09/11 12:18 incident)
class FakeVision:
    """Two image tokens (virtual ids, >= BASE) in front of the text; one span."""
    BASE = 10 ** 6

    def plan(self, rec):
        return rec

    def expand(self, ids, plans, whole=True):
        return ([self.BASE, self.BASE + 1] if whole else []) + list(ids), (["span"] if whole else None)

    def real_ids(self, ids):
        return [0 if t >= self.BASE else t for t in ids]


class ImageFamily(Family):
    vision = FakeVision()

    def render(self, body):
        r = super().render(body)
        r.images = [object()]
        return r


def test_recorded_image_turn_loop_fires_in_the_first_third():
    """Synthetic fixture shaped like a replayed incident: ~1.8K-char block of long lines looping from char ~0.9K, ~39K chars streamed."""
    import os
    text = open(os.path.join(os.path.dirname(__file__), "data", "loop_image_turn_reasoning.txt")).read()
    hit = replay(text)
    assert hit and hit["kind"] == "line" and hit["at"] < len(text) // 3


def test_recorded_image_turn_short_line_loop_fires_within_a_few_periods():
    """Synthetic fixture shaped like the second recorded loop: 'Let me do it.' style lines (under 40 chars: the period or text rule has to catch it)."""
    import os
    text = open(os.path.join(os.path.dirname(__file__), "data", "loop_image_turn_short_lines.txt")).read()
    hit = replay(text)
    assert hit and hit["kind"] in ("period", "text") and hit["at"] < 12500


def test_loop_guard_runs_on_a_prompt_with_images_and_resumes_with_the_spans():
    HL.COALESCE = True
    prefix = "Let me think. " + prose(5)
    out, eng = post([(prefix + "\n" + LOOP * 2, "length"), (ANSWER, "stop")], ds41_parser, family=ImageFamily)
    msg = out["choices"][0]["message"]
    assert len(eng.calls) == 2, "the loop guard must watch a reply to a prompt with images"
    (ids1, kw1), (ids2, kw2) = eng.calls
    close = Tok().encode("</think>")
    assert ids1[:2] == [FakeVision.BASE, FakeVision.BASE + 1] and ids2[:len(ids1)] == ids1 and ids2[-len(close):] == close
    assert kw1["vl"] == ["span"] and kw2["vl"] == ["span"]                      # the image spans ride on the resumed request
    assert kw2["real_ids"][:len(ids1)] == kw1["real_ids"] and len(kw2["real_ids"]) == len(ids2)   # real ids: one per id, the reply as sampled
    assert kw2["real_ids"][:2] == [0, 0] and kw2["real_ids"][2:] == ids2[2:]
    assert msg["content"] == ANSWER and out["choices"][0]["finish_reason"] == "stop"

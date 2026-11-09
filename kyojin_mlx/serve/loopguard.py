"""Loop guard for the thinking block of a served reply (flag loop_guard). Pure Python, CPU only, no model.

A sampled or greedy thinking block can fall into a verbatim loop that never ends by itself. Three rules watch the tokens and the text
of the block (the answer is not watched: valid code repeats lines):

  period  the token stream repeats with a period of at most MAX_PERIOD tokens for a sustained stretch
  text    the last TEXT_WINDOW chars compress to under TEXT_RATIO and a SPAN_LEN-char span came back SPAN_MIN times in the last SPAN_TAIL
          chars, and it holds for TEXT_SUSTAIN tokens (a loop that drifts, no exact period)
  line    a line of at least LINE_MIN non-space chars is seen LINE_REPEAT times (exact match after strip): a long-period loop, a block of
          dozens of lines that comes back whole

`ThinkGuard.feed(tokens, text)` returns None or {"kind", "at" (chars of thinking text when it fired), "tokens"}.
`Recovery` is the per-request state of the response: one forced close of the think block, a second loop ends the request."""
from __future__ import annotations

import os
import zlib
from collections import Counter

MAX_PERIOD = 64
PERIOD_K, PERIOD_MIN_SPAN = 4, 48          # a period n fires after max(n * (K - 1), MIN_SPAN - n) repeated tokens
CHECK_EVERY = 16                           # tokens between two evaluations of the text rule
TEXT_WINDOW, TEXT_RATIO, TEXT_MIN_CHARS = 1000, 0.30, 200
SPAN_TAIL, SPAN_LEN, SPAN_MIN = 3000, 50, 3
TEXT_SUSTAIN = 320
LINE_MIN, LINE_REPEAT = 40, 3


def enabled() -> bool:
    """Flag loop_guard (KYOJIN_LOOP_GUARD, default on), read per request."""
    return os.environ.get("KYOJIN_LOOP_GUARD", "1") not in ("0", "")


def text_loop(text: str) -> bool:
    w = text[-TEXT_WINDOW:]
    if len(w) < TEXT_MIN_CHARS or len(zlib.compress(w.encode())) / max(len(w.encode()), 1) >= TEXT_RATIO:
        return False
    t = text[-SPAN_TAIL:]
    return max(Counter(t[i:i + SPAN_LEN] for i in range(len(t) - SPAN_LEN + 1)).values(), default=0) >= SPAN_MIN


class ThinkGuard:
    def __init__(self):
        self.ids: list = []
        self.run = [0] * (MAX_PERIOD + 1)
        self.text = ""                      # tail of the thinking text for the text rule
        self.chars = 0                      # chars of thinking text seen
        self.line = ""                      # the unfinished line
        self.seen: Counter = Counter()      # long lines (stripped) -> times seen
        self.fence = False                  # inside a ``` code fence: code repeats its own lines, they are not counted
        self.last_check = 0
        self.flag_from: int | None = None

    def feed(self, tokens, text: str = "") -> dict | None:
        hit = self._lines(text)
        self.text += text
        self.chars += len(text)
        if len(self.text) > 4 * SPAN_TAIL:
            self.text = self.text[-2 * SPAN_TAIL:]
        if hit:
            return hit
        ids, run = self.ids, self.run
        for t in tokens:
            i = len(ids)
            ids.append(int(t))
            for n in range(1, MAX_PERIOD + 1):
                r = run[n] + 1 if i >= n and ids[i] == ids[i - n] else 0
                run[n] = r
                if r >= max(n * (PERIOD_K - 1), PERIOD_MIN_SPAN - n):
                    return {"kind": "period", "at": self.chars, "tokens": i + 1, "period": n}
        if len(ids) - self.last_check >= CHECK_EVERY:
            prev, self.last_check = self.last_check, len(ids)
            if text_loop(self.text[-(SPAN_TAIL + 200):]):
                if self.flag_from is None:
                    self.flag_from = prev
                if len(ids) - self.flag_from >= TEXT_SUSTAIN:
                    return {"kind": "text", "at": self.chars, "tokens": len(ids)}
            else:
                self.flag_from = None
        return None

    def _lines(self, text: str) -> dict | None:
        """Called before `chars` counts `text`. `at` = chars of thinking text up to the end of the line that completed the third copy."""
        if not text:
            return None
        pos = self.chars - len(self.line)             # offset of the unfinished line
        parts = (self.line + text).split("\n")
        self.line = parts.pop()
        for p in parts:
            pos += len(p) + 1
            q = p.strip()
            if q.startswith("```"):
                self.fence = not self.fence
                continue
            if not self.fence and len("".join(q.split())) >= LINE_MIN:
                self.seen[q] += 1
                if self.seen[q] >= LINE_REPEAT:
                    return {"kind": "line", "at": pos, "tokens": len(self.ids), "line": q[:80]}
        return None


class LineWatch:
    """Early loop signal (flag spec_typical_cooldown): a detector only, it never ends a reply. `feed(text)` is True when a long line (the `line`
    rule's normalisation: stripped, >= LINE_MIN non-space chars, outside ``` fences) completes for the 2nd time within the last WINDOW chars."""
    WINDOW = 16384

    def __init__(self):
        self.chars = 0
        self.line = ""
        self.last: dict = {}                # stripped line -> char offset where its last copy ended
        self.first: dict = {}               # stripped line -> char offset where the earliest copy of the current chain started
        self.first_at: int | None = None    # set by a feed() that returned True: where the first copy of the repeated line starts (flag spec_typical_rewind)
        self.fence = False

    def feed(self, text: str) -> bool:
        if not text:
            return False
        parts = (self.line + text).split("\n")
        pos = self.chars - len(self.line)
        self.line = parts.pop()
        self.chars += len(text)
        hit = False
        self.first_at = None
        for p in parts:
            beg = pos
            pos += len(p) + 1
            q = p.strip()
            if q.startswith("```"):
                self.fence = not self.fence
                continue
            if not self.fence and len("".join(q.split())) >= LINE_MIN:
                if pos - self.last.get(q, -10 ** 9) <= self.WINDOW:
                    if not hit:
                        self.first_at = self.first.get(q, beg)
                    hit = True
                else:
                    self.first[q] = beg
                self.last[q] = pos
        if len(self.last) > 512:
            lim = self.chars - self.WINDOW
            self.last = {k: v for k, v in self.last.items() if v >= lim}
            self.first = {k: v for k, v in self.first.items() if k in self.last}
        return hit


THINK_CLOSE = "</think>"                   # the chat templates of every family here (ds41, qwen, glm) close a think block with this text


def in_think(parser) -> bool:
    """The reply parser is inside a think block (ds41 / glm parsers carry `mode`, the qwen one wraps a splitter that does)."""
    mode = getattr(parser, "mode", None)
    if mode is None:
        mode = getattr(getattr(parser, "sp", None), "mode", None)
    return mode == "reasoning"


class ReplyGuard:
    """The guard of one served reply, driven by the HTTP thread that streams it.

    observe(toks, events, was_think) after every batch of tokens: "loop" when the thinking loops. closes() says whether the think block may
    still be force-closed (once per request); a second loop, or an EOS inside an open block after the close, is final."""

    def __init__(self, parser):
        self.parser = parser
        self.on = enabled() and (hasattr(parser, "mode") or hasattr(getattr(parser, "sp", None), "mode"))
        self.mon = ThinkGuard()
        self.gen: list = []                 # every token the reply produced so far (the prompt of a resumed request)
        self.used = False
        self.hit: dict | None = None

    def observe(self, toks, events, was_think: bool) -> str | None:
        self.gen.extend(int(t) for t in toks)
        if not self.on or not was_think:
            return None
        text = "".join(ev[1] for ev in events if ev[0] == "reasoning")
        self.hit = self.mon.feed(toks, text)
        if self.hit and not in_think(self.parser):           # the block closed inside this batch: nothing left to close
            self.hit = None
        return "loop" if self.hit else None

    def eos_open(self, reason: str) -> bool:
        """The engine ended the turn on a stop token while the think block is still open."""
        return self.on and reason == "stop" and in_think(self.parser)

    def closes(self) -> bool:
        """True once per request: the caller closes the think block and lets the model answer. False: the request ends."""
        if self.used:
            return False
        self.used = True
        return True

"""Flag tok_inc: a long prompt is tokenized from where it differs from a recent prompt. A chat client sends the whole conversation every turn, so a whole
tokenize costs 0.85 s at 1M tokens, and the chat layer does it twice (the prompt, then the cut before the generation header), under the one tokenizer lock.

Why the splice is the tokenize of the whole text: the tokenizer works left to right on segments (no normalizer on these models, added tokens and regex
splits look only at the text they match) and BPE merges stay inside a segment. So two runs over the same text that share a token boundary and then agree on
SYNC tokens stay together until the text changes. The new text restarts BACK tokens before the first differing char, and the old ids are kept up to the first
sync point; no sync point well before that char (a segment longer than the window, a tokenizer with other rules) = a whole tokenize."""
import os

import numpy as np

ON = os.environ.get("KYOJIN_TOK_INC", "1") not in ("0", "")
MIN_CHARS = 65536                                               # below this a whole tokenize takes ~15 ms
KEEP = 8                                                        # recent long prompts (one per live conversation), ~12 MB per 1M tokens
BACK = 256                                                      # tokens the restart sits before the first differing char
GUARD = 64                                                      # ... and the agreeing tokens must end this far before it
SYNC = 32
STEP = 65536
PROBE = "Hello world, 12345 le café 日本語 🙂\n\ndef f(x):\n    return x**2\n"
HITS = {"inc": 0, "whole": 0, "no_sync": 0}


def common_prefix(a: str, b: str) -> int:
    n, i = min(len(a), len(b)), 0
    while i < n and a[i:i + STEP] == b[i:i + STEP]:
        i += STEP
    if i >= n:
        return n
    hi = min(i + STEP, n)
    while i < hi:                                               # a[:i] == b[:i], and the first difference is below hi
        m = (i + hi + 1) // 2
        if a[i:m] == b[i:m]:
            i = m
        else:
            hi = m - 1
    return i


class _Entry:
    __slots__ = ("text", "ids", "starts")

    def __init__(self, text, ids, starts):
        self.text, self.ids, self.starts = text, ids, starts


class IncTok:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        self.entries = []
        try:                                                    # a tokenizer class that changes its encode keeps the plain path
            if self.backend is not None and self.backend.encode(PROBE, add_special_tokens=False).ids != list(tokenizer.encode(PROBE, add_special_tokens=False)):
                self.backend = None
        except Exception:                                       # noqa: BLE001
            self.backend = None

    def _run(self, text: str, base: int = 0):
        e = self.backend.encode(text, add_special_tokens=False)
        ids = np.array(e.ids, dtype=np.int32)
        starts = np.array([o[0] for o in e.offsets], dtype=np.int32) + base
        return ids, starts

    def _splice(self, e, text: str, lcp: int):
        i = int(np.searchsorted(e.starts, lcp, side="right")) - 1       # the token holding the first differing char
        b = i - BACK
        while b > 0 and e.starts[b - 1] == e.starts[b]:                 # a char split over tokens: restart on its first token
            b -= 1
        if b <= 0:
            return None
        ids, starts = self._run(text[int(e.starts[b]):], int(e.starts[b]))
        end = i - GUARD
        for j in range(min(BACK - GUARD - SYNC, len(ids) - SYNC)):
            m = int(np.searchsorted(e.starts, starts[j]))
            if m >= len(e.starts) or e.starts[m] != starts[j] or m + SYNC > end:
                if m + SYNC > end:
                    break
                continue
            if np.array_equal(ids[j:j + SYNC], e.ids[m:m + SYNC]) and np.array_equal(starts[j:j + SYNC], e.starts[m:m + SYNC]):
                return np.concatenate([e.ids[:m], ids[j:]]), np.concatenate([e.starts[:m], starts[j:]])
        HITS["no_sync"] += 1
        return None

    def encode(self, text: str) -> list:
        if not ON or self.backend is None or len(text) < MIN_CHARS:
            return self.tokenizer.encode(text, add_special_tokens=False)
        best, lcp = None, 0
        for e in self.entries:
            n = common_prefix(e.text, text)
            if n > lcp:
                best, lcp = e, n
        if best is not None and lcp == len(text) == len(best.text):      # the same text again
            self.entries.remove(best)
            self.entries.append(best)
            return best.ids.tolist()
        got = self._splice(best, text, lcp) if best is not None and lcp >= MIN_CHARS // 2 else None
        if got is None:
            HITS["whole"] += 1
            got = self._run(text)
        else:
            HITS["inc"] += 1
        if best is None or lcp < len(text):                              # a strict prefix of a kept text (a cut) adds nothing to keep
            self.entries = [e for e in self.entries if not (e is best and lcp == len(e.text))]
            self.entries = (self.entries + [_Entry(text, *got)])[-KEEP:]
        return got[0].tolist()

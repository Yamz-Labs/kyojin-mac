# SPDX-License-Identifier: Apache-2.0
"""Per-round trace of the speculative depth policy (flag `round_log`, default 0).

One JSON line per decode round: what the policy decided, on which measured inputs, and what the round then produced. The
point is to price a policy change from the round itself: `sum(tok) / sum(ms)` is the decoded tok/s of the traced rounds, so a
policy that plans the same depth on a faster engine can be told apart from one that reacts to the timing (and from one that
reacts to the text, which the traced token ids show).

The trace is off unless `KYOJIN_ROUND_LOG` is 1. It is bounded: at most `KYOJIN_ROUND_LOG_MAX` lines (default 20000), then the
sink closes; each record is one buffered line, no allocation per token. `KYOJIN_ROUND_LOG_PATH` defaults to
`scratch/roundlog.jsonl` under the server's working directory (the engine dir).

Records (short keys keep a line small; `-` means the field does not apply to that round kind):

    k      "r" (verify round) or "p" (plain step)
    sid    session id, ph = first 8 hex of the prompt's token hash, m = sessions in the pass
    r      rounds the session had emitted before this round
    pl     depth the MTP policy asked for (before the lookup plan), rows = rows of the pass this session planned
    dec    the MTP policy's label: cap (drafts at the cap), probe (a cap probe after plain rounds), plain, other (override /
           shared pass), - (plain round)
    draft  proposals the verify pass used, raw = before the confidence gate trimmed them, cut = rows the gate cut
    acc    drafts accepted, tok = tokens this round emitted (acc + 1 for a verify round), t = those token ids,
           ln = the round's pending token (the row before t; the reply's first token has no record, it is the prefill sample)
    src    flag copy_v2 only: the draft source of the round, copy (own-token match) / block (draft head) / plain (no draft); dlen = draft tokens verified
    cf     the draft head's raw block confidences of this round (the confidence gate's input; the reward curve's x axis)
    ms     wall ms of the round
    ct     s.cap_tok (tokens per cap round, the rate rule's acceptance side), cn = s.cap_n (its sample count),
    ir     s.idle_rounds, rg = s.rate_gap, pr = 1 while the round is a probe
    pm     engine.plain_ms (the rate rule's plain cost), cm = median of the measured cap-round ms (its drafted cost)
    asuc   s.acc_succ, atr = s.acc_trial (the per-position acceptance estimate of the other policies), sk = spec_k
"""
from __future__ import annotations

import json
import os

ENV = "KYOJIN_ROUND_LOG"
PATH_ENV = "KYOJIN_ROUND_LOG_PATH"
MAX_ENV = "KYOJIN_ROUND_LOG_MAX"
DEFAULT_PATH = "scratch/roundlog.jsonl"
DEFAULT_MAX = 20000


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "")


def median(v) -> float:
    v = sorted(v)
    return float(v[(len(v) - 1) // 2]) if v else 0.0


class Sink:
    """Append-only JSONL sink with a hard line bound (one open file, no growth past the bound)."""

    def __init__(self, path: str, limit: int):
        self.path, self.limit, self.n, self.f, self.closed = path, max(1, limit), 0, None, False

    def write(self, rec: dict) -> bool:
        if self.n >= self.limit:
            return False
        if self.f is None:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.f = open(self.path, "a", buffering=1)
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n += 1
        if self.n >= self.limit:
            self.f.write(json.dumps({"end": self.n}) + "\n")
            self.f.close()
            self.f, self.closed = None, True
        return True


_sink: Sink | None = None
_sink_key: tuple | None = None


def record(rec: dict, stats=None) -> bool:
    """Write one record when the flag is on. `stats` (an engine Counter, optional) gets the live totals the A/B reads back:
    rlog_rounds / rlog_tok / rlog_ms / rlog_draft / rlog_acc, so tok/s and accepted drafts per round can be read without the file
    (and keep counting after the file's line bound has closed the sink)."""
    global _sink, _sink_key
    if not enabled():
        return False
    if stats is not None:
        stats["rlog_rounds"] += 1
        stats["rlog_tok"] += int(rec.get("tok") or 0)
        stats["rlog_ms"] += float(rec.get("ms") or 0.0)
        stats["rlog_draft"] += int(rec.get("draft") or 0)
        stats["rlog_acc"] += int(rec.get("acc") or 0)
        if rec.get("k") == "p":
            stats["rlog_plain"] += 1
    path = os.environ.get(PATH_ENV) or DEFAULT_PATH
    try:
        limit = int(os.environ.get(MAX_ENV) or DEFAULT_MAX)
    except ValueError:
        limit = DEFAULT_MAX
    if _sink is None or _sink_key != (path, limit):
        _sink, _sink_key = Sink(path, limit), (path, limit)
    return _sink.write(rec)


def reset_sink() -> None:
    """Test hook: forget the open file (a new KYOJIN_ROUND_LOG_PATH takes effect on the next record anyway)."""
    global _sink, _sink_key
    if _sink is not None and _sink.f is not None:
        _sink.f.close()
    _sink, _sink_key = None, None

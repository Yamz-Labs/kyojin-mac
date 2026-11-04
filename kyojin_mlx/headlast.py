"""LM head on the last row only, for the callers that read nothing else from a prompt chunk.

The head of a prompt chunk costs 2 * hidden * vocab flops per row (DS4.1: 1.3 GFLOP, ~4 % of the prefill) plus a [rows, vocab] logits tensor. A
caller that reads  only (solo / multi-user / serve prefill) wraps its forward in ; chunks above LAST_ROW_MIN rows then
get [B, 1, vocab]. Callers that consume other prompt rows (gates, scoring, tools) call the model bare and keep every row. KYOJIN_HEAD_LAST=0 turns
the scope into a no-op (every row, as the stock model)."""
import os
from contextlib import contextmanager

ENABLED = os.environ.get("KYOJIN_HEAD_LAST", "1") != "0"
LAST_ROW_MIN = 64                                                         # chunks of at most this many rows (decode, verify, MTP) keep every row
_depth = 0


@contextmanager
def last_row():
    global _depth
    if not ENABLED:
        yield
        return
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1


def active() -> bool:
    return _depth > 0

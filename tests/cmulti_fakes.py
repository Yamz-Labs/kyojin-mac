"""Fake backends for the multi-session engine tests (cloud/cmulti-2). CPU, mlx only, no model.

GridFake: the next token hashes the token history AND the prefill chunk partition (every chunk's row count), so a changed chunk grid, a snapshot
reused off the session's own grid, or a split chunk changes the text. The partition is the part of the history the real model is sensitive to
(MoE / CED tiling depends on the rows of a chunk). Decode steps add no partition: the real decode is row invariant.
`calls` records every backend call, to compare the call trace of two runs. `pf_ms` / `step_ms`: simulated wall cost of one prefill row / one decode step."""
import time

import mlx.core as mx

from kyojin_mlx.serve.backends import Backend

V = 97


def nxt(hist, parts):
    h = sum((i + 3) * t for i, t in enumerate(hist)) * 31 + len(hist)
    h += sum((j + 1) * 1009 * p for j, p in enumerate(parts))
    return h % V


def onehot(rows):
    out = [[0.0] * V for _ in rows]
    for r, t in zip(out, rows):
        r[t] = 10.0
    return mx.array(out, dtype=mx.float32)


class GridFake(Backend):
    kind = "fake"
    max_rows = 16
    bytes_per_token = 8

    def __init__(self, pf_ms: float = 0.0, step_ms: float = 0.0, layers: int = 0, yield_cb=None):
        self.pf_ms, self.step_ms = pf_ms, step_ms
        self.calls: list = []
        self.layers = layers                    # > 0: a chunk is `layers` layer groups, the hook (if any) is called after each (like bound_layers)
        self.layer_hook = None
        self.forks = 0

    supports_yield = True

    def set_prefill_yield(self, cb):
        self.layer_hook = cb

    def make_cache(self):
        return {"h": [], "parts": []}

    def fork(self, cache):
        self.forks += 1
        return {"h": list(cache["h"]), "parts": list(cache["parts"])}

    def prefill_chunk(self, cache, ids, draft=None, skip_dec=False):
        self.calls.append(("pf", len(cache["h"]), len(ids)))
        if self.layers and self.layer_hook is not None:
            for _ in range(self.layers):
                if self.pf_ms:
                    time.sleep(self.pf_ms * len(ids) / 1000.0 / self.layers)
                self.layer_hook()
        elif self.pf_ms:
            time.sleep(self.pf_ms * len(ids) / 1000.0)
        cache["h"].extend(ids)
        cache["parts"].append(len(ids))
        return onehot([nxt(cache["h"], cache["parts"])])

    def batch(self, caches):
        return list(caches)

    def unbatch(self, bc, n):
        return list(bc)

    def step(self, bc, tokens):
        self.calls.append(("step", len(bc)))
        if self.step_ms:
            time.sleep(self.step_ms / 1000.0)
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt(c["h"], c["parts"]) for c in bc])

    def state(self, cache):
        return []

    def nbytes(self, cache):
        return sum(self.bytes_per_token * len(c["h"]) for c in cache) if isinstance(cache, list) else self.bytes_per_token * len(cache["h"])

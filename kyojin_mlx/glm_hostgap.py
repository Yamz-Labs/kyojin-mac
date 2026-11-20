# SPDX-License-Identifier: Apache-2.0
"""Early submit of the GLM-5.3 decode step (flag KYOJIN_GLM_HOSTGAP, default off): the GPU starts the step while the host is still building it.

Why (N3-HOSTGAP, measured on the full model, plain step, 1 session, lane flags): the engine already launches step k+1 from the lazy argmax of step k before it reads
step k (Engine.lookahead), but the host builds the WHOLE python graph of a step (5.4 ms for 45 layers, nothing committed) and calls `mx.async_eval` once at its end.
MLX's own throttle (`eval_impl`: it waits for one in-flight command buffer to finish when too many are active, ~10) keeps that call, and with it the host, in lockstep
with the GPU, so it returns ~2.3 ms before step k ends. The next tick then starts its 5.4 ms graph build with only 2.3 ms of GPU work left to hide it behind: the GPU
idles ~3.1 ms between two steps (wall 20.3 ms vs GPU busy 17.4 ms; G2h "host gap"). The 4-session step is the same (python 5.5 ms, gap ~4.5 ms).

Fix: submit the graph with `mx.async_eval` after the first FIRST layers and then every EVERY layers of a decode step, so the first command buffers of step k+1 are
committed ~0.5 ms into the tick, queued behind the tail of step k, and the GPU goes from step k to step k+1 without a gap. Same ops on the same inputs in the same
order: only the point where the graph is committed moves, so the tokens are bit-identical. Only decode steps (<= KYOJIN_GLM_HOSTGAP_ROWS rows per session; default 1):
the verify rows of an MTP round already submit per DSA layer (KYOJIN_GLM_FUSE_DSA_EVAL).

Installed on the decoder layers (one instance subclass per layer, like glm_compile) after every other layer hook of the load; switched at call time by ENABLED."""
from __future__ import annotations

import os
import types

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_GLM_HOSTGAP", "0") not in ("0", "")
FIRST = int(os.environ.get("KYOJIN_GLM_HOSTGAP_FIRST", "1"))            # layers built before the first submit (0 is read as 1)
EVERY = int(os.environ.get("KYOJIN_GLM_HOSTGAP_EVERY", "4"))            # layers between two later submits
ROWS = int(os.environ.get("KYOJIN_GLM_HOSTGAP_ROWS", "1"))              # most rows per session of a pass that submits early (1 = plain decode steps)
STATS = {"submits": 0, "passes": 0}


def _make_cls(cls):
    base_call = cls.__call__

    def __call__(self, x, *args, **kwargs):
        out = base_call(self, x, *args, **kwargs)
        if not ENABLED:
            return out
        hg = self._hg
        if hg.idx == 0:
            STATS["passes"] += 1
        if getattr(x, "ndim", 0) != 4 or x.shape[1] > ROWS:           # [sessions, rows, hc, hidden]: a prefill chunk / verify rows keep their own submits
            return out
        c = hg.idx + 1                                                # layers built so far
        first = max(1, FIRST)
        if c < hg.n and c >= first and (c - first) % max(1, EVERY) == 0:
            h = out[0] if isinstance(out, tuple) else out
            try:
                mx.async_eval(h)
                STATS["submits"] += 1
            except Exception:                                          # inside a compiled trace: nothing to submit
                pass
        return out

    return type("Hg" + cls.__name__, (cls,), {"__call__": __call__})


def install(lm) -> int:
    """Arm every decoder layer of the language model `lm` (idempotent); returns the number of layers armed."""
    layers = lm.model.layers
    n, k = len(layers), 0
    for i, layer in enumerate(layers):
        if "_hg" in layer.__dict__:
            continue
        layer._hg = types.SimpleNamespace(idx=i, n=n)                   # not a dict: nn.Module keeps dict values as items
        layer.__class__ = _make_cls(type(layer))
        k += 1
    return k

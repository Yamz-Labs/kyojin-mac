"""Did a lever run? One counter per lever site, off by default (KYOJIN_FIRED=1 or flags.set("fired", 1)).

A lever flag that changes nothing is indistinguishable from a lever that never ran. With `fired` on, each site that a lever can change bumps
HITS[name] where it takes the lever's path (and HITS["<name>_off"] style keys where the site was reached but a gate kept the old path), and
serve.server.engage() lists HITS under /v1/stats "engage" like every other module-level HITS dict. Off, `hit` is one global read and a return.

Sites inside an mx.compile region or a replay recording bump at trace / record time only (once per signature), not per token: a nonzero count there
proves the lever's branch was traced, the fuse_layer / verify_replay STATS call counters say how often the traced region then ran."""
from __future__ import annotations

import os

ON = os.environ.get("KYOJIN_FIRED", "0") not in ("0", "")
HITS: dict = {}


def hit(name: str, n: int = 1) -> None:
    if ON:
        HITS[name] = HITS.get(name, 0) + n


def reset() -> None:
    HITS.clear()

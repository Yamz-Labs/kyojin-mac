# SPDX-License-Identifier: Apache-2.0
"""Measurement probes of the serve engine (read-only: they never change a token, a weight or a routing decision).

spec_ptap  (KYOJIN_SPEC_PTAP=1, KYOJIN_SPEC_PTAP_OUT=<jsonl path>; read per round)
    One JSON line per sampled (T > 0) session and speculative round: the draft source (mtp / lookup), the sampling parameters, and for EVERY draft
    position j the target probability of the draft token under the exact distribution the engine samples from (serve.sampling filters, then the
    temperature: specsam.logp), the entropy of that distribution, whether the verify row's sample equalled the draft, and the number accepted.
    The values are one small graph over the verify logits that are already evaluated; it is evaluated and read once per round, after the
    round's own sync. Offline use: simulate "typical acceptance" thresholds.

router_tap (KYOJIN_ROUTER_TAP=1, runtime flag `router_tap`)
    Decoded rows (plain and verify) of every MoE router: the routing weights as the engine applies them (after normalisation and scaling), sorted by rank,
    as a share of the row's total. Per layer and rank r: sum and sum of squares of the share, and the number of rows whose share is < 0.02 / 0.05 /
    0.10 / 0.15. Accumulated on the device (one [6, k] array per layer), flushed to host counters every FLUSH_EVERY engine ticks, shown in /v1/stats.
    Layers are numbered by the order a router module is first seen (the order a forward visits them).
"""
from __future__ import annotations

import json
import os
import threading
import time

import mlx.core as mx
import numpy as np

THRESHOLDS = (0.02, 0.05, 0.10, 0.15)
FLUSH_EVERY = 8
ROUTER = os.environ.get("KYOJIN_ROUTER_TAP", "0") not in ("0", "")      # flag router_tap (flags.py rewrites this attribute on POST /admin/flags)


# ------------------------------------------------------------------------------------------------------------------------ spec_ptap
def ptap_on() -> bool:
    return os.environ.get("KYOJIN_SPEC_PTAP", "0") not in ("0", "") and bool(os.environ.get("KYOJIN_SPEC_PTAP_OUT"))


def _row_stats(row: mx.array, sampling, tok: int | None):
    """(p_target(tok), entropy) of one target logits row under the sampling distribution."""
    from . import specsam
    lp = specsam.logp(row, sampling)
    p = mx.exp(lp)
    ent = -mx.sum(mx.where(p > 0, p * lp, 0.0))
    return (mx.exp(lp[tok]) if tok is not None else mx.array(0.0)), ent


def ptap_collect(members: list, toks: list, res, lookups) -> dict:
    """{member index: (p_draft [n], entropy [n])} (python floats) for the sampled sessions that drafted this round, over EVERY draft position.
    One graph, one eval and one read for the whole round. Any failure returns {} (a probe never breaks serving)."""
    try:
        graphs: dict = {}
        for k, s in enumerate(members):
            if s.sampling.greedy or len(toks[k]) < 2:
                continue
            r0, dr = res.starts[k], toks[k][1:]
            ps, hs = zip(*[_row_stats(res.logits[r0 + j], s.sampling, int(dr[j])) for j in range(len(dr))])
            graphs[k] = (mx.stack(ps), mx.stack(hs))
        if not graphs:
            return {}
        flat = [a for pair in graphs.values() for a in pair]
        mx.eval(*flat)
        return {k: (p.tolist(), h.tolist()) for k, (p, h) in graphs.items()}
    except Exception:                                               # noqa: BLE001
        return {}


_wlock = threading.Lock()


def ptap_write(k: int, s, tk: list, entry, matches: list, accepted: int, lookup: bool, rs: bool) -> None:
    try:
        p, h = entry
        smp = s.sampling
        line = {"t": round(time.time(), 3), "src": "lookup" if lookup else "mtp", "rs": bool(rs), "temperature": smp.temperature, "top_p": smp.top_p,
                "top_k": smp.top_k, "min_p": smp.min_p, "n": len(p), "p_draft": p, "entropy": h, "match": [bool(m) for m in matches], "accepted": int(accepted)}
        with _wlock, open(os.environ["KYOJIN_SPEC_PTAP_OUT"], "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:                                               # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------------------------------ router_tap
class _Router:
    def __init__(self):
        self.ids: dict = {}                    # id(router module) -> layer number
        self.acc: dict = {}                    # layer -> lazy [6, k] device accumulator (sum, sumsq, 4 counts) since the last flush
        self.rows: dict = {}                   # layer -> lazy rows counted since the last flush (python int: shapes are static)
        self.tot: dict = {}                    # layer -> np.float64 [6, k] since process start
        self.n: dict = {}                      # layer -> rows since process start
        self.ticks = 0


R = _Router()


def router_tap(module, wts: mx.array) -> None:
    """Record the applied routing weights wts [rows, 1 | , k] of one router call (a decode call: one token per row). Reads only."""
    w = wts.reshape(-1, wts.shape[-1]).astype(mx.float32)
    w = mx.sort(w, axis=-1)[:, ::-1]                                # rank 1 = largest
    share = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    st = mx.stack([share.sum(0), (share * share).sum(0)] + [(share < t).astype(mx.float32).sum(0) for t in THRESHOLDS])
    layer = R.ids.setdefault(id(module), len(R.ids))
    R.acc[layer] = st if layer not in R.acc else R.acc[layer] + st
    R.rows[layer] = R.rows.get(layer, 0) + int(w.shape[0])


def router_flush(force: bool = False) -> None:
    """Engine thread, once per tick: every FLUSH_EVERY ticks evaluate the device accumulators and add them to the host counters."""
    if not R.acc:
        return
    R.ticks += 1
    if not force and R.ticks % FLUSH_EVERY:
        return
    try:
        layers = sorted(R.acc)
        mx.eval(*[R.acc[l] for l in layers])
        for l in layers:
            a = np.array(R.acc[l], dtype=np.float64)
            R.tot[l] = a if l not in R.tot else R.tot[l] + a
            R.n[l] = R.n.get(l, 0) + R.rows.get(l, 0)
    finally:
        R.acc.clear()
        R.rows.clear()


def router_snapshot() -> dict | None:
    """The /v1/stats block (None when nothing was recorded): per layer {rows, sum, sumsq, lt02, lt05, lt10, lt15} (lists over rank 1..k) and the same over all layers."""
    if not R.tot:
        return None
    names = ("sum", "sumsq") + tuple("lt%02d" % round(t * 100) for t in THRESHOLDS)

    def block(a, rows):
        return {"rows": int(rows), **{n: [round(float(x), 6) for x in a[i]] for i, n in enumerate(names)}}
    layers = sorted(R.tot)
    kmax = max(R.tot[l].shape[1] for l in layers)                       # routers of another top-k (e.g. a draft head) stay per layer only
    main = [l for l in layers if R.tot[l].shape[1] == kmax]
    allr = sum(R.tot[l] for l in main)
    return {"layers": {str(l): block(R.tot[l], R.n[l]) for l in layers}, "all": block(allr, sum(R.n[l] for l in main)), "all_k": int(kmax),
            "other_k_layers": [str(l) for l in layers if l not in main]}


def router_reset() -> None:
    R.acc.clear(); R.rows.clear(); R.tot.clear(); R.n.clear(); R.ids.clear(); R.ticks = 0

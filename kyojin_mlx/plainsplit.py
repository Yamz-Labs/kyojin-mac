# SPDX-License-Identifier: Apache-2.0
"""Probe: the host split of one plain decode step, and whether the next step is really encoded while the one in flight runs (flag `plain_split`,
default 0).

The engine times a plain round with `spec_rt_r1` / `spec_rdraft_r1`, one number for the build and one for the rest. That is not enough to know
where the 21.73 ms of a plain tick go against the 13.36 ms the GPU is busy (opgap): the tick is [build N+1][read N], and the GPU is running N
while the host builds N+1. This probe brackets the round with the same uptime clock the Metal command buffers stamp, wraps the two blocking calls
of the path (`mx.async_eval`, `mx.array.tolist`) and reads back the commit range of the step launched in the tick and of the step it consumed:

  plan      the round's own plan before the plain step (depth / lookup / epoch checks)
  build     the plain launch, wall (the graph the host builds; what a compiled / replayed step would remove)
  encode    the part of `build` spent inside `mx.async_eval` (the layer-bound submits, bound_layers DEC_EVERY) - a subset of build, not extra
  wait      `y.tolist()`: the call that waits for the step in flight (a subset of consume; the text itself is made by the HTTP thread)
  accept    consume - wait (accept loop, one queue put per token)
  post      the round's bookkeeping after the read (stats, ring, cost fit, kv note, the tick's own tail)
  round     plan + build + consume + post: the sum the table has to close

and, from the command buffers, per tick: the GPU busy time of the step this tick consumed and of the step it launched (`cur_*`), the sum of
the per-buffer spans (`cur_sum`; > busy means the buffers overlap), the gaps inside the step split by cause (`cur_idleh` = nothing left to run,
i.e. the host had not committed it yet; `cur_idleq` = a buffer was already queued, the driver's turn-around), the fraction of the build covered by
GPU work (`overlap`), the ms the GPU had nothing to run between two steps (`starve` = this step's first GPU start minus the previous step's GPU
end), the wait the read paid for the GPU (`wait_gpu`) and the dispatch count of the step (`cur_nd`). `kernelStartTime` / `kernelEndTime` are
NOT used: on this driver they do not bracket the kernels (scratch/cbstamp_roof.py: a 1.07 ms copy reports an 11.5 ms kernel window).

`cur_busy_close` is the same union read at the tick's own end, i.e. the number opgap's `gpu_span` folds in: the step is still in flight there, so
the buffers whose stamps have not landed are missing from it.

Rows go to KYOJIN_PLAIN_SPLIT_OUT (JSONL, capped) for medians; the sums go to /v1/stats as ps_<field>_r<rows> (+ ps_n_r<rows>), so a run can be
read without the file. Nothing is added to the graph or the GPU: six clock reads per tick, one per submit, and the cbcount counters the lane
already builds for gpu_span. `cur_*` / `starve` are one tick late by construction: a commit's GPU stamps land when the buffer completes."""
from __future__ import annotations

import ctypes
import json
import os
import time

import mlx.core as mx

from . import gpuspan as GS

OUT = os.environ.get("KYOJIN_PLAIN_SPLIT_OUT") or os.path.join(os.getcwd(), "scratch", "plain_split.jsonl")
MAXL = int(os.environ.get("KYOJIN_PLAIN_SPLIT_MAX", "20000"))
FIELDS = ("plan", "build", "encode", "wait", "accept", "post", "tick", "round",
          "prev_busy", "prev_n", "prev_resid", "overlap",
          "cur_start", "cur_end", "cur_busy", "cur_sum", "cur_n", "cur_nd",
          "cur_gaps", "cur_idleh", "cur_idleq",
          "starve", "wait_gpu", "first_commit", "bufs", "build_n", "wait_n", "busy_close", "n_close")

_seq = [0]
_round = [0.0, -1]                      # uptime of the round's start, the round's sequence number
_last = [0.0]                           # last round's own wall ms (stored by tick_end)
_state = {"prev": None, "last_end": 0.0}
_row = [None]                           # last plain tick's row, waiting for its own step's GPU stamps
_out = [None]
_n = [0]
_d0, _d1, _k0 = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()

ZERO = dict(beg=0.0, fin=0.0, busy=0.0, n=0, nd=0, gsum=0.0, gaps=0, idleh=0.0, idleq=0.0)


def on() -> bool:
    """Read per call (flag `plain_split`), like the other serve probes."""
    return os.environ.get("KYOJIN_PLAIN_SPLIT", "0") not in ("0", "")


def _clk() -> float:
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) * 1e-9


def _lib():
    if GS._lib is None:
        GS.install()
        if GS._lib is None:
            return None
    GS._lib.cb_rec(1)               # count the dispatches of a buffer (the names stay unused: no MLX_MAX_OPS_PER_BUFFER change)
    return GS._lib


def _union(iv: list) -> tuple:
    """(first start, last end, union length) of a list of [start, end) intervals."""
    if not iv:
        return (0.0, 0.0, 0.0)
    iv.sort()
    a, b = iv[0]
    beg, fin, busy = a, iv[-1][1], 0.0
    for s, e in iv[1:]:
        if s > b:
            busy += b - a
            a, b = s, e
        else:
            b = max(b, e)
    busy += b - a
    return (beg, fin, busy)


def _scan(s0: int, s1: int) -> dict:
    """The commit range [s0, s1) read from the driver's stamps: GPU spans (scheduling to end incl. fence waits), kernel spans (where the kernels
    ran), both unioned, plus the sum of the GPU spans and the mean wait before a buffer's kernels start. Only the buffers whose stamps have landed
    count (n); 0 where none has."""
    C = _lib()
    if C is None or s1 <= s0:
        return dict(ZERO)
    g, gsum, nd, comm = [], 0.0, 0, []
    for s in range(s0, s1):
        C.cb_times(s, ctypes.byref(_d0), ctypes.byref(_d1))
        C.cb_commit_time(s, ctypes.byref(_k0))
        comm.append(_k0.value)
        nd += C.cb_ndisp(s)
        a, b = _d0.value, _d1.value
        if not (a > 0 and b > a and b - a < 2.0):
            continue
        g.append((a, b))
        gsum += b - a
    gbeg, gfin, gbusy = _union(g)
    # the gaps inside the step, by cause (the gpu_span rule): nothing left to run at the gap's start (the host had not committed
    # it yet) is host idle; a buffer already queued is the driver's turn-around.
    iv = sorted(g)
    ih = iq = 0.0
    ngaps = 0
    for (a0, b0), (a1, b1) in zip(iv, iv[1:]):
        if a1 > b0:
            ngaps += 1
            gap = a1 - b0
            pend = sum(1 for c in comm if c <= b0) - sum(1 for _, e in iv if e <= b0)
            if pend > 0:
                iq += gap
            else:
                ih += gap
    return dict(beg=gbeg, fin=gfin, busy=gbusy, n=len(g), nd=nd, gsum=gsum, gaps=ngaps, idleh=ih, idleq=iq)


def tick_begin() -> None:
    """Engine, once per round, before the plan: the window `plan` and `post` are measured against."""
    if not on():
        _round[0] = 0.0
        return
    _seq[0] += 1
    _round[0], _round[1] = _clk(), _seq[0]


def tick_end() -> None:
    """Engine, after the round: the round's own wall (the last column that is not in the plain tick)."""
    if _round[0]:
        _last[0] = (_clk() - _round[0]) * 1e3
    _round[0] = 0.0


class Tick:
    """One plain tick: the marks, and the two blocking calls of the path (wall + thread CPU + count)."""

    def __init__(self):
        self.t0 = _clk()
        self.seq = _seq[0]
        self.plan = (self.t0 - _round[0]) * 1e3 if (_round[0] and _round[1] == self.seq) else -1.0
        self.marks = {"start": self.t0}
        self.ae = [0.0, 0.0, 0]                              # async_eval: wall, thread CPU, calls (inside the build)
        self.tl = [0.0, 0.0, 0]                              # tolist: wall, thread CPU, calls (inside consume)
        self._saved = (mx.async_eval, mx.array.tolist)
        aev, tl = self._saved
        ae, tlacc = self.ae, self.tl

        def ae_(*a, **k):
            t, c = time.perf_counter(), time.thread_time()
            try:
                return aev(*a, **k)
            finally:
                ae[0] += time.perf_counter() - t
                ae[1] += time.thread_time() - c
                ae[2] += 1

        def tl_(self_):
            t, c = time.perf_counter(), time.thread_time()
            try:
                return tl(self_)
            finally:
                tlacc[0] += time.perf_counter() - t
                tlacc[1] += time.thread_time() - c
                tlacc[2] += 1

        mx.async_eval, mx.array.tolist = ae_, tl_
        C = _lib()
        self.s0 = C.cb_commits() if C is not None else 0

    def mark(self, name: str) -> None:
        self.marks[name] = _clk()

    def close(self, stats: dict, rows: int) -> None:
        mx.async_eval, mx.array.tolist = self._saved
        C = _lib()
        if C is None:
            return
        s1 = C.cb_commits()
        now = _clk()
        m = self.marks
        tb = m.get("launch", now)
        tc = m.get("consume", tb)
        te = m.get("end", now)
        build = (tb - self.t0) * 1e3
        consume = (tc - tb) * 1e3
        wait = self.tl[0] * 1e3
        tick = (te - self.t0) * 1e3
        r = {"rows": rows, "seq": self.seq, "plan": self.plan, "build": build, "encode": self.ae[0] * 1e3,
             "build_cpu": self.ae[1] * 1e3, "build_n": self.ae[2], "wait": wait, "wait_cpu": self.tl[1] * 1e3,
             "wait_n": self.tl[2], "accept": consume - wait, "consume": consume, "post": tick - build - consume,
             "tick": tick, "bufs": s1 - self.s0, "first_commit": 0.0, "wait_gpu": wait,
             "prev_busy": 0.0, "prev_n": 0, "prev_resid": 0.0, "overlap": 0.0,
             "cur_start": 0.0, "cur_end": 0.0, "cur_busy": 0.0, "cur_sum": 0.0,
             "cur_n": 0, "cur_nd": 0, "cur_gaps": 0, "cur_idleh": 0.0, "cur_idleq": 0.0,
             "starve": 0.0, "busy_close": 0.0, "n_close": 0}
        ct = ctypes.c_double()
        if s1 > self.s0:
            C.cb_commit_time(self.s0, ct)
            r["first_commit"] = (ct.value - self.t0) * 1e3
            sc = _scan(self.s0, s1)                 # what opgap's gpu_span sees at the tick's own end: a step still in flight
            r["busy_close"], r["n_close"] = sc["busy"] * 1e3, sc["n"]
        p = _state["prev"]                          # the step this tick consumed (launched last tick): stamps complete now
        sv = dict(ZERO)
        if p is not None and p["s1"] > p["s0"]:
            sv = _scan(p["s0"], p["s1"])
            r["prev_n"], r["prev_busy"] = sv["n"], sv["busy"] * 1e3
            r["prev_resid"] = (sv["fin"] - self.t0) * 1e3 if sv["fin"] else 0.0
            if sv["fin"] and sv["beg"]:             # the part of this tick's build the GPU spent running the step in flight
                r["overlap"] = max(0.0, (min(tb, sv["fin"]) - max(self.t0, sv["beg"])) * 1e3)
        q = _row[0]                                 # the previous tick's own row: its step is `sv`, complete now
        if q is not None:
            if q["bufs"] <= 0:                      # the tick that launched nothing (end of the reply): no step of its own
                _row[0] = None
            else:
                q["cur_start"], q["cur_busy"] = (sv["beg"] - q["_t0"]) * 1e3, sv["busy"] * 1e3
                q["cur_end"], q["cur_sum"] = (sv["fin"] - q["_t0"]) * 1e3, sv["gsum"] * 1e3
                q["cur_n"], q["cur_nd"] = sv["n"], sv["nd"]
                q["cur_gaps"], q["cur_idleh"], q["cur_idleq"] = sv["gaps"], sv["idleh"] * 1e3, sv["idleq"] * 1e3
                q["starve"] = max(0.0, (sv["beg"] - _state["last_end"]) * 1e3) if (sv["beg"] and _state["last_end"]) else 0.0
                q["round"] = _last[0] if q["seq"] == _seq[0] - 1 else -1.0
                q.pop("_t0", None)
                _emit(stats, q, q.pop("_rows"))
                _row[0] = None
        if sv["fin"]:
            _state["last_end"] = sv["fin"]
        _state["prev"] = {"s0": self.s0, "s1": s1}
        r["_s0"], r["_s1"], r["_t0"], r["_rows"] = self.s0, s1, self.t0, rows
        _row[0] = r


def _emit(stats: dict, r: dict, rows: int) -> None:
    """Fold one finished row into the stats (sums, /v1/stats ps_<field>_r<rows>) and append it to the JSONL (bounded)."""
    n = f"ps_n_r{rows}"
    stats[n] = stats.get(n, 0) + 1
    for k in FIELDS:
        kk = f"ps_{k}_r{rows}"
        stats[kk] = stats.get(kk, 0.0) + (r.get(k) or 0.0)
    if OUT and _n[0] < MAXL:
        if _out[0] is None:
            _out[0] = open(OUT, "a", buffering=1)
        try:
            _out[0].write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                      if not k.startswith("_")}) + "\n")
            _n[0] += 1
        except Exception:
            pass


def begin(stats: dict) -> Tick:
    return Tick()

# SPDX-License-Identifier: Apache-2.0
"""Wall-clock sampling of the engine thread's own Python stack: which file:line the served round's time goes to (probe, no GPU work).

`sync(stats)` runs once per engine loop iteration from the engine thread. It follows KYOJIN_HOST_SAMPLE (period in microseconds, 0 = off) and
KYOJIN_HOST_SAMPLE_OUT (dump path): while armed, a helper thread samples `sys._current_frames()[engine thread]` every period and adds the wall
time since the previous sample to that frame's innermost site (filename, lineno, function). The totals are therefore a wall attribution of the
engine thread: host work and every blocking call it sits in (a `mx.eval` wait shows up as the eval's call site, because the engine thread is
inside the C call when the sample lands).

The sampler's own lateness separates the two kinds of off-CPU time. MLX holds the GIL while it waits for the GPU (scratch/gil_test.py), so a
sample that arrives more than 1.6 periods late is time the engine thread held the GIL inside a C call; if the engine thread had released the
GIL and waited for it, the sampler (a Python thread too) would have been served. That part is counted `gil` per site, and the section's
non-blocking off-CPU is the rest (a real re-acquire). Nothing here changes a token, a plan or a graph: `sync` is one os.environ read when the
lever is off, and the sampler is a daemon thread that ends with the dump.

Dump format (one line per site, seconds descending):
    total <s> samples <n> sites <n> gil_held <s> late <s> sampler_cpu <s>
    <seconds> <count> <gil-seconds> <file>:<line> <function>
"""
from __future__ import annotations

import os
import sys
import threading
import time

LATE = 1.6                       # a sample this many periods late = the engine thread held the GIL (sys.setswitchinterval 5 ms)
FLUSH_S = 2.0                    # while armed, the dump is rewritten every FLUSH_S seconds (a killed run still leaves its attribution)


class Sampler:
    """Samples `tid`'s Python stack every `period` seconds until stopped; `dump(path)` writes the attribution."""

    def __init__(self, tid: int, period: float, path: str = ""):
        self.tid, self.period, self.path = tid, period, path
        self._every = max(200, int(round(FLUSH_S / period)))     # samples between two live snapshots of the dump
        self.sites: dict = {}                       # (file, line, func) -> [seconds, count, gil_seconds]
        self.total = self.gil_held = self.late = self.cpu = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="kyojin-pysample", daemon=True)
        self._th.start()

    def _run(self) -> None:
        period, late_at = self.period, self.period * LATE
        frames = sys._current_frames
        last = time.perf_counter()
        while not self._stop.is_set():
            c0 = time.thread_time()
            self._stop.wait(period)
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt <= 0.0:
                continue
            f = frames().get(self.tid)
            if f is not None:                                       # frames() hands the innermost frame of the thread: the call site it is in
                code = f.f_code
                key = (code.co_filename, f.f_lineno, code.co_name)
                slot = self.sites.get(key)
                if slot is None:
                    slot = self.sites[key] = [0.0, 0, 0.0]
                slot[0] += dt
                slot[1] += 1
                held = dt > late_at
                if held:                            # the GIL was not handed over: this time is not a re-acquire
                    slot[2] += dt
                    self.gil_held += dt
                    self.late += dt - period
            self.total += dt
            self.samples += 1
            self.cpu += time.thread_time() - c0
            if self.path and self.samples % self._every == 0:
                try:
                    self.write(self.path)                            # a live snapshot: the file is there whatever ends the run
                except OSError:
                    pass

    def dump(self, path: str) -> None:
        self._stop.set()
        self._th.join(timeout=2.0)
        self.write(path)

    def write(self, path: str) -> None:
        rows = sorted(list(self.sites.items()), key=lambda kv: -kv[1][0])
        with open(path, "w") as out:
            out.write("total %.6f samples %d sites %d gil_held %.6f late %.6f sampler_cpu %.6f\n"
                      % (self.total, self.samples, len(rows), self.gil_held, self.late, self.cpu))
            for (fn, line, name), (s, n, g) in rows:
                out.write("%.6f %d %.6f %s:%d %s\n" % (s, n, g, fn, line, name))


_lock = threading.Lock()
_sampler: Sampler | None = None
_armed: tuple | None = None


def sync(stats=None) -> None:
    """Engine thread, once per loop iteration: follow the sampling lever (start / stop and dump). Two os.environ reads when off."""
    global _sampler, _armed
    want = os.environ.get("KYOJIN_HOST_SAMPLE", "0")
    out = (os.environ.get("KYOJIN_HOST_SAMPLE_OUT") or "").strip() or os.path.join(os.getcwd(), "pysample.txt")
    if want not in ("0", ""):
        if _sampler is None:
            with _lock:
                if _sampler is None:
                    _sampler = Sampler(threading.get_ident(), max(200, int(want)) / 1e6, out)
                    if stats is not None:
                        stats["n_hs_sample_arm"] += 1
        elif _armed is not None and _armed != (want, out):      # a new period or a new path: close the old run first
            off(stats)
    elif _sampler is not None:
        off(stats)
    if _sampler is not None:
        _armed = (want, out)


def off(stats=None) -> None:
    """Stop the sampler and write its attribution to KYOJIN_HOST_SAMPLE_OUT (the file the lever was armed with)."""
    global _sampler, _armed
    with _lock:
        s, _armed, _sampler = _sampler, None, None
    if s is None:
        return
    path = (os.environ.get("KYOJIN_HOST_SAMPLE_OUT") or "").strip() or os.path.join(os.getcwd(), "pysample.txt")
    try:
        s.dump(path)
    except OSError:
        return
    if stats is not None:
        st = stats
        st["t_hs_sample_total"] += s.total
        st["t_hs_sample_gil"] += s.gil_held
        st["t_hs_sample_cpu"] += s.cpu
        st["n_hs_sample"] += s.samples
        st["n_hs_sample_dump"] += 1

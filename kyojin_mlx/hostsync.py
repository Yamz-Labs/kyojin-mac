"""Host-side cost counters of a served round: forced syncs and graph nodes built (no GPU needed; the counts do not depend on the device).

`Probe()` is a context manager. While it is active it counts, per kind, the calls that make the host wait for the GPU or hand work to it:
  eval        mx.eval (blocking)                     async_eval  mx.async_eval (no wait)
  tolist      mx.array.tolist (blocking if the array is not computed yet)   read  `read()` of an array an eval already computed (no wait)      item  mx.array.item (same)
  asarray     np.asarray of an mx array through `asnumpy` (call sites that read the GPU use it)
`graph_nodes(*outs)` counts the non-view operations of the lazy graph under `outs` (what the host built and the stream thread must encode),
`graph_census(*outs)` the same by primitive. Both walk the graph with mx.export_to_dot (tiny models only: the dot text is large)."""
import collections
import contextlib
import os
import re
import tempfile
import time

import mlx.core as mx
import numpy as np
from . import fired as FD

VIEWS = {"Reshape", "Transpose", "AsType", "Broadcast", "Squeeze", "ExpandDims", "Slice", "StopGradient", "Depends", "ArrayPrint"}
_active: list = []
_reading = [0]

TIMED = os.environ.get("KYOJIN_HOST_TIMED", "0") not in ("0", "")
KINDS = ("eval", "async_eval", "tolist", "item")


def enabled() -> bool:
    """KYOJIN_VERIFY_ASYNC=1 (read per call): the served round's exact host-side levers (packed readback, async settle / flush, one commit index array)."""
    return os.environ.get("KYOJIN_VERIFY_ASYNC", "0") == "1"


class Timed:
    """Wall and thread-CPU time inside each blocking call kind while the context is active (profiling only: ~2 clock reads per call).

    On exit adds to `stats`: t_hs_<key>_<kind>_w / _c (seconds), n_hs_<key>_<kind> calls, max_hs_<key>_<kind>_w (the longest single call), and
    the section's own t_hs_<key>_w / _c / n_hs_<key>. wall ~ cpu = host work of the call; wall >> cpu = the call waited with the GIL held
    (MLX does not release it, scratch/gil_test.py), i.e. a GPU wait the section pays. The section's cpu is read by the caller (engine's
    spec_rvcpu_r<rows>), so the residual off-CPU of the section is wall - cpu - sum(kind: wall - cpu)."""

    def __init__(self, stats, key: str):
        self.stats, self.key, self.acc = stats, key, {k: [0.0, 0.0, 0, 0.0] for k in KINDS}
        self.wall = self.cpu = 0.0
        self.n = 0

    def __enter__(self):
        self._saved = (mx.eval, mx.async_eval, mx.array.tolist, mx.array.item)
        ev, aev, tl, it = self._saved
        acc = self.acc

        def wrap(name, fn):
            slot = acc[name]
            def call(*a, **k):
                t, c = time.perf_counter(), time.thread_time()
                try:
                    return fn(*a, **k)
                finally:
                    w = time.perf_counter() - t
                    slot[0] += w
                    slot[1] += time.thread_time() - c
                    slot[2] += 1
                    slot[3] = max(slot[3], w)
            return call

        mx.eval, mx.async_eval = wrap("eval", ev), wrap("async_eval", aev)
        mx.array.tolist = wrap("tolist", tl)
        mx.array.item = wrap("item", it)
        self._t, self._c = time.perf_counter(), time.thread_time()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.perf_counter() - self._t, time.thread_time() - self._c
        mx.eval, mx.async_eval, mx.array.tolist, mx.array.item = self._saved
        st, key = self.stats, self.key
        st["t_hs_" + key + "_w"] += self.wall
        st["t_hs_" + key + "_c"] += self.cpu
        st["n_hs_" + key] += 1
        for k, (w, c, n, mxw) in self.acc.items():
            if n:
                st["t_hs_%s_%s_w" % (key, k)] += w
                st["t_hs_%s_%s_c" % (key, k)] += c
                st["n_hs_%s_%s" % (key, k)] += n
                st["max_hs_%s_%s_w" % (key, k)] = max(st.get("max_hs_%s_%s_w" % (key, k), 0.0), mxw)
        return False

    def sync_wall(self) -> float:
        """Wall time spent inside blocking calls (the part of the section a wait can hide in)."""
        return sum(w for w, _c, _n, _m in self.acc.values())


def timed(stats, key: str):
    """`with hostsync.timed(stats, key)` when KYOJIN_HOST_TIMED is on, a null context otherwise (one test per section when off)."""
    return Timed(stats, key) if TIMED else contextlib.nullcontext()


class Probe:
    def __init__(self):
        self.n = collections.Counter()

    def __enter__(self):
        self._saved = (mx.eval, mx.async_eval, mx.array.tolist, mx.array.item)
        ev, aev, tl, it = self._saved
        n = self.n

        def eval_(*a, **k):
            n["eval"] += 1
            return ev(*a, **k)

        def async_eval_(*a, **k):
            n["async_eval"] += 1
            return aev(*a, **k)

        def tolist_(self_):
            n["read" if _reading[0] else "tolist"] += 1
            return tl(self_)

        def item_(self_):
            n["item"] += 1
            return it(self_)
        mx.eval, mx.async_eval, mx.array.tolist, mx.array.item = eval_, async_eval_, tolist_, item_
        _active.append(self)
        return self

    def __exit__(self, *exc):
        mx.eval, mx.async_eval, mx.array.tolist, mx.array.item = self._saved
        _active.remove(self)

    def blocking(self) -> int:
        """Calls that can make the host wait: every eval / tolist / item / asarray (an async_eval never waits)."""
        return self.n["eval"] + self.n["tolist"] + self.n["item"] + self.n["asarray"]

    def reset(self) -> None:
        self.n.clear()


def read(a):
    """a.tolist() of an array a previous mx.eval already computed: a copy, never a wait. Counted as `read`, not as a blocking `tolist`."""
    _reading[0] += 1
    FD.hit("verify_async")
    try:
        return a.tolist()
    finally:
        _reading[0] -= 1


def asnumpy(a):
    """np.asarray(a) for a GPU array, counted by the active probes."""
    for p in _active:
        p.n["asarray"] += 1
    return np.asarray(a)


def graph_census(*outs) -> collections.Counter:
    path = os.path.join(tempfile.gettempdir(), f"hostsync_{os.getpid()}.dot")
    try:
        mx.export_to_dot(path, *outs)
        with open(path) as f:
            labels = re.findall(r'label ?= ?"([^"]*)"', f.read())
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return collections.Counter(l for l in labels if l not in VIEWS)


def graph_nodes(*outs) -> int:
    return sum(graph_census(*outs).values())

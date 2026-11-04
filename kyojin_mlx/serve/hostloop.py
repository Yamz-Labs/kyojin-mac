# SPDX-License-Identifier: Apache-2.0
"""Host side of the serve loop: what runs next to the decode thread (no mlx import, so it is testable without Metal).

Five levers, each behind its own flag (kyojin_mlx.flags, default off, read per call through the module attributes below):

  KYOJIN_SERVE_COALESCE  the HTTP thread drains every token the engine queued meanwhile, writes the SSE frames of the whole
                         batch with one `write`, and still emits one delta per token (the frame count a client sees is the same
                         as without the lever; `drain_events`). One wake of the HTTP thread per round instead of one per token.
  KYOJIN_SERVE_NODELAY   TCP_NODELAY on the accepted sockets (a small SSE frame is not held back by Nagle until the client's delayed ACK).
  KYOJIN_SERVE_GC        gc.freeze() of the loaded model, a larger young generation, a full collection only when the engine idles.
  KYOJIN_SERVE_SWITCH    sys.setswitchinterval in microseconds (0 = the interpreter's own value): how long a thread that wants the GIL waits.
  KYOJIN_SERVE_PROF      attribution: host wall time (t_hp_*) and thread CPU time (t_hpc_*) per section of the tick, GC pause time, GIL
                         scheduling lag, into Engine.stats (so GET /v1/stats shows them under "timing"). Wall ~ CPU = host work; wall >> CPU = a
                         hidden wait (GPU sync, GIL, lock). It perturbs what it measures: the lag sleeper wakes every LAG_MS and takes the GIL.

Nothing here touches the model, the logits or the order of tokens."""
from __future__ import annotations

import gc
import os
import sys
import threading
import time

from .. import fired as FD

COALESCE = os.environ.get("KYOJIN_SERVE_COALESCE", "0") not in ("0", "")
NODELAY = os.environ.get("KYOJIN_SERVE_NODELAY", "0") not in ("0", "")
GC = os.environ.get("KYOJIN_SERVE_GC", "0") not in ("0", "")
SWITCH_US = int(os.environ.get("KYOJIN_SERVE_SWITCH", "0") or 0)
PROF = os.environ.get("KYOJIN_SERVE_PROF", "0") not in ("0", "")

GC_GEN0 = 20000                      # young-generation threshold while KYOJIN_SERVE_GC is on (the interpreter's own is 700)
GC_GEN1 = 20
GC_GEN2 = 200
LAG_MS = 10.0                        # period of the GIL-lag sleeper while KYOJIN_SERVE_PROF is on (0 = no sleeper)

# ------------------------------------------------------------------------------------------------ incremental detokenization


class Detok:
    """Incremental detokenization on the ids so far (a multi-byte character cut by a token boundary waits for its tail)."""

    def __init__(self, app):
        self.app, self.ids, self.prefix, self.read = app, [], 0, 0

    def push(self, tok: int) -> str:
        self.ids.append(tok)
        old = self.app.decode(self.ids[self.prefix:self.read])
        new = self.app.decode(self.ids[self.prefix:])
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        old = self.app.decode(self.ids[self.prefix:self.read])
        new = self.app.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return new[len(old):]

    def push_many(self, toks) -> str:
        """The text of several tokens with two decodes instead of two per token. Taken only when the window grows cleanly; a window that ends
        inside a character, or whose new text does not extend the old, goes token by token from the state before the batch (the same text
        `push` gives, piece by piece). While `VERIFY_LEFT` > 0 the first batches are also run token by token on a copy: a tokenizer whose
        decode depends on the window start (clean-up of spaces, say) is found out and batching is switched off for the process."""
        global VERIFY_LEFT, BATCH_BAD
        n = len(toks)
        if n == 0:
            return ""
        if n == 1 or BATCH_BAD:
            return "".join([self.push(t) for t in toks])
        state = (len(self.ids), self.prefix, self.read)
        shadow = None
        if VERIFY_LEFT > 0:
            VERIFY_LEFT -= 1
            shadow = Detok(self.app)
            shadow.ids, shadow.prefix, shadow.read = list(self.ids), self.prefix, self.read
        self.ids.extend(toks)
        old = self.app.decode(self.ids[self.prefix:self.read])
        new = self.app.decode(self.ids[self.prefix:])
        if new.startswith(old) and len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            out = new[len(old):]
        else:
            del self.ids[state[0]:]
            out = "".join([self.push(t) for t in toks])
            return out
        if shadow is not None:
            ref = "".join([shadow.push(t) for t in toks])
            if ref != out:                                       # not the per-token text: serve per token from here on, and give this batch's reference
                BATCH_BAD = True
                self.ids, self.prefix, self.read = shadow.ids, shadow.prefix, shadow.read
                return ref
        return out


VERIFY_LEFT = 64                    # batches checked against the per-token text before the batched path is trusted
BATCH_BAD = False


class Reply:
    """Token ids in, parser events out. Stop strings cut the text before the parser sees it."""

    def __init__(self, app, parser, stops: list):
        self.detok, self.parser = Detok(app), parser
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buf = ""
        self.stopped = False
        self.n_tokens = 0

    def _release(self, final: bool) -> list:
        if self.stops:
            cut = [(self.buf.find(s), s) for s in self.stops if self.buf.find(s) >= 0]
            if cut:
                pos = min(cut)[0]
                text, self.buf, self.stopped = self.buf[:pos], "", True
                return self.parser.feed(text) if text else []
            keep = 0 if final else self.hold
        else:
            keep = 0
        cut = max(0, len(self.buf) - keep)                       # (a negative slice index released the held tail of a buffer shorter than `hold`: a stop string cut by the release was never found)
        text, self.buf = self.buf[:cut], self.buf[cut:]
        return self.parser.feed(text) if text else []

    def push(self, tok: int) -> list:
        if self.stopped:
            return []
        self.n_tokens += 1
        self.buf += self.detok.push(tok)
        return self._release(False)

    def push_many(self, toks) -> list:
        """The parser events of several tokens at once. With stop strings the tokens go one by one (`n_tokens` counts up to the token that
        completed the stop string, and the usage line reports it)."""
        if self.stopped:
            return []
        if self.stops or len(toks) == 1:
            ev = []
            for t in toks:
                ev += self.push(t)
            return ev
        self.n_tokens += len(toks)
        self.buf += self.detok.push_many(toks)
        return self._release(False)

    def finish(self) -> list:
        ev = []
        if not self.stopped:
            self.buf += self.detok.flush()
            ev += self._release(True)
        return ev + self.parser.finish()


def drain(q, first, nowait_exc) -> tuple:
    """(tokens, tail): `first` (a ("tok", id) event) and every ("tok", id) already queued behind it; `tail` is the first other event met
    (a "done"), or None. `nowait_exc` = queue.Empty."""
    toks = [first[1]]
    get = q.get_nowait
    while True:
        try:
            ev = get()
        except nowait_exc:
            return toks, None
        if ev[0] != "tok":
            return toks, ev
        toks.append(ev[1])


def drain_events(reply, toks) -> list:
    """The parser events of a drained batch, fed one token at a time. The coalesce lever still writes the frames of the whole
    batch with ONE `write`, but a client sees one `delta` per token, exactly as without the lever. Feeding the batch to
    `Reply.push_many` merges several tokens into one delta event: the concatenated text is the same, the token count a client
    derives from the frames is not (the 30/10 lane A/B read 514 -> 299 frames on identical text, and any tok/s computed from
    the frame count looked 2x slower). The batched detokenizer stays available (`push_many`), the wire keeps its granularity."""
    ev = []
    for t in toks:
        ev += reply.push(t)
    return ev


def new_queue():
    """The engine -> HTTP queue of one request: a SimpleQueue (C, no lock) with the coalescing lever, the classic Queue otherwise."""
    import queue
    return queue.SimpleQueue() if COALESCE else queue.Queue()


# ------------------------------------------------------------------------------------------------ GC, GIL interval

_gc_on = False
_gc_saved = None
_gc_idle_done = False
_sw_state = 0
_sw_orig = None


def gc_sync() -> None:
    """Engine thread, once per loop iteration: follow KYOJIN_SERVE_GC (read as a module attribute, so POST /admin/flags switches it)."""
    global _gc_on, _gc_saved, _gc_idle_done
    _gc_idle_done = False
    if GC == _gc_on:
        return
    if GC:
        _gc_saved = gc.get_threshold()
        gc.collect()
        gc.freeze()                                              # the loaded model and everything built so far leave the collector's lists
        gc.set_threshold(GC_GEN0, GC_GEN1, GC_GEN2)
        FD.hit("serve_gc")
    else:
        gc.unfreeze()
        gc.set_threshold(*(_gc_saved or (700, 10, 10)))
    _gc_on = GC


def idle() -> None:
    """Engine thread, before it waits for work: with the lever on, the cycles that built up during the last request are collected now (once per idle)."""
    global _gc_idle_done, _last
    _last = 0.0                                                  # the wait below is not a loop section: the next mark only starts a chain
    if _gc_on and not _gc_idle_done:
        _gc_idle_done = True
        gc.collect()
        FD.hit("serve_gc_idle")


def switch_sync() -> None:
    """Apply KYOJIN_SERVE_SWITCH (microseconds; 0 puts back the interval the process started with)."""
    global _sw_state, _sw_orig
    if SWITCH_US == _sw_state:
        return
    if _sw_orig is None:
        _sw_orig = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_US * 1e-6 if SWITCH_US > 0 else _sw_orig)
    _sw_state = SWITCH_US
    FD.hit("serve_switch" if SWITCH_US > 0 else "serve_switch_off")


# ------------------------------------------------------------------------------------------------ attribution

_stats = None
_last = 0.0
_last_cpu = 0.0
_gc_t0 = 0.0
_lag_thread = None
_lag_stop = threading.Event()


def _gc_cb(phase, info):
    global _gc_t0
    st = _stats
    if st is None:
        return
    if phase == "start":
        _gc_t0 = time.perf_counter()
    else:
        st["t_hp_gc"] += time.perf_counter() - _gc_t0
        st["n_hp_gc"] += 1
        st[f"n_hp_gc{info.get('generation', 0)}"] += 1


def _lag_loop(period: float):
    """Sleeps `period` and records how late it woke: the time a runnable thread waits for the GIL while the engine thread builds a graph."""
    while not _lag_stop.is_set():
        t = time.perf_counter()
        time.sleep(period)
        lag = time.perf_counter() - t - period
        st = _stats
        if st is not None:
            st["t_hp_giltick"] += max(0.0, lag)
            st["n_hp_giltick"] += 1
            if lag > 0.002:
                st["n_hp_gil2ms"] += 1


_prof_on = False


def bind(stats) -> None:
    """Engine thread, once per loop iteration: follow KYOJIN_SERVE_PROF, hand over the stats counter the sections add to."""
    global _stats, _prof_on, _lag_thread
    if PROF == _prof_on:
        return
    _prof_on = PROF
    if PROF:
        _stats = stats
        gc.callbacks.append(_gc_cb)
        _lag_stop.clear()
        if LAG_MS > 0:
            _lag_thread = threading.Thread(target=_lag_loop, args=(LAG_MS / 1e3,), name="kyojin-giltick", daemon=True)
            _lag_thread.start()
    else:
        _lag_stop.set()
        if _gc_cb in gc.callbacks:
            gc.callbacks.remove(_gc_cb)
        _stats = None


def reset() -> None:
    """Start a section chain (the first `mark` after it is measured from here)."""
    global _last, _last_cpu
    if PROF:
        _last, _last_cpu = time.perf_counter(), time.thread_time()


def mark(name: str) -> None:
    """Add the wall time since the previous mark / reset to section `name` (stats t_hp_<name>, n_hp_<name>). A single boolean test when the lever is off."""
    global _last, _last_cpu
    if not PROF:
        return
    now, cpu = time.perf_counter(), time.thread_time()
    st = _stats
    if _last == 0.0:
        _last, _last_cpu = now, cpu
        return
    if st is not None:
        st["t_hp_" + name] += now - _last
        st["t_hpc_" + name] += cpu - _last_cpu
        st["n_hp_" + name] += 1
    _last, _last_cpu = now, cpu

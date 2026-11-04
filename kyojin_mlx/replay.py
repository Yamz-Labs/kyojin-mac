# SPDX-License-Identifier: Apache-2.0
"""Live replay of static regions: a function of mx.arrays whose Metal command stream does not depend on the VALUES of its inputs (fixed shapes, array inputs only:
the compiled draft head, the draft-row append, any fixed-shape block of a step) is recorded once from an eager run and replayed as a Metal indirect command buffer
(ICB) from a one-launch trigger op. CPU per call: the graph build and the per-kernel encode (2.6 ms for the 3-stage draft head on the Air) become one op and a few
rebinds (tens of microseconds). The GPU work is the same kernels with the same bindings, so the results are bit-identical to the eager call.

    f = replay.region(fn, "name")      # same signature as fn; KYOJIN_REPLAY=0 (default) returns fn itself
    out = f(*arrays)                   # call 1: eager (warm), call 2: eager + record, then replay; first KYOJIN_REPLAY_CHECK (2) replays also run eager and compare bitwise

Automatic fallback to the normal path: a different signature (shapes / dtypes / non-array arguments) is a different entry (recorded on its own, at most MAX_ENTRIES per
region), a build that cannot be replayed (a pipeline created before the hooks, a result not found in the stream, ICB limits) or a shadow check that differs marks the entry
dead and every call goes through `fn`. Nothing is kept between calls (state is just the arrays you pass and get back), so there is no state to desynchronise.

How it works (replay/rp.mm, "static regions"): the recording stores every dispatch of one eager call (pipeline, grid, constants, buffer bindings). Buffers inside the input
arrays are rebound at every call to the buffers MLX gives the trigger op; buffers born during the call become private scratch of the region; everything else (weights,
persistent constants) is bound in place. The results are copied from the scratch into the trigger op's own outputs. Requires the Metal hooks to be installed before MLX
builds its pipelines (`install()` runs from `kyojin_mlx/__init__` when KYOJIN_REPLAY is set)."""
from __future__ import annotations

import ctypes
import os
import subprocess

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_REPLAY", "0") not in ("0", "")
CHECK = int(os.environ.get("KYOJIN_REPLAY_CHECK", "2"))          # shadow checks (eager next to the replay, bitwise) on the first replays of an entry
TRIAL = os.environ.get("KYOJIN_REPLAY_TRIAL", "1") != "0"        # replay of the recorded inputs right after the recording (must reproduce it); 0 only for timing runs whose outputs are garbage (tools/light.py)
AUDIT = int(os.environ.get("KYOJIN_REPLAY_AUDIT", "256"))        # then one replay in AUDIT is shadow-checked too (0 = never): a region that stopped being static is caught within AUDIT calls
RECORD_AFTER = int(os.environ.get("KYOJIN_REPLAY_AFTER", "4"))      # a signature is recorded on its Nth call (the recording stalls ~50-100 ms: rare signatures, e.g. odd row-count patterns, stay eager)
SLOTS = int(os.environ.get("KYOJIN_REPLAY_SLOTS", "3"))          # ICB copies per entry (calls in flight); more are made on demand
MAX_ENTRIES = 64
BUDGET = int(float(os.environ.get("KYOJIN_REPLAY_MB", "1536")) * 1e6)       # private scratch of all ready entries; a signature that does not fit stays eager
VERBOSE = os.environ.get("KYOJIN_REPLAY_VERBOSE", "0") not in ("0", "")
MONITOR = float(os.environ.get("KYOJIN_REPLAY_MONITOR", "0"))                    # seconds: print the replay state and the MLX memory periodically (leak hunting)
STATS = dict(eager=0, record=0, replay=0, check=0, mismatch=0, dead=0, build_fail=0)

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None
_installed = False
_scratch_slot = 60                                                # recording slots 60.. are ours (the tape tools use 0..15)
_live: list = []                                                  # ready entries


def _load():
    global _lib
    if _lib is not None:
        return _lib
    d = os.path.join(_HERE, "..", "replay")
    so = os.path.join(d, "build", "librp.dylib")
    src = os.path.join(d, "rp.mm")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(so), exist_ok=True)
        subprocess.run(["clang++", "-O2", "-std=c++17", "-fobjc-arc", "-dynamiclib", "-framework", "Metal", "-framework", "Foundation", src, "-o", so],
                       check=True, capture_output=True)
    C = ctypes
    lib = C.CDLL(so)
    lib.rp_region_new.restype = C.c_int
    lib.rp_region_build.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int]
    lib.rp_region_build.restype = C.c_int
    lib.rp_region_stat.argtypes = [C.c_int, C.c_int]
    lib.rp_region_stat.restype = C.c_long
    lib.rp_region_drop.argtypes = [C.c_int]
    lib.rp_region_free.argtypes = [C.c_int]
    lib.rp_rec_begin.argtypes = [C.c_int]
    lib.rp_rec_end.restype = C.c_int
    lib.rp_rec_snap_born.argtypes = [C.c_int, C.c_int]
    _lib = lib
    return lib


def _report() -> None:
    if STATS["record"] or STATS["replay"]:
        import sys
        print(f"replay.STATS {STATS}", file=sys.stderr, flush=True)


def install() -> None:
    """Install the Metal hooks (idempotent). Must run before MLX builds the pipelines the regions use."""
    global _installed
    if _installed:
        return
    _load().rp_install(0 if os.environ.get("RP_NOICB") else 1)
    _installed = True
    import atexit
    atexit.register(_report)                                           # one line at exit: the logs of the gates show that regions were recorded and replayed


def _addr(x: mx.array) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(x)))


_VIEW = {mx.float16: mx.uint16, mx.bfloat16: mx.uint16, mx.float32: mx.uint32}


def same(a: mx.array, b: mx.array) -> bool:
    """Bitwise equality (NaN payloads and signed zeros count)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    v = _VIEW.get(a.dtype)
    if v is not None:
        a, b = a.view(v), b.view(v)
    return bool(mx.array_equal(a, b).item())


def _flat(out):
    if isinstance(out, mx.array):
        return [out], lambda xs: xs[0]
    kind = type(out)
    assert isinstance(out, (list, tuple)) and all(isinstance(o, mx.array) for o in out), "a region returns an array or a flat list / tuple of arrays"
    return list(out), (lambda xs: kind(xs))


class _Entry:
    __slots__ = ("calls", "state", "h", "trig", "shapes", "dtypes", "checks", "wrap", "n_in", "bytes", "replays")

    def __init__(self):
        self.calls = 0
        self.replays = 0
        self.state = "new"            # new -> warm -> ready | dead
        self.h = -1
        self.trig = None
        self.checks = 0


import weakref
_regions = weakref.WeakSet()
_monitor_on = []


def _monitor() -> None:
    import threading, time

    def run():
        lib = _load()
        while True:
            time.sleep(MONITOR)
            ents = [(r.name, e) for r in list(_regions) for e in list(r.entries.values())]
            slots = sum(lib.rp_region_stat(e.h, 8) for _, e in ents if e.h >= 0 and e.state == "ready")
            print(f"replay.monitor: {STATS} entries {len(ents)} ready {sum(e.state == 'ready' for _, e in ents)} ICB slots {slots} | MLX active {mx.get_active_memory() / 1e6:.0f} MB cache {mx.get_cache_memory() / 1e6:.0f} MB", flush=True)
    threading.Thread(target=run, daemon=True).start()


def _release(entries: dict) -> None:
    """A dropped Region (e.g. a test's drafter): give the scratch of its ready entries back. The GPU is synchronised first."""
    try:
        lib = _load()
        mx.synchronize()
        for e in entries.values():
            if e.state == "ready" and e.h >= 0:
                lib.rp_region_free(e.h)
                e.state = "dead"
                if e in _live:
                    _live.remove(e)
    except Exception:
        pass


class Region:
    def __init__(self, fn, name: str):
        self.fn, self.name = fn, name
        self.entries: dict = {}
        weakref.finalize(self, _release, self.entries).atexit = False
        _regions.add(self)
        if MONITOR and not _monitor_on:
            _monitor_on.append(1)
            _monitor()

    @staticmethod
    def _key(args):
        return tuple((tuple(a.shape), a.dtype) if isinstance(a, mx.array) else ("v", repr(a)) for a in args)

    def __call__(self, *args):
        e = self.entries.get(key := self._key(args))
        if e is None:
            if len(self.entries) >= MAX_ENTRIES:                                  # shapes seen once (ring still growing) make room for steady-state ones
                for k, v in self.entries.items():
                    if v.state != "ready":
                        del self.entries[k]
                        break
                else:
                    STATS["eager"] += 1
                    return self.fn(*args)
            e = self.entries[key] = _Entry()
        if e.state == "dead":
            STATS["eager"] += 1
            return self.fn(*args)
        if e.state == "ready":
            return self._replay(e, args)
        e.calls += 1
        if e.calls < RECORD_AFTER:                                       # eager first: lets lazily built kernels / parameters exist, and keeps one-off signatures out of the table
            e.state = "warm"
            STATS["eager"] += 1
            return self.fn(*args)
        return self._record(e, args)

    # -- recording -------------------------------------------------------------------------------------------------------------------------------
    def _arrays(self, args):
        return [a for a in args if isinstance(a, mx.array)]

    def _record(self, e: _Entry, args):
        import time
        t_rec = time.perf_counter()
        lib = _load()
        arrs = [mx.contiguous(a) for a in self._arrays(args)]
        mx.eval(*arrs)
        mx.synchronize()
        it = iter(arrs)
        cargs = [next(it) if isinstance(a, mx.array) else a for a in args]
        spans = sorted((_addr(a), _addr(a) + max(a.nbytes, 1)) for a in arrs)
        if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):                  # two inputs share memory (rings restored from one prefix snapshot): their bindings could not be told apart, record a later call
            e.calls = 0
            STATS["eager"] += 1
            return self.fn(*args)
        old = mx.set_cache_limit(0)                                      # every allocation of the recorded call is a new Metal buffer (born tracking is exact)
        mx.clear_cache()
        ncmd = -1
        try:
            lib.rp_rec_begin(_scratch_slot)
            lib.rp_rec_snap_born(_scratch_slot, 1)
            out = self.fn(*cargs)
            flat, rebuild = _flat(out)
            mx.eval(*flat)
            mx.synchronize()
            ncmd = lib.rp_rec_end()
        finally:
            if ncmd < 0:                                                  # fn raised: close the recording
                lib.rp_rec_end()
            mx.set_cache_limit(old)
        STATS["record"] += 1
        h = lib.rp_region_new()
        n_in, n_out = len(arrs), len(flat)
        ip = (ctypes.c_size_t * n_in)(*[_addr(a) for a in arrs])
        il = (ctypes.c_size_t * n_in)(*[a.nbytes for a in arrs])
        op = (ctypes.c_size_t * n_out)(*[_addr(o) for o in flat])
        ol = (ctypes.c_size_t * n_out)(*[o.nbytes for o in flat])
        rc = lib.rp_region_build(h, _scratch_slot, n_in, ip, il, n_out, op, ol, SLOTS)
        if rc != 0 or ncmd == 0:
            STATS["build_fail"] += 1
            e.state = "dead"
            if VERBOSE:
                print(f"replay[{self.name}]: build failed rc={rc} cmds={ncmd}: eager from now on", flush=True)
            return out
        e.h, e.n_in = h, n_in
        e.shapes = [tuple(o.shape) for o in flat]
        e.dtypes = [o.dtype for o in flat]
        e.wrap = rebuild
        e.trig = mx.fast.metal_kernel(name=f"kyojin_region_trigger_{h}", input_names=[f"i{k}" for k in range(n_in)], output_names=[f"o{k}" for k in range(n_out)],
                                      source="if (thread_position_in_grid.x == 0) { o0[0] = o0[0]; }")
        e.bytes = lib.rp_region_stat(h, 3) + SLOTS * 1_000_000
        if sum(x.bytes for x in _live) + e.bytes > BUDGET:                    # no room for another region's scratch: this signature stays eager (ready entries may have lazy trigger ops in flight, never released)
            lib.rp_region_free(h)
            STATS["build_fail"] += 1
            e.state = "dead"
            if VERBOSE:
                print(f"replay[{self.name}]: scratch budget {BUDGET / 1e6:.0f} MB reached: eager for this signature", flush=True)
            return out
        e.state = "ready"
        _live.append(e)
        trial = e.trig(inputs=arrs, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)     # replay of the recorded inputs: must give the recorded results (also builds the trigger pipeline now, not on a serving call)
        mx.eval(*trial)
        if TRIAL and not all(same(a, b) for a, b in zip(trial, flat)):
            self._kill(e, "the replay of the recorded call differs from it")
            return out
        if VERBOSE:
            print(f"replay[{self.name}]: recorded {ncmd} dispatches, {lib.rp_region_stat(h, 6)} rebinds per call, {lib.rp_region_stat(h, 4)} scratch buffers "
                  f"({lib.rp_region_stat(h, 3) / 1e6:.1f} MB), {lib.rp_region_stat(h, 5)} bound in place, barriers {lib.rp_region_stat(h, 9)} + {lib.rp_region_stat(h, 10)} added, {SLOTS} ICB slots, "
                  f"{1e3 * (time.perf_counter() - t_rec):.0f} ms for the recording call + build + trial", flush=True)
        return out

    # -- replay ----------------------------------------------------------------------------------------------------------------------------------
    def _replay(self, e: _Entry, args):
        arrs = [mx.contiguous(a) for a in self._arrays(args)]
        outs = e.trig(inputs=arrs, grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=e.shapes, output_dtypes=e.dtypes)
        STATS["replay"] += 1
        e.replays += 1
        if e.checks < CHECK or (AUDIT and e.replays % AUDIT == 0):
            e.checks += 1
            STATS["check"] += 1
            ref = _flat(self.fn(*args))[0]
            mx.eval(*outs, *ref)
            if not all(same(a, b) for a, b in zip(outs, ref)):
                STATS["mismatch"] += 1
                self._kill(e, "shadow check differs")
                return e.wrap(ref)
        return e.wrap(outs)

    def _kill(self, e: _Entry, why: str) -> None:
        """Entry dead: every call of this signature goes through `fn` from now on, the scratch of the region is released (the GPU is idle: the check evaluated)."""
        STATS["dead"] += 1
        e.state = "dead"
        _load().rp_region_free(e.h)
        if e in _live:
            _live.remove(e)
        if VERBOSE:
            print(f"replay[{self.name}]: {why}: entry dead, eager from now on", flush=True)

    def stats(self) -> dict:
        lib = _load()
        return {str(k): dict(state=v.state, replays=lib.rp_region_stat(v.h, 0) if v.h >= 0 else 0, rebinds=lib.rp_region_stat(v.h, 1) if v.h >= 0 else 0,
                              slots=lib.rp_region_stat(v.h, 8) if v.h >= 0 else 0) for k, v in self.entries.items()}


def region(fn, name: str = "region"):
    """`fn` as a replayed region when KYOJIN_REPLAY is on (and the hooks are installed), else `fn` itself."""
    if not ENABLED:
        return fn
    try:
        install()
    except Exception as ex:                                               # no clang / no Metal hooks on this machine: the normal path
        import sys
        print(f"replay: hooks not installed ({ex}); regions run eagerly", file=sys.stderr, flush=True)
        return fn
    return Region(fn, name)

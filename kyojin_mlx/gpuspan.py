# SPDX-License-Identifier: Apache-2.0
"""GPU busy and idle of one served round, from the per-command-buffer GPU start/end stamps (probe, flag `gpu_span`).

MLX gives no GPU timer; Metal does: every command buffer carries `GPUStartTime` / `GPUEndTime`. tools/cbcount/cbcount.mm
swizzles `-[MTLCommandBuffer commit]`, numbers each commit and keeps (commit time, GPU start, GPU end) for the last 65536.
A round is a contiguous commit range: the engine brackets the tick with `begin()` / `end()`, so

    busy  = length of the union of the round's [GPUStart, GPUEnd] spans   (the GPU really executing this round)
    idle  = (last GPU end - first GPU start) - busy                        (holes between the round's buffers)
    starve= first GPU start - first commit                                 (the GPU had nothing while the host committed)
    lead  = peak (committed - completed) inside the round                  (how far the host ran ahead: MLX's MAX_ACTIVE_TASKS is 10)

No MLX change, no xctrace: the counters read the driver's own stamps, so nothing is added to the GPU.
Exactness: the values are timers, the tokens do not see them.
"""
from __future__ import annotations

import ctypes
import json
import os
import time

ON = os.environ.get("KYOJIN_GPU_SPAN", "0") not in ("0", "")
KERN = os.environ.get("KYOJIN_GPU_KERN", "0") not in ("0", "")   # also price every compute dispatch by its Metal function name
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cbcount", "libcbcount.dylib")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cbcount", "cbcount.mm")
_lib = None
_seq = [0]
_t0 = [0.0]
_kn = [{}]       # name id -> name, filled once


def _clock() -> float:
    """The Metal stamps' clock (mach uptime), so head / tail are the same units as the spans."""
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) * 1e-9 if hasattr(time, "CLOCK_UPTIME_RAW") else time.monotonic()


def install() -> bool:
    """Load (build once) the counters. False when the library is missing and cannot be built: the probe then costs nothing."""
    global _lib
    if _lib is not None:
        return True
    try:
        if not os.path.isfile(LIB):
            import subprocess
            subprocess.run(["clang++", "-O2", "-std=c++17", "-fobjc-arc", "-dynamiclib", "-framework", "Metal",
                            "-framework", "Foundation", SRC, "-o", LIB], check=True, capture_output=True)
        L = ctypes.CDLL(LIB)
        L.cb_install()
        L.cb_commits.restype = ctypes.c_long
        L.cb_commit_time.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        L.cb_times.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        if hasattr(L, "cb_ndisp"):
            L.cb_ndisp.argtypes = [ctypes.c_long]
            L.cb_ndisp.restype = ctypes.c_long
            L.cb_disp.argtypes = [ctypes.c_long, ctypes.c_int]
            L.cb_disp.restype = ctypes.c_int
            L.cb_kname.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            L.cb_kname.restype = ctypes.c_int
            L.cb_rec.argtypes = [ctypes.c_int]
            L.cb_rec(1 if KERN else 0)                   # recording follows the flag, also for a server that installed the hooks at startup
        _lib = L
        return True
    except Exception:
        return False


def begin() -> None:
    if not (ON or KERN):
        return
    if _lib is None and not install():
        return
    _lib.cb_rec(1 if KERN else 0)           # the flag can be flipped on a loaded server: recording follows it on every tick
    _seq[0] = _lib.cb_commits()
    _t0[0] = _clock()


def kname(i: int) -> str:
    """Metal function name of interned name id i (cached)."""
    d = _kn[0]
    s = d.get(i)
    if s is None:
        buf = ctypes.create_string_buffer(256)
        _lib.cb_kname(i, buf, 256)
        s = buf.value.decode("utf-8", "replace")
        d[i] = s
    return s


def _sane(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_.-") else "_" for c in s)


def _merge(iv: list) -> list:
    """Union of the spans as a sorted list of disjoint intervals (the GPU can run overlapping buffers: gaps must be read off
    this, not off the commit order, or a gap inside an overlap is counted twice)."""
    if not iv:
        return []
    iv = sorted(iv)
    out, (a, b) = [], iv[0]
    for s, e in iv[1:]:
        if s > b:
            out.append((a, b))
            a, b = s, e
        else:
            b = max(b, e)
    out.append((a, b))
    return out


SEQ_LOG = os.environ.get("KYOJIN_GPU_SEQ_LOG", "")        # probe only, gpu_kern on: write the kernel order of one one-row round here
BIG_LOG = os.environ.get("KYOJIN_GPU_BIG_LOG", "/tmp/kyojin_gpu_big.log")      # probe only, gpu_span on: one line per round with a gap over 15 ms

MARKS = []
JOBS = []
TL_LOG = os.environ.get("KYOJIN_GPU_TL", "")           # probe only: per-buffer timeline of every completed round (decgap)
_prev = [None]       # (first, last+1) commit of the previous round: read again one round later, when its stamps have landed


def end(stats: dict, rows: int, prefix: str = "gpu") -> None:
    """Close the round opened by `begin()` and fold its numbers into `stats` (per row count, like spec_rt_r<rows>). The same fold runs on the
    previous round's commit range under prefix `<prefix>L`: a pipelined step is still in flight at its own tick's end (its stamps are missing from
    the `gpu_*` numbers), one tick later every buffer of it has completed, so `gpuL_span_r<rows>` is the step's true first-start to last-end span."""
    if not (ON or KERN) or _lib is None:
        return
    s0, s1 = _seq[0], _lib.cb_commits()
    if TL_LOG and s1 > s0 and rows <= 8:                    # host layer stamps of this round, in ms from its first commit
        c0 = ctypes.c_double()
        _lib.cb_commit_time(s0, ctypes.byref(c0))
        with open(TL_LOG, "a") as fh:
            fh.write(json.dumps({"marks": rows, "t0": round((_t0[0] - c0.value) * 1e3, 3), "m": [[round((a - c0.value) * 1e3, 3), round((b - c0.value) * 1e3, 3)] for a, b in MARKS], "j": [[p, round((a - c0.value) * 1e3, 3), round((b - c0.value) * 1e3, 3), n] for p, a, b, n in JOBS]}) + "\n")
    del MARKS[:]
    del JOBS[:]
    if _prev[0] is not None:
        _fold(stats, rows, prefix + "L", *_prev[0])
    _prev[0] = (s0, s1)
    _fold(stats, rows, prefix, s0, s1)


def _fold(stats: dict, rows: int, prefix: str, s0: int, s1: int) -> None:
    if s1 <= s0:
        return
    t0, t1 = ctypes.c_double(), ctypes.c_double()
    ct = ctypes.c_double()
    spans, commits = [], []
    kn = {} if KERN else None
    ktxt = 0.0
    kb1 = kmix = 0
    seq = [] if (KERN and SEQ_LOG) else None
    tl = []
    for s in range(s0, s1):
        _lib.cb_times(s, ctypes.byref(t0), ctypes.byref(t1))
        _lib.cb_commit_time(s, ctypes.byref(ct))
        commits.append(ct.value)
        if TL_LOG:
            nm = ""
            if KERN and _lib.cb_ndisp(s) > 0:
                nm = _sane(kname(_lib.cb_disp(s, 0)))
            tl.append([ct.value, t0.value, t1.value, nm])
        if t1.value > 0 and t0.value > 0 and t1.value > t0.value and t1.value - t0.value < 2.0:
            spans.append((t0.value, t1.value))            # a buffer whose stamps never landed (0/0) or a bogus one is left out
            if kn is not None:
                n = _lib.cb_ndisp(s)
                ms = (t1.value - t0.value) * 1e3
                if seq is not None:
                    seq.extend(f"{_sane(kname(_lib.cb_disp(s, i)))} {ms / n:.1f}" for i in range(n))
                ktxt += ms
                if n == 1:
                    kb1 += 1
                    k = _sane(kname(_lib.cb_disp(s, 0)))
                    a, b = kn.get(k, (0.0, 0))
                    kn[k] = (a + ms, b + 1)
                elif n > 1:
                    kmix += 1
                    for i in range(n):                   # an even split: only exact when one dispatch per buffer (MLX_MAX_OPS_PER_BUFFER=1)
                        k = _sane(kname(_lib.cb_disp(s, i)))
                        a, b = kn.get(k, (0.0, 0))
                        kn[k] = (a + ms / n, b + 1)
    if seq and rows == 1 and not os.path.exists(SEQ_LOG):
        with open(SEQ_LOG, "w") as fh:                       # the ordered kernel names (and us each) of the first one-row round
            fh.write("\n".join(seq))
    if not spans:
        return
    if TL_LOG and prefix.endswith("L") and rows <= 8:     # probe: one JSON line per completed round, per buffer (commit, GPU start, GPU end) in ms from the first commit
        o = min(commits)
        with open(TL_LOG, "a") as fh:
            fh.write(json.dumps({"rows": rows, "b": [[round((c - o) * 1e3, 3), round((a - o) * 1e3, 3), round((e - o) * 1e3, 3), n] for c, a, e, n in tl]}) + "\n")
    m = _merge(spans)
    busy = sum(e - a for a, e in m)
    beg, fin = m[0][0], m[-1][1]
    gaps = [(b[0] - a[1]) * 1e3 for a, b in zip(m, m[1:])]
    lead, cur, ev = 0, 0, sorted([(c, 1) for c in commits] + [(e, -1) for _, e in spans])
    for _, k in ev:
        cur += k
        lead = max(lead, cur)
    r = rows
    st = stats
    st[f"{prefix}_n_r{r}"] = st.get(f"{prefix}_n_r{r}", 0) + 1
    st[f"{prefix}_buf_r{r}"] = st.get(f"{prefix}_buf_r{r}", 0.0) + (s1 - s0)
    st[f"{prefix}_busy_r{r}"] = st.get(f"{prefix}_busy_r{r}", 0.0) + busy * 1e3
    st[f"{prefix}_idle_r{r}"] = st.get(f"{prefix}_idle_r{r}", 0.0) + (fin - beg - busy) * 1e3
    st[f"{prefix}_span_r{r}"] = st.get(f"{prefix}_span_r{r}", 0.0) + (fin - beg) * 1e3
    st[f"{prefix}_gap_r{r}"] = st.get(f"{prefix}_gap_r{r}", 0.0) + sum(gaps)
    st[f"{prefix}_gapl_r{r}"] = st.get(f"{prefix}_gapl_r{r}", 0.0) + sum(g for g in gaps if g > 0.5)
    st[f"{prefix}_gmax_r{r}"] = max(st.get(f"{prefix}_gmax_r{r}", 0.0), gaps and max(gaps) or 0.0)
    # idle by cause: at the start of a gap, buffers already committed but not finished (queued) or nothing left to run (host)
    iq = ih = 0.0
    for a, b in zip(m, m[1:]):
        g = (b[0] - a[1]) * 1e3
        pend = sum(1 for c in commits if c <= a[1]) - sum(1 for _, e in spans if e <= a[1])
        if pend > 0:
            iq += g
        else:
            ih += g
    st[f"{prefix}_idleq_r{r}"] = st.get(f"{prefix}_idleq_r{r}", 0.0) + iq
    st[f"{prefix}_idleh_r{r}"] = st.get(f"{prefix}_idleh_r{r}", 0.0) + ih
    st[f"{prefix}_g50_r{r}"] = st.get(f"{prefix}_g50_r{r}", 0) + sum(1 for g in gaps if g > 0.5)
    if BIG_LOG:                                              # every gap over 15 ms: when, in which round shape, between which buffers
        big = [(i, round(g, 1)) for i, g in enumerate(gaps) if g > 15.0]
        if big:
            with open(BIG_LOG, "a") as fh:
                fh.write(f"{time.time():.3f} {prefix} r={r} bufs={len(m)} span={(fin - beg) * 1e3:.1f} big={big}\n")
    st[f"{prefix}_gc_r{r}"] = st.get(f"{prefix}_gc_r{r}", 0) + len(gaps)
    now = _clock()
    st[f"{prefix}_tick_r{r}"] = st.get(f"{prefix}_tick_r{r}", 0.0) + (now - _t0[0]) * 1e3
    st[f"{prefix}_head_r{r}"] = st.get(f"{prefix}_head_r{r}", 0.0) + (beg - _t0[0]) * 1e3
    st[f"{prefix}_tail_r{r}"] = st.get(f"{prefix}_tail_r{r}", 0.0) + (now - fin) * 1e3
    st[f"{prefix}_starve_r{r}"] = st.get(f"{prefix}_starve_r{r}", 0.0) + (beg - min(commits)) * 1e3
    st[f"{prefix}_lead_r{r}"] = max(st.get(f"{prefix}_lead_r{r}", 0), lead)
    if kn is not None:                                   # probe gpu_kern: GPU ms per Metal kernel name (one dispatch per buffer is exact)
        st[f"{prefix}_kspan_r{r}"] = st.get(f"{prefix}_kspan_r{r}", 0.0) + ktxt
        st[f"{prefix}_kb1_r{r}"] = st.get(f"{prefix}_kb1_r{r}", 0) + kb1
        st[f"{prefix}_kmix_r{r}"] = st.get(f"{prefix}_kmix_r{r}", 0) + kmix
        for k, (ms, cnt) in kn.items():
            kk = f"{prefix}_k_{k}_r{r}"
            st[kk] = st.get(kk, 0.0) + ms
            kk = f"{prefix}_kc_{k}_r{r}"
            st[kk] = st.get(kk, 0) + cnt

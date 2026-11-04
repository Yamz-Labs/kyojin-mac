"""Server lifecycle on the Mac: start under the caller's lock (one heavy job at a time), keep the machine awake with `caffeinate -ims -w <pid>`
(never -d / -u), wait for GET /health == 200 (the kyojin server opens its port after the load), sample the footprint with `top` (NOT ps RSS),
scan the log for OOM, kill by PID only (no pkill). Adapted from the CUDA/ROCm server.py."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import backends


def health(port: int) -> int:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status
    except urllib.error.HTTPError as e:      # 503 while loading
        return e.code
    except Exception:
        return 0


OOM_MARKERS = ("out of memory", "metal::malloc", "kiogpucommandbuffercallbackerrorotheroutofmemory", "insufficient memory", "std::bad_alloc")


def oom_in(log: Path) -> bool:
    try:
        t = log.read_text(errors="replace").lower()
    except OSError:
        return False
    return any(m in t for m in OOM_MARKERS)


def guard_kill_for(guard_log: str, pid: int) -> str | None:
    """The macguard line that TERMed `pid` (a probe killed by the guard is 'does not fit under the cap', not a crash of the model)."""
    try:
        for line in reversed(Path(guard_log).read_text(errors="replace").splitlines()[-200:]):
            if f"TERM {pid} " in line:
                return line.strip()
    except OSError:
        pass
    return None


class Handle:
    """What running_server yields: the process, whether it came up, and the footprint seen while it ran."""

    def __init__(self, proc, ok):
        self.proc, self.ok, self.peak_gib, self.stop = proc, ok, 0.0, threading.Event()

    def sample(self):
        while True:
            f = backends.footprint_gib(self.proc.pid)
            if f:
                self.peak_gib = max(self.peak_gib, f)
            if self.stop.wait(4):
                return

    def over_cap(self, cap_gib: float) -> bool:
        return self.peak_gib > cap_gib


@contextlib.contextmanager
def running_server(prof: dict, repo: Path, argv: list[str], port: int, log: Path, env_extra: dict | None = None, ready_s: int = 1500, logf=print):
    """Yield a Handle; ok=False (process exited or timeout) lets the caller treat it as 'does not fit'."""
    env = {**os.environ, **prof["env"], **(env_extra or {})}
    fh = open(log, "wb")
    p = subprocess.Popen([prof["python"], "-u", *argv], cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    awake = None
    if os.uname().sysname == "Darwin":
        awake = subprocess.Popen(["caffeinate", "-ims", "-w", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    h = Handle(p, False)
    threading.Thread(target=h.sample, daemon=True).start()
    t0 = time.time()
    try:
        while time.time() - t0 < ready_s:
            if p.poll() is not None:
                break
            if health(port) == 200:
                h.ok = True; break
            time.sleep(2)
        logf(f"server pid {p.pid} {'ready' if h.ok else 'NOT ready'} after {time.time()-t0:.0f} s, footprint {h.peak_gib:.2f} GiB")
        yield h
    finally:
        h.stop.set()
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)       # own process group, scoped to our PID
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(30)
        if awake and awake.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                awake.terminate()
        fh.close()

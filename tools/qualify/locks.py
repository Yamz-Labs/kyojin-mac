"""GPU lock on the Mac: ONE heavy job at a time (25/10 kernel panic: two 29 GiB jobs on 16 GB). flock on the same file as `lockf -k ~/kyojin-mlx/.gpu.lock`
(flock(2) on macOS, the same lock the day-1 runner and the gates take). A job takes it for its whole run and releases it between jobs, so another
agent's job is never blocked longer than one qualify job. Before each job: free memory and the tail of the macguard log."""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

MIN_FREE_FRAC = 0.30     # memory_pressure free % needed to start a job (the guard kills below 8 %)


@contextlib.contextmanager
def gpu_locks(paths: list[str], log=print, wait_s: float = 6 * 3600):
    """Take every lock in order (flock, polling so a wait can be reported); release in reverse order."""
    fds = []
    try:
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
            t0 = time.time(); said = False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.time() - t0 > wait_s:
                        os.close(fd); raise TimeoutError(f"lock {p} not free after {wait_s:.0f} s")
                    if not said:
                        log(f"waiting for lock {p}"); said = True
                    time.sleep(5)
            fds.append(fd)
            log(f"lock held {p} (waited {time.time()-t0:.0f} s)")
        yield
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
        log("locks released")


def guard_tail(guard_log: str, n: int = 3) -> list[str]:
    try:
        return Path(guard_log).read_text(errors="replace").strip().splitlines()[-n:]
    except OSError:
        return []

"""Hard memory guard of the serve engine (independent of any cause of a memory spike).

Budget: KYOJIN_MEM_BUDGET_GIB (default 0.88 x the physical RAM, i.e. about 225 GiB on a 256 GB Mac; 0 switches the guard off).
Used memory: the larger of MLX active + buffer cache and the process physical footprint (the footprint also sees what MLX does not count).
Slope: KYOJIN_MEM_GIB_PER_1K (default 0.12), GiB of active memory per 1000 prompt tokens: the leak-free footprint of 1.10.2 (KV ~0.11 GiB per 1K
plus cache, 106 -> 126 GiB from 4.7K to 134K tokens; 1.10.0 / 1.10.1 grew 0.36 per 1K through the step-compile leak). It is a constant on purpose: the engine's own bytes-per-token (`Engine.bpt`) bills the
cache ledger, which does not see the private copy a prefix hit makes, so it under-estimates what the process really holds.

The class is pure (readers are injected), so the projection and the escalation are tested on CPU without Metal."""
from __future__ import annotations

import ctypes
import os
import sys

GIB = 2 ** 30
DEFAULT_FRACTION = 0.88
DEFAULT_GIB_PER_1K = 0.12
DEFAULT_CHECK_TICKS = 16                         # decode ticks between two in-flight checks (prefill chunks are checked every chunk)


class MemoryBudgetExceeded(RuntimeError):
    """A request that cannot fit in the memory budget (admission: HTTP 413) or was stopped in flight (HTTP 503 / SSE error)."""


class PromptTooLong(MemoryBudgetExceeded):
    pass


def physical_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def phys_footprint_bytes() -> int | None:
    """Physical footprint of this process (macOS proc_pid_rusage, ri_phys_footprint); None elsewhere or on error. One cheap syscall."""
    if sys.platform != "darwin":
        return None
    try:
        buf = (ctypes.c_uint64 * 40)()                                  # rusage_info_v2 is 36 x 8 bytes; room to spare
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
            return None
        return int(buf[9])                                             # v2 layout: uuid(2) + user, system, pkg_idle, interrupt, pageins, wired, resident_size, phys_footprint
    except Exception:                                                  # noqa: BLE001
        return None


def budget_bytes(env=None, ram: int | None = None) -> int | None:
    """The byte budget, or None when the guard is off (KYOJIN_MEM_BUDGET_GIB=0, or no RAM size and no explicit value)."""
    env = os.environ if env is None else env
    raw = env.get("KYOJIN_MEM_BUDGET_GIB")
    if raw not in (None, ""):
        gib = float(raw)
        return int(gib * GIB) if gib > 0 else None
    ram = physical_ram_bytes() if ram is None else ram
    return int(DEFAULT_FRACTION * ram) if ram else None


def slope_bytes_per_token(env=None) -> float:
    env = os.environ if env is None else env
    return float(env.get("KYOJIN_MEM_GIB_PER_1K") or DEFAULT_GIB_PER_1K) * GIB / 1000.0


class MemGuard:
    def __init__(self, budget: int, slope: float, read_active, read_cache, read_footprint=None, clear_cache=None, check_ticks: int = DEFAULT_CHECK_TICKS):
        self.budget, self.slope = int(budget), float(slope)
        self._active, self._cache, self._foot, self._clear = read_active, read_cache, read_footprint, clear_cache
        self.check_ticks = max(1, int(check_ticks))

    @classmethod
    def from_env(cls, env=None, ram: int | None = None):
        """The guard the server runs with, on MLX readers; None when switched off."""
        env = os.environ if env is None else env
        b = budget_bytes(env, ram)
        if b is None:
            return None
        import mlx.core as mx
        return cls(b, slope_bytes_per_token(env), mx.get_active_memory, mx.get_cache_memory, phys_footprint_bytes, mx.clear_cache,
                   int(env.get("KYOJIN_MEM_CHECK_TICKS") or DEFAULT_CHECK_TICKS))

    def active(self) -> int:
        return int(self._active())

    def cache(self) -> int:
        return int(self._cache())

    def used(self) -> int:
        """In-flight reading: MLX active + buffer cache, or the process footprint when that is larger."""
        u = self.active() + self.cache()
        f = self._foot() if self._foot is not None else None
        return max(u, int(f)) if f else u

    def base(self) -> int:
        """Admission reading: what is held for good. The reclaimable MLX buffer cache is left out (the footprint less that cache counts)."""
        f = self._foot() if self._foot is not None else None
        a = self.active()
        return max(a, int(f) - self.cache()) if f else a

    def clear_cache(self) -> None:
        if self._clear is not None:
            self._clear()

    def project(self, tokens: int) -> int:
        return self.base() + int(tokens * self.slope)

    def max_tokens(self) -> int:
        return max(int((self.budget - self.base()) / self.slope), 0) if self.slope > 0 else 0

    def admit(self, tokens: int) -> None:
        """Refuse a prompt whose projected memory (everything held now + all its tokens, which also covers a prefix-restore copy) is over the budget."""
        if self.project(tokens) > self.budget:
            raise PromptTooLong(f"prompt too long for the memory budget: {tokens} tokens would take the process to about {self.project(tokens) / GIB:.0f} GiB, "
                                f"the budget is {self.budget / GIB:.0f} GiB (KYOJIN_MEM_BUDGET_GIB); max ~{self.max_tokens()} tokens now")

    def over(self, extra: int = 0) -> bool:
        return self.used() + int(extra) > self.budget

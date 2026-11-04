"""Residency and buffer-cache policy fitted to the live set: KYOJIN_WIRED_FIT, KYOJIN_CACHE_FIT.

wire_memory (serve.backends) wires up to max_recommended_working_set_size before the weights exist, so the MLX residency set has room for every
buffer MLX allocates later: each transient output that is born inside a step is added to the set, and the set is committed (made resident) before
the command buffer that uses it. WIRED_FIT sets the limit to the live set after the load plus a headroom (KYOJIN_WIRED_FIT_MB, default 1024), so
the weights stay wired and the per-step transients, which come out of the buffer cache, do not each grow the set. The rival engine has the same policy as an opt-in
(`fit`, zero headroom; its default wires the maximum like wire_memory).

MLX also trims its buffer cache once active + cache memory passes 0.95 of max_recommended_working_set_size (the allocator's gc limit): the trim
runs inside malloc, on the thread that builds the graph. On a 172 GiB pack the cache limit of weights / 10 (16 GiB) reaches past that line.
CACHE_FIT lowers the cache limit to the room left under it, so the cache is trimmed by the completion path (free) instead of inside malloc.

The sizing is pure integer arithmetic (fit_wired, fit_cache); apply() reads the memory counters and calls the MLX setters. Same bytes, same kernels:
the output does not change."""
from __future__ import annotations

import os

MIB = 1 << 20
GC_FRACTION = 0.95            # MLX allocator: cached buffers are released when active + cache would pass this part of the recommended working set
WIRED_FIT = os.environ.get("KYOJIN_WIRED_FIT", "0") == "1"
WIRED_FIT_MB = int(os.environ.get("KYOJIN_WIRED_FIT_MB", "1024"))
CACHE_FIT = os.environ.get("KYOJIN_CACHE_FIT", "0") == "1"
CACHE_FIT_MB = int(os.environ.get("KYOJIN_CACHE_FIT_MB", "2048"))      # room kept free under the gc line for transients and KV growth


def fit_wired(active: int, recommended: int, headroom_mb: int) -> int:
    """Wired limit in bytes: the live set plus headroom, rounded up to a MiB, never above the recommended working set (0 recommended = no cap)."""
    want = -(-(max(0, active) + max(0, headroom_mb) * MIB) // MIB) * MIB
    return min(want, recommended) if recommended > 0 else want


def fit_cache(active: int, recommended: int, current: int, reserve_mb: int, floor_mb: int = 256) -> int:
    """Buffer-cache limit in bytes: the current limit, lowered to the room under the gc line (GC_FRACTION of recommended) after the live set and a
    reserve; never below floor_mb, never raised."""
    if recommended <= 0:
        return current
    room = int(recommended * GC_FRACTION) - max(0, active) - max(0, reserve_mb) * MIB
    return min(current, max(floor_mb * MIB, room))


def apply(log=print) -> dict:
    """After the weights are loaded: apply the fits that are on. Returns what was set (bytes)."""
    import mlx.core as mx
    rec = int(mx.device_info()["max_recommended_working_set_size"])
    active = int(mx.get_active_memory())
    out = {}
    if WIRED_FIT:
        lim = fit_wired(active, rec, WIRED_FIT_MB)
        mx.set_wired_limit(lim)
        out["wired"] = lim
        log(f"wired fit: limit {lim / 2 ** 30:.1f} GiB = live {active / 2 ** 30:.1f} GiB + {WIRED_FIT_MB} MiB (KYOJIN_WIRED_FIT)")
    if CACHE_FIT:
        cur = mx.set_cache_limit(0)                       # returns the previous limit
        lim = fit_cache(active, rec, cur, CACHE_FIT_MB)
        mx.set_cache_limit(lim)
        out["cache"] = lim
        log(f"cache fit: limit {lim / 2 ** 20:.0f} MB (was {cur / 2 ** 20:.0f} MB; gc line {GC_FRACTION * rec / 2 ** 30:.1f} GiB, live {active / 2 ** 30:.1f} GiB)")
    return out

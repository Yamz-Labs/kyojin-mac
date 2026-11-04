# SPDX-License-Identifier: Apache-2.0
"""Apple-platform levers that need no MLX fork and no Metal at import time (opmac; facts and A/B in docs/reports/opmac_REPORT.md).

eval_qos (KYOJIN_EVAL_QOS, runtime): MLX 0.32.4 encodes a step's GPU work inline on the thread that calls eval / async_eval (transforms.cpp
eval_impl -> gpu::eval; scheduler::enqueue is used by the CPU streams only). That thread is the engine thread. Its QoS class decides which core
type and which frequency the scheduler gives the encode (python graph build + Metal encode: the CPU side of a step, ULTRA.md s.12-14).
`sync_qos()` runs once per engine tick: it moves the calling thread to QOS_CLASS_USER_INTERACTIVE while the flag is on and back to the class it
had when the flag is off. QoS is per thread, so a flags.set() from the HTTP thread takes effect at the engine thread's next tick.

residency_refresh_ms (KYOJIN_RESIDENCY_REFRESH_MS, start only): MLX can re-request residency of its sets every n ms on a background thread
(MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS, resident.cpp). MLX reads it when the Metal device is built, so `apply_residency_refresh_env` runs from
the package __init__ before mlx is imported. 0 = MLX default (no refresh thread)."""
from __future__ import annotations

import ctypes
import os
import sys

QOS_USER_INTERACTIVE = 0x21
QOS_USER_INITIATED = 0x19
QOS_DEFAULT = 0x15
QOS_UTILITY = 0x11
QOS_BACKGROUND = 0x09
QOS_UNSPECIFIED = 0x00
QOS_NAMES = {QOS_USER_INTERACTIVE: "USER_INTERACTIVE", QOS_USER_INITIATED: "USER_INITIATED", QOS_DEFAULT: "DEFAULT",
             QOS_UTILITY: "UTILITY", QOS_BACKGROUND: "BACKGROUND", QOS_UNSPECIFIED: "UNSPECIFIED"}

QOS_ENV = "KYOJIN_EVAL_QOS"
REFRESH_ENV = "KYOJIN_RESIDENCY_REFRESH_MS"
MLX_REFRESH_ENV = "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"
REFRESH_DEFAULT = "1000"                                           # stops macOS dropping the Metal residency mid-prefill (wired-to-anonymous flip, compressor spike); "0" = off

_LIBSYSTEM = "/usr/lib/libSystem.B.dylib"
_state: dict = {}                    # per thread id: QoS class the thread had before we raised it (present while raised)


def qos_name(q: int) -> str:
    return QOS_NAMES.get(q, hex(q))


def _libc():
    lib = ctypes.CDLL(_LIBSYSTEM)
    lib.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
    lib.pthread_set_qos_class_self_np.restype = ctypes.c_int
    lib.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    lib.pthread_get_qos_class_np.restype = ctypes.c_int
    lib.pthread_self.argtypes = []
    lib.pthread_self.restype = ctypes.c_void_p
    return lib


def get_qos(libc) -> int:
    """QoS class of the calling thread."""
    cls, prio = ctypes.c_uint(0), ctypes.c_int(0)
    rc = libc.pthread_get_qos_class_np(libc.pthread_self(), ctypes.pointer(cls), ctypes.pointer(prio))
    if rc != 0:
        raise OSError(rc, "pthread_get_qos_class_np failed")
    return cls.value


def set_qos(libc, qos: int) -> None:
    """Set the QoS class of the calling thread (relative priority 0). OSError on a non-zero return."""
    rc = libc.pthread_set_qos_class_self_np(qos, 0)
    if rc != 0:
        raise OSError(rc, f"pthread_set_qos_class_self_np({hex(qos)}) failed")


def sync_qos(environ=None, libc=None, platform: str | None = None, tid: int | None = None) -> str | None:
    """Bring the calling thread in line with KYOJIN_EVAL_QOS. Returns a log line when something changed (or failed), else None.
    Cheap when nothing changes: one dict lookup and one environment read."""
    env = os.environ if environ is None else environ
    want = env.get(QOS_ENV, "0") not in ("0", "")
    tid = tid if tid is not None else _thread_id()
    raised = tid in _state
    if want == raised:
        return None
    if (platform or sys.platform) != "darwin":
        if want:
            _state[tid] = QOS_UNSPECIFIED                           # remember the request: say it once, not every tick
            return f"[kyojin] {QOS_ENV}=1 ignored: thread QoS is a macOS call"
        _state.pop(tid, None)
        return None
    try:
        lib = libc or _libc()
        before = get_qos(lib)
        if want:
            set_qos(lib, QOS_USER_INTERACTIVE)
            _state[tid] = before
            return f"[kyojin] eval thread QoS {qos_name(before)} -> {qos_name(get_qos(lib))} ({QOS_ENV}=1)"
        orig = _state.pop(tid)
        if orig != QOS_UNSPECIFIED:
            set_qos(lib, orig)
        return f"[kyojin] eval thread QoS {qos_name(before)} -> {qos_name(get_qos(lib))} ({QOS_ENV}=0)"
    except (OSError, AttributeError) as e:
        if want:
            _state[tid] = QOS_UNSPECIFIED                           # do not retry every tick
        else:
            _state.pop(tid, None)
        return f"[kyojin] eval thread QoS change failed: {e}"


def _thread_id() -> int:
    import threading
    return threading.get_ident()


def apply_residency_refresh_env(environ=None) -> bool:
    """KYOJIN_RESIDENCY_REFRESH_MS=n (n > 0) -> MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=n unless already set. Must run before mlx is imported
    (MLX reads it when the Metal device is built); returns False without touching anything once mlx.core is loaded."""
    env = os.environ if environ is None else environ
    if environ is None and "mlx.core" in sys.modules:
        return False
    try:
        ms = int(env.get(REFRESH_ENV, REFRESH_DEFAULT) or 0)
    except ValueError:
        return False
    if ms <= 0:
        return False
    env.setdefault(MLX_REFRESH_ENV, str(ms))
    return True

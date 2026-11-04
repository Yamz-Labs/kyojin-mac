"""Flag tick_trace: where the engine thread spends the first decode round after a prefill. A sampler thread reads the engine thread's Python stack every
10 ms for a few seconds and prints the runs that lasted at least 30 ms (stderr). Observability only, nothing runs with the flag off."""
import os
import sys
import threading
import time
import traceback


def on() -> bool:
    return os.environ.get("KYOJIN_TICK_TRACE", "0") == "1"


def agg() -> bool:
    return os.environ.get("KYOJIN_TICK_AGG", "0") == "1"


_busy = _stop = False


def start_agg(seconds: float = 12.0) -> None:
    """tick_agg: from the first prefill chunk, sum the sampled time per Python stack (leaf 4 frames), print the top 25 when the engine leaves the prefill."""
    global _busy
    if _busy:
        return
    _busy = True
    ident = threading.get_ident()

    def run():
        global _busy, _stop
        _stop = False
        tot, t0, t = {}, time.perf_counter(), time.perf_counter()
        while t - t0 < seconds:
            time.sleep(0.005)
            n = time.perf_counter()
            f = sys._current_frames().get(ident)
            if f is None:
                break
            key = " < ".join(f"{os.path.basename(x.filename)}:{x.lineno}" for x in reversed(traceback.extract_stack(f, limit=4)))
            tot[key] = tot.get(key, 0.0) + n - t
            t = n
            if _stop:
                break
        print(f"tick_trace agg total {(t - t0) * 1e3:.0f} ms", file=sys.stderr, flush=True)
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:25]:
            print(f"tick_trace agg {v * 1e3:6.0f} ms {k}", file=sys.stderr, flush=True)
        _busy = False

    threading.Thread(target=run, daemon=True, name="ticktrace").start()


def stop() -> None:
    global _stop
    _stop = True


def start(seconds: float = 4.0) -> None:
    ident = threading.get_ident()

    def run():
        t0 = t = time.perf_counter()
        cur, since = None, t0
        while t - t0 < seconds:
            time.sleep(0.01)
            t = time.perf_counter()
            f = sys._current_frames().get(ident)
            if f is None:
                return
            st = traceback.extract_stack(f, limit=6)
            key = " < ".join(f"{os.path.basename(x.filename)}:{x.lineno}:{x.name}" for x in reversed(st))
            if key != cur:
                if cur is not None and t - since >= 0.03:
                    print(f"tick_trace {(since - t0) * 1e3:6.0f} +{(t - since) * 1e3:5.0f} ms {cur}", file=sys.stderr, flush=True)
                cur, since = key, t
        print("tick_trace end", file=sys.stderr, flush=True)

    threading.Thread(target=run, daemon=True, name="ticktrace").start()

"""Did a lever run? One counter per lever site, off by default (KYOJIN_FIRED=1 or flags.set("fired", 1)).

A lever flag that changes nothing is indistinguishable from a lever that never ran. With `fired` on, each site that a lever can change bumps
HITS[name] where it takes the lever's path (and HITS["<name>_off"] style keys where the site was reached but a gate kept the old path), and
serve.server.engage() lists HITS under /v1/stats "engage" like every other module-level HITS dict. Off, `hit` is one global read and a return.

Sites inside an mx.compile region or a replay recording bump at trace / record time only (once per signature), not per token: a nonzero count there
proves the lever's branch was traced, the fuse_layer / verify_replay STATS call counters say how often the traced region then ran."""
from __future__ import annotations

import os

ON = os.environ.get("KYOJIN_FIRED", "0") not in ("0", "")
HITS: dict = {}


# ---- flag audit: ON BY DEFAULT (KYOJIN_FLAG_AUDIT=0 turns it off). One plain integer dict increment per site call, no sync, no array read; the reason strings are
# short and only built on the fallback branch. The counters are listed under /v1/stats "engage" ("fired": STATS) and one compact line per minute of activity
# goes to the served log (start_reporter), so a flag that silently falls back to stock shows in the log of every run.
AUDIT = os.environ.get("KYOJIN_FLAG_AUDIT", "1") not in ("0", "")
LOG = os.environ.get("KYOJIN_FIRED_LOG", "1") not in ("0", "")            # the first occurrence of each (flag, reason) pair is also logged at once
STATS: dict = {}
_seen: set = set()
_sink = None                                                               # log function of the server (set by start_reporter)


def hit(name: str, n: int = 1) -> None:
    if ON:
        HITS[name] = HITS.get(name, 0) + n


def fire(flag: str) -> None:
    """The fast path of `flag` ran (a python-side count: inside an mx.compile region / replay it counts at trace / record time, see the module doc)."""
    if AUDIT:
        k = flag + ".fire"
        STATS[k] = STATS.get(k, 0) + 1


def fb(flag: str, reason: str) -> None:
    """The site of `flag` was reached and a guard kept the stock path, for `reason` (one counter per flag x reason; the first occurrence of each pair is logged)."""
    if AUDIT:
        k = flag + ".fb." + reason
        n = STATS.get(k, 0)
        STATS[k] = n + 1
        if n == 0 and LOG:
            _say(f"[fired] {flag}: stock path, {reason}")


def _say(msg: str) -> None:
    if _sink is not None:
        _sink(msg)
    else:
        import sys
        print(msg, file=sys.stderr, flush=True)


def summary(prev: dict | None = None) -> str:
    """One compact line: per flag `fire/fallback` call counts (and the top fallback reasons), delta against `prev` when given (a snapshot dict)."""
    prev = prev or {}
    fl: dict = {}
    for k, v in list(STATS.items()):
        d = v - prev.get(k, 0)
        if d <= 0:
            continue
        flag, _, rest = k.partition(".")
        e = fl.setdefault(flag, [0, 0, {}])
        if rest == "fire":
            e[0] += d
        else:
            e[1] += d
            e[2][rest[3:]] = d
    out = []
    for flag in sorted(fl):
        f, b, rs = fl[flag]
        top = ",".join(f"{r}:{n}" for r, n in sorted(rs.items(), key=lambda t: -t[1])[:2])
        out.append(f"{flag} {f}/{b}" + (f" ({top})" if b else ""))
    return "; ".join(out)


def start_reporter(log, period: float = 60.0):
    """Daemon thread: every `period` s with counter activity, log one line `[flags 60s] flag fire/fallback (top reasons) ...` (fire = fast path ran, fallback = stock)."""
    global _sink
    import threading
    import time
    _sink = log
    if not AUDIT:
        return None

    def loop():
        prev = dict(STATS)
        while True:
            time.sleep(period)
            line = summary(prev)
            prev = dict(STATS)
            if line:
                try:
                    log("[flags %ds] %s" % (period, line))
                except Exception:
                    pass

    t = threading.Thread(target=loop, name="flag-audit", daemon=True)
    t.start()
    return t


def reset() -> None:
    HITS.clear()
    STATS.clear()
    _seen.clear()

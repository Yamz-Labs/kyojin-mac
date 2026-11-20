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


FIRES = os.environ.get("KYOJIN_HY4_FIRES_LOG", "1") not in ("0", "")      # lever totals appended to the periodic line (default on; 0 drops them)

# (label, module, dict, key): cumulative lever counters of already-imported modules (never imports one: a lane that did not load the module has no lever there)
_TOTALS = (("qdec_fast", "kyojin_mlx.foreign_metal", "PATHS", "qdec_fast"),
           ("absorb_gemv", "kyojin_mlx.foreign_skeleton_mla", "MlaAttention.PATHS", "absorb_gemv"),
           ("fixed_gather", "kyojin_mlx.foreign_skeleton_mla", "MlaAttention.PATHS", "fixed_gather"))


def totals() -> dict:
    """Cumulative path counters of the levers (python-side: inside an mx.compile region they count at trace time, so nonzero means the branch was taken)."""
    import sys
    out = {}
    for label, mod, attr, key in _TOTALS:
        m = sys.modules.get(mod)
        if m is None:
            continue
        d = m
        for a in attr.split("."):
            d = getattr(d, a, None)
        if isinstance(d, dict):
            out[label] = d.get(key, 0)
    return out


def format_line(period: float, line: str, tot: dict | None = None) -> str:
    """`[flags 60s] <fire/fallback per flag>[ | totals qdec_fast=N absorb_gemv=N fixed_gather=N]` (the totals part only when FIRES and a counter exists)."""
    out = "[flags %ds] %s" % (period, line)
    if FIRES and tot:
        out += " | totals " + " ".join(f"{k}={v}" for k, v in tot.items())
    return out


def start_reporter(log, period: float = 60.0):
    """Daemon thread: one line `[flags 60s] flag fire/fallback (top reasons) ... | totals qdec_fast=N ...` per `period` s with counter activity
    (fire = fast path ran, fallback = stock), plus one early line as soon as the counters of the first request have settled."""
    global _sink
    import threading
    import time
    _sink = log
    if not AUDIT:
        return None

    def snap():
        return dict(STATS), (totals() if FIRES else {})

    def loop():
        tick = min(2.0, period)
        prev, last = snap(), snap()
        due, first = time.time() + period, True
        while True:
            time.sleep(tick)
            cur = snap()
            settled = cur == last
            last = cur
            now = time.time()
            if not (now >= due or (first and settled and cur != prev)):
                continue
            line = summary(prev[0]) or ("-" if FIRES and cur[1] != prev[1] else "")
            prev, due = cur, now + period
            if line:
                first = False
                try:
                    log(format_line(period, line, cur[1]))
                except Exception:
                    pass

    t = threading.Thread(target=loop, name="flag-audit", daemon=True)
    t.start()
    return t


def reset() -> None:
    HITS.clear()
    STATS.clear()
    _seen.clear()

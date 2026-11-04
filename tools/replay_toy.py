"""Gate of kyojin_mlx.replay on toy functions (a fresh process with KYOJIN_REPLAY=1; tests/test_replay.py runs it).
  --mode toy      a compiled fixed-shape function with closure weights: 60 calls with new inputs, bitwise vs the plain call, CPU per call (build + async_eval)
  --mode shapes   the same region called with 3 different input shapes: one entry each, all bitwise, then back to the first shape
  --mode stateful a function whose graph depends on a Python counter (a constant array built per call): NOT a static region; the shadow check must catch it, mark the entry
                  dead and keep returning the eager results (every call equals the plain call)
  --mode alias    the same array passed twice for the first calls (rings restored from one prefix snapshot), then different arrays: a recording made while two inputs share
                  memory would bind both to one of them; it must wait for a call whose inputs are distinct, and every result equals the plain call
  --mode inflight 40 replays queued before the first evaluation (the ICB ring has to grow): every result bitwise vs the plain call
usage: KYOJIN_REPLAY=1 python tools/replay_toy.py [--mode toy]"""
import argparse, os, statistics, sys, time
os.environ.setdefault("KYOJIN_REPLAY", "1")
import mlx.core as mx
from kyojin_mlx import replay as RP

ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="toy", choices=("toy", "shapes", "stateful", "inflight", "alias")); a = ap.parse_args()
rng = mx.random
w1 = rng.normal((512, 512)).astype(mx.float16); w2 = rng.normal((512, 256)).astype(mx.float16); mx.eval(w1, w2)


def fn(x, p, y):
    h = x @ w1
    h = mx.softmax(h.astype(mx.float32), axis=-1) * p.astype(mx.float32)
    for _ in range(6):
        h = mx.tanh(h @ w1.astype(mx.float32)[:, :512]) + h * 0.5
    z = (h.astype(mx.float16) @ w2) + y
    return z, mx.argmax(h, axis=-1), mx.sum(h, axis=-1)[None]


def inputs(i, rows=4):
    x = rng.normal((rows, 512)).astype(mx.float16); p = mx.array(i + 1, dtype=mx.int32); y = rng.normal((rows, 256)).astype(mx.float16)
    mx.eval(x, p, y)
    return x, p, y


def check(r, e):
    return all(RP.same(a_, b_) for a_, b_ in zip(r, e))


bad = 0
if a.mode == "toy":
    cf = mx.compile(fn); f = RP.region(cf, "toy"); cpu_e, cpu_r = [], []
    for i in range(60):
        x, p, y = inputs(i)
        t0 = time.perf_counter(); r = f(x, p, y); mx.async_eval(*r); t1 = time.perf_counter(); mx.eval(*r)
        t2 = time.perf_counter(); e = cf(x, p, y); mx.async_eval(*e); t3 = time.perf_counter(); mx.eval(*e)
        ok = check(r, e); bad += not ok
        if i >= 6: cpu_r.append(t1 - t0); cpu_e.append(t3 - t2)
        if not ok: print("call", i, "DIFFERS")
    print(f"CPU per call (build + async_eval) us: plain min {1e6 * min(cpu_e):.0f} med {1e6 * statistics.median(cpu_e):.0f} | replay min {1e6 * min(cpu_r):.0f} med {1e6 * statistics.median(cpu_r):.0f}")
    ready = sum(v["state"] == "ready" for v in f.stats().values())
    assert ready == 1 and RP.STATS["replay"] >= 50, (RP.STATS, f.stats())
elif a.mode == "shapes":
    cf = mx.compile(fn); f = RP.region(cf, "shapes")
    for i in range(60):
        x, p, y = inputs(i, rows=(2, 4, 7)[i % 3] if i < 45 else 2)
        bad += not check(f(x, p, y), cf(x, p, y))
    st = f.stats()
    print("entries", {k[:30]: v["state"] for k, v in st.items()})
    assert len(st) == 3 and all(v["state"] == "ready" for v in st.values()), st
elif a.mode == "stateful":
    n = [0]

    def sfn(x):
        n[0] += 1
        return (x * mx.array(float(n[0]), dtype=mx.float32)).astype(mx.float16)             # a constant array written by the host each call: the stream is the same, the data is not

    f = RP.region(sfn, "stateful")
    for i in range(12):
        x = rng.normal((8, 64)).astype(mx.float16); mx.eval(x)
        r = f(x); n_before = n[0]
        bad += not RP.same(r, (x * mx.array(float(n_before), dtype=mx.float32)).astype(mx.float16))      # the result must be the one of the call that just ran (eager semantics)
    st = f.stats()
    print("states", [v["state"] for v in st.values()], RP.STATS)
    assert [v["state"] for v in st.values()] == ["dead"] and RP.STATS["mismatch"] >= 1, (st, RP.STATS)
elif a.mode == "alias":
    def afn(x, y):
        return x * 2.0 + y * 3.0, mx.sum(x - y)[None]
    cf = mx.compile(afn); f = RP.region(cf, "alias")
    for i in range(24):
        x = rng.normal((8, 64)).astype(mx.float32); mx.eval(x)
        y = x if i < 8 else rng.normal((8, 64)).astype(mx.float32)                        # aliasing for the first 8 calls, distinct afterwards
        mx.eval(y)
        bad += not check(f(x, y), cf(x, y))
    print("states", [v["state"] for v in f.stats().values()])
    assert [v["state"] for v in f.stats().values()] == ["ready"] and RP.STATS["mismatch"] == 0, (f.stats(), RP.STATS)
else:
    cf = mx.compile(fn); f = RP.region(cf, "inflight")
    for i in range(3):
        x, p, y = inputs(i); mx.eval(*f(x, p, y))
    ins = [inputs(100 + i) for i in range(40)]
    outs = [f(*t) for t in ins]
    mx.eval(*[o for r in outs for o in r])
    bad = sum(not check(r, cf(*t)) for r, t in zip(outs, ins))
    slots = [v["slots"] for v in f.stats().values()]
    print("ICB slots after 40 queued replays:", slots)
print(f"mode {a.mode}: mismatching calls {bad} | stats {RP.STATS}")
sys.exit(1 if bad else 0)

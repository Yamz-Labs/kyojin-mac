#!/usr/bin/env python3
"""Serve-loop bench: ms per host op and ms per engine step, old path against each kyojin_mlx.serve.hostloop lever (cloud/oploop-2).

  python tools/bench_serve_loop.py [--budget 45] [--tokens-per-round 4] [--gpu-ms 12] [--tokenizer PATH] [--json OUT]

What runs (all bounded: < 60 s with the default budget, < 2 GiB of mlx arrays, real DS4.1-flash geometry D 5120 / inter 2304 / 40 layers):

 1. ops     per-call cost of the host ops of one served token: detokenize per token vs one batch of n tokens, the SSE frame (json + encode),
            queue put/get (Queue vs SimpleQueue), `hostloop.mark` off / on, `gc.collect` of the model-sized heap. With --tokenizer PATH the
            decode is the real tokenizer's (transformers), else a byte decode (its cost is a floor: HF decode is 10-40 x dearer).
 2. gil     does a spinning Python thread run while the engine thread waits in mx.eval / .tolist()? (needs mlx; says whether the switch interval
            can matter at all while the GPU is busy)
 3. loop    an engine thread (graph build of 40 layers x ~38 nodes, async_eval, read of the step in flight, one `put` per token) with a real HTTP-thread
            analogue (hostloop.Reply + a TCP loopback socket and a reader thread), arms interleaved: base, each lever alone, all levers.
            Reported per arm: engine ms/step (median over rounds), p99 and max step, GC pause ms/step, HTTP-thread CPU ms/token, frames per token,
            client latency engine put -> byte on the client (fake tokenizer only), engine-side `t_hp_*` sections.
            Without mlx it falls back to a Python stand-in (build = allocations, GPU = a deadline slept on with the GIL released): only for the
            harness check, the numbers then say nothing about the Mac.

Every arm sets the module attributes of kyojin_mlx.serve.hostloop (what POST /admin/flags does); nothing is read from a pack. The bench prints; it
claims no gain: the reader of the table decides."""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import queue
import re
import socket
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mlx.core as mx
except Exception:                                                    # no mlx: the stand-in stepper
    mx = None

from kyojin_mlx.serve import hostloop as HL                          # noqa: E402

LAYERS, D, INTER, VOCAB = 40, 5120, 2304, 129280                     # DS4.1 flash geometry
NODES_PER_LAYER = 38                                                 # ~1500 graph nodes per step, the measured dispatch count of the served step

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=45.0, help="seconds for the loop arms (the other parts take ~10 s)")
ap.add_argument("--mode", choices=("plain", "spec"), default="plain",
                help="plain: the pipelined plain step (the next step is built while the one in flight runs). spec: a speculative round as served: build "
                     "(async_eval every 2 layers), blocking read, commit, no lookahead; host build time then adds 1:1 to the round")
ap.add_argument("--tokens-per-round", type=int, default=4, help="tokens the engine commits per step (1 = plain step, 4 = a speculative round)")
ap.add_argument("--gpu-ms", type=float, default=12.0, help="stand-in GPU time per step when mlx is missing")
ap.add_argument("--build-objs", type=int, default=20000, help="stand-in: Python objects allocated by one graph build when mlx is missing")
ap.add_argument("--heap-objects", type=int, default=400000, help="live Python objects of the 'loaded model' the collector has to walk")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--tokenizer", default=None, help="a tokenizer directory (transformers) for the decode cost; default: byte decode")
ap.add_argument("--json", default=None)
args = ap.parse_args()
T_START = time.perf_counter()

# ------------------------------------------------------------------------------------------------ tokenizer


class FakeApp:
    """id -> '<id>;' so a client can match text to token ids; decode is a join (the cost of a bare Python decode, not of HF)."""
    def decode(self, ids):
        return "".join(f"{i};" for i in ids)


class HFApp:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lock = threading.Lock()                                  # the server's tok_lock

    def decode(self, ids):
        with self.lock:
            return self.tok.decode(ids, skip_special_tokens=False)


APP = HFApp(args.tokenizer) if args.tokenizer else FakeApp()
REAL_TOK = args.tokenizer is not None


class Raw:
    def feed(self, text):
        return [("content", text)] if text else []

    def finish(self):
        return []


def frame(rid, text):
    obj = {"id": rid, "object": "chat.completion.chunk", "created": 1700000000, "model": "m",
           "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


def token_pool(n):
    if REAL_TOK:
        ids = APP.tok.encode(("The history of the Roman Empire begins with the founding of Rome. def f(x): return x + 1\n" * 200))
        return [ids[i % len(ids)] for i in range(n)]
    return list(range(1, n + 1))

# ------------------------------------------------------------------------------------------------ 1. host ops


def per_call_us(fn, n):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


def ops():
    out = {}
    toks = token_pool(4096)
    for n in (1, 4, 8):
        def per_token(n=n):
            d = HL.Detok(APP)
            for i in range(n):
                d.push(toks[i])
        def batch(n=n):
            d = HL.Detok(APP)
            d.push_many(toks[:n])
        HL.VERIFY_LEFT, HL.BATCH_BAD = 0, False
        reps = 3000 if not REAL_TOK else 500
        out[f"detok_{n}tok_pertoken_us"] = per_call_us(per_token, reps)
        out[f"detok_{n}tok_batch_us"] = per_call_us(batch, reps)
    out["sse_frame_us"] = per_call_us(lambda: frame("chatcmpl-abc123", "hello world;"), 20000)
    for name, q in (("Queue", queue.Queue()), ("SimpleQueue", queue.SimpleQueue())):
        def pg(q=q):
            q.put(("tok", 1))
            q.get()
        out[f"put_get_{name}_us"] = per_call_us(pg, 50000)
    HL.PROF = False
    out["mark_off_us"] = per_call_us(lambda: HL.mark("x"), 200000)
    st = collections.Counter()
    HL.PROF = True
    HL.bind(st)
    HL.reset()
    out["mark_on_us"] = per_call_us(lambda: HL.mark("x"), 100000)
    HL.PROF = False
    HL.bind(st)
    return out

# ------------------------------------------------------------------------------------------------ 2. GIL while the engine waits


def gil_probe():
    if mx is None:
        return {"gil_released_in_eval": "n/a (no mlx)"}
    cnt = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            cnt[0] += 1
    a = mx.random.normal((8192, 8192)).astype(mx.float16)
    mx.eval(a)

    def heavy():
        y = a
        for _ in range(24):
            y = (y @ a) * 0.001
        return y
    mx.eval(heavy())
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(0.05)
    c0, t0 = cnt[0], time.perf_counter()
    time.sleep(0.2)
    idle = (cnt[0] - c0) / (time.perf_counter() - t0)
    y = heavy()
    mx.async_eval(y)
    c0, t0 = cnt[0], time.perf_counter()
    y.tolist() if y.size < 100 else mx.eval(y)
    dt_eval = time.perf_counter() - t0
    in_eval = (cnt[0] - c0) / dt_eval
    z = mx.sum(heavy())
    mx.async_eval(z)
    c0, t0 = cnt[0], time.perf_counter()
    z.tolist()
    dt_tl = time.perf_counter() - t0
    in_tl = (cnt[0] - c0) / dt_tl
    stop.set()
    del a, y, z
    return {"spinner_iters_per_s_idle": round(idle), "spinner_iters_per_s_in_eval": round(in_eval), "eval_wait_ms": round(dt_eval * 1e3, 1),
            "spinner_iters_per_s_in_tolist": round(in_tl), "tolist_wait_ms": round(dt_tl * 1e3, 1),
            "gil_released_in_eval": in_eval > 0.3 * idle, "gil_released_in_tolist": in_tl > 0.3 * idle}

# ------------------------------------------------------------------------------------------------ 3. loop

_HEAP = None


def make_heap(n):
    """n live container objects with cycles: what the collector walks in a process that holds a loaded model (module tree, caches, closures)."""
    global _HEAP
    heap = []
    for i in range(n // 4):
        a = {"i": i, "w": [i, i + 1], "t": (i, str(i))}
        b = {"a": a}
        a["b"] = b
        heap.append(a)
    _HEAP = heap


class Stepper:
    """One decode step as the engine launches it: a lazy graph of LAYERS x ~38 nodes (two big GEMVs per layer), an argmax -> token."""

    def __init__(self):
        if mx is None:
            self.fin = 0.0
            self.keep = []
            return
        nbytes = LAYERS * 2 * D * INTER * 2
        assert nbytes < 3.5 * 2 ** 30, "bank over the 3.5 GiB bound"
        mx.random.seed(0)
        self.w = []
        for _ in range(LAYERS):
            a = (mx.random.normal((D, INTER)) * 0.02).astype(mx.float16)
            b = (mx.random.normal((INTER, D)) * 0.02).astype(mx.float16)
            mx.eval(a, b)
            self.w.append((a, b))
        self.x0 = mx.random.normal((1, D)).astype(mx.float16)
        mx.eval(self.x0)

    def launch(self, state, every=0):
        """-> (state, lazy token id). state = the lazy hidden state of the step before (None at the start). every: async_eval each n layers (spec mode)."""
        if mx is None:
            t = time.perf_counter()
            objs = [(i, str(i), [i]) for i in range(args.build_objs)]             # a graph build's worth of short-lived containers
            self.keep = (self.keep + objs[:200])[-2000:]                          # some survive (the lazy graph is alive until it ran)
            start = max(self.fin, time.perf_counter())
            self.fin = start + args.gpu_ms / 1e3
            return self.fin, self.fin
        x = self.x0 if state is None else state
        for li, (a, b) in enumerate(self.w):
            h = x @ a
            for _ in range((NODES_PER_LAYER - 4) // 2):
                h = mx.sigmoid(h) * h * 1.0001
            x = x + (h @ b) * 0.01
            x = x * 0.999
            if every and (li + 1) % every == 0:
                mx.async_eval(x)
        tok = mx.argmax(x, axis=-1)
        mx.async_eval(tok)
        return x, tok

    @staticmethod
    def read(tok):
        if mx is None:
            d = tok - time.perf_counter()
            if d > 0:
                time.sleep(d)                                   # the GIL is released while the GPU runs, like mx.eval
            return 0
        return int(tok.tolist()[0])


class Client:
    """A loopback TCP client that timestamps every byte batch it reads (the SSE reader)."""

    def __init__(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        self.cli = socket.create_connection(srv.getsockname())
        self.sock, _ = srv.accept()
        srv.close()
        if HL.NODELAY:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.arrive = {}
        self.frames = 0
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        pat = re.compile(rb"(\d+);")
        while True:
            try:
                b = self.cli.recv(65536)
            except OSError:
                return
            if not b:
                return
            now = time.perf_counter()
            self.frames += b.count(b"\n\n")
            if not REAL_TOK:
                for m in pat.finditer(b):
                    self.arrive.setdefault(int(m.group(1)), now)

    def close(self):
        self.sock.close()
        self.t.join(timeout=2)
        self.cli.close()


def consumer(q, client, cpu, sent):
    """The HTTP thread of one streamed request (server._completion): get, [drain], detok, frames, write."""
    reply = HL.Reply(APP, Raw(), [])
    rid = "chatcmpl-abc123"
    sock = client.sock
    n = 0
    t0 = time.thread_time()
    while True:
        ev = q.get()
        if ev[0] == "done":
            break
        if HL.COALESCE:
            toks, tail = HL.drain(q, ev, queue.Empty)
            events = reply.push_many(toks)
            data = b"".join(frame(rid, e[1]) for e in events)
            if data:
                sock.sendall(data)
            n += len(toks)
            if tail is not None:
                break
        else:
            for e in reply.push(ev[1]):
                sock.sendall(frame(rid, e[1]))
            n += 1
    cpu.append((time.thread_time() - t0, n))


def run_arm(cfg, steps, prof=False):
    """One engine run of `steps` steps with the lever set `cfg`; returns a dict of metrics."""
    HL.COALESCE, HL.NODELAY, HL.GC, HL.SWITCH_US = cfg.get("coalesce", False), cfg.get("nodelay", False), cfg.get("gc", False), cfg.get("switch", 0)
    HL.PROF = prof                                                   # timing arms run with the profiler OFF: its lag sleeper would perturb them
    HL.LAG_MS = 10.0
    HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False
    stats = collections.Counter()
    HL.bind(stats)
    HL.gc_sync()
    HL.switch_sync()
    gc.collect()
    pauses = []
    t_gc = [0.0]

    def cb(phase, info):
        if phase == "start":
            t_gc[0] = time.perf_counter()
        else:
            pauses.append(time.perf_counter() - t_gc[0])
    gc.callbacks.append(cb)
    client = Client()
    q = HL.new_queue()
    cpu, put_t = [], {}
    th = threading.Thread(target=consumer, args=(q, client, cpu, None), daemon=True)
    th.start()
    stepper = Stepper()
    ntok = args.tokens_per_round
    pool = iter(range(1, 10 ** 9))
    step_ms = []
    spec = args.mode == "spec"
    state, tok = (None, None) if spec else stepper.launch(None)
    HL.reset()
    t_all = time.perf_counter()
    for i in range(steps):
        t0 = time.perf_counter()
        if spec:                                                     # build, wait, commit; nothing in flight while the host works
            state, tok = stepper.launch(state, every=2)
            HL.mark("build")
            stepper.read(tok)
            HL.mark("wait")
            nstate, ntok_lazy = state, tok
        else:
            nstate, ntok_lazy = stepper.launch(state)                # the next step is built while the one in flight runs
            stepper.read(tok)
        now = time.perf_counter()
        for _ in range(ntok):                                        # the commit: one put per token
            t = next(pool)
            put_t[t] = time.perf_counter()
            q.put(("tok", t))
        HL.mark("commit")
        state, tok = nstate, ntok_lazy
        step_ms.append((time.perf_counter() - t0) * 1e3)
        HL.mark("tick")
    wall = time.perf_counter() - t_all
    if not spec:
        stepper.read(tok)
    q.put(("done", "stop", {}))
    th.join(timeout=10)
    time.sleep(0.05)
    lat = [(client.arrive[t] - put_t[t]) * 1e3 for t in put_t if t in client.arrive]
    n_tok = steps * ntok
    gc.callbacks.remove(cb)
    frames = client.frames
    client.close()
    HL.PROF = False
    HL.bind(stats)
    HL.GC, HL.SWITCH_US = False, 0
    HL.gc_sync()
    HL.switch_sync()
    srt = sorted(step_ms)
    return {"step_ms": wall / steps * 1e3, "p99_ms": srt[int(0.99 * (len(srt) - 1))], "max_ms": srt[-1],
            "gc_ms_per_step": sum(pauses) / steps * 1e3, "gc_n": len(pauses), "gc_max_ms": max(pauses, default=0.0) * 1e3,
            "http_cpu_ms_per_tok": (cpu[0][0] / max(1, cpu[0][1]) * 1e3) if cpu else float("nan"),
            "frames_per_tok": frames / n_tok, "lat_mean_ms": statistics.mean(lat) if lat else float("nan"),
            "lat_p99_ms": sorted(lat)[int(0.99 * (len(lat) - 1))] if lat else float("nan"),
            "sec": {k[5:]: [round(v / max(1, steps) * 1e3, 3), round(stats["t_hpc_" + k[5:]] / max(1, steps) * 1e3, 3)]
                    for k, v in stats.items() if k.startswith("t_hp_") and not k.endswith("giltick")},
            "gil_lag_ms": (stats["t_hp_giltick"] / max(1, stats["n_hp_giltick"]) * 1e3), "gil_over_2ms": stats["n_hp_gil2ms"]}


ARMS = [("base", {}), ("coalesce", {"coalesce": True}), ("nodelay", {"nodelay": True}), ("gc", {"gc": True}),
        ("switch500us", {"switch": 500}), ("switch50ms", {"switch": 50000}), ("all", {"coalesce": True, "nodelay": True, "gc": True, "switch": 500})]            # (switch50ms: the other direction, the HTTP thread runs mostly while the engine waits)


def loop_arms():
    make_heap(args.heap_objects)
    ref = run_arm({}, 20)
    per_run = max(ref["step_ms"], 1.0) * 1e-3
    steps = int(max(30, min(400, args.budget / (len(ARMS) * args.rounds * per_run * 1.15))))
    res = collections.defaultdict(list)
    for r in range(args.rounds):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]          # rotate the order: no arm always runs first
        for name, cfg in order:
            res[name].append(run_arm(cfg, steps))
    return steps, res


def med(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    return statistics.median(v) if v else float("nan")


def main():
    out = {"mlx": mx is not None, "tokenizer": args.tokenizer or "byte-decode", "tokens_per_round": args.tokens_per_round}
    print(f"# python {sys.version.split()[0]}  gc thresholds {gc.get_threshold()}  switch interval {sys.getswitchinterval() * 1e6:.0f} us  mode={args.mode}")
    print(f"# bench_serve_loop  mlx={'yes' if mx else 'NO (stand-in stepper: harness check only)'}  tokenizer={out['tokenizer']}  tokens/step={args.tokens_per_round}")
    o = ops()
    out["ops_us"] = o
    print("\n== host ops (us per call)")
    for k, v in o.items():
        print(f"  {k:32s} {v:9.2f}")
    g = gil_probe()
    out["gil"] = g
    print("\n== GIL while the engine waits on the GPU")
    for k, v in g.items():
        print(f"  {k:32s} {v}")
    steps, res = loop_arms()
    out["steps_per_run"] = steps
    prof = run_arm({}, steps, prof=True)                             # one separate pass with the profiler on: where the host time of a step goes
    print(f"\n== engine loop, {steps} steps/run, {args.rounds} rounds interleaved (medians); engine ms/step is wall per step incl. the GPU wait")
    hdr = f"  {'arm':12s} {'ms/step':>8s} {'d vs base':>9s} {'p99':>7s} {'max':>7s} {'gc ms/step':>10s} {'gc n':>5s} {'gc max':>7s} {'http cpu ms/tok':>15s} {'frames/tok':>10s} {'lat mean':>8s} {'lat p99':>8s} {'gil lag ms':>10s}"
    print(hdr)
    base = med(res["base"], "step_ms")
    out["arms"] = {}
    for name, _ in ARMS:
        rows = res[name]
        m = {k: med(rows, k) for k in ("step_ms", "p99_ms", "max_ms", "gc_ms_per_step", "gc_n", "gc_max_ms", "http_cpu_ms_per_tok", "frames_per_tok",
                                       "lat_mean_ms", "lat_p99_ms", "gil_lag_ms")}
        m["spread_ms"] = max(r["step_ms"] for r in rows) - min(r["step_ms"] for r in rows)
        m["sections_ms_per_step"] = rows[-1]["sec"]
        out["arms"][name] = m
        print(f"  {name:12s} {m['step_ms']:8.3f} {m['step_ms'] - base:+9.3f} {m['p99_ms']:7.2f} {m['max_ms']:7.2f} {m['gc_ms_per_step']:10.4f} {m['gc_n']:5.0f} {m['gc_max_ms']:7.2f} "
              f"{m['http_cpu_ms_per_tok']:15.4f} {m['frames_per_tok']:10.3f} {m['lat_mean_ms']:8.3f} {m['lat_p99_ms']:8.3f} {m['gil_lag_ms']:10.4f}")
    print("\n  (d vs base inside the round-to-round spread is noise; spread per arm:",
          ", ".join(f"{n} {out['arms'][n]['spread_ms']:.3f}" for n, _ in ARMS) + " ms)")
    out["sections_prof_pass"] = prof["sec"]
    print("\n== engine-thread sections, profiler-on pass (base flags): ms per step as [wall, thread cpu]; wall ~ cpu = host work, wall >> cpu = a wait")
    for k, v in prof["sec"].items():
        print(f"  {k:10s} wall {v[0]:8.3f}  cpu {v[1]:8.3f}")
    print(f"  gc in that pass: {prof['gc_ms_per_step']:.4f} ms/step, {prof['gc_n']} collections, longest {prof['gc_max_ms']:.2f} ms; GIL lag mean {prof['gil_lag_ms']:.3f} ms, "
          f"{prof['gil_over_2ms']} wake-ups over 2 ms")
    if mx is not None:
        print(f"  mlx peak memory {mx.get_peak_memory() / 2 ** 30:.2f} GiB (bound 3.5)")
    print(f"  total {time.perf_counter() - T_START:.1f} s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()

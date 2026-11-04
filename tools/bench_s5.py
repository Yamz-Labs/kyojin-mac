#!/usr/bin/env python3
"""S5 end-to-end decode A/B on the Mac: S5 off vs on, interleaved in ONE process (other jobs share the GPU), greedy, mlx-lm style async decode.
usage: bench_s5.py MODEL_DIR [--tokens 96] [--rounds 8] [--json OUT.json]
Per run: fresh cache, 128-token prefill, 8 warm-up tokens, then D timed tokens. Reported per variant (median / min over rounds):
  tok/s       wall clock of the async loop
  build ms    CPU time to build the lazy graph of one token (python + C++ op creation)
  submit ms   CPU time of mx.async_eval() (graph walk + command encoding + commit)
  cpu ms      build + submit (wall clock of the two calls: submit also waits when the GPU queue is full)
  main_cpu / proc_cpu  thread CPU time of the python thread / user+sys CPU time of the whole process (incl. MLX's encode thread) per token: no
              GPU waiting in these, so they predict the token time of a machine whose GPU is fast enough to never be the bottleneck
Both variants must emit the same greedy tokens (S5 is bit-exact)."""
import argparse
import json
import os
import resource
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

from kyojin_mlx import fused_group as FG
from kyojin_mlx.exl3_linear import load_exl3

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("--tokens", type=int, default=96); ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--json", default=None)
ap.add_argument("--cpu-steps", type=int, default=400, help="steps of the isolated CPU measure (sync decode, S5 off / on alternating step by step)")
a = ap.parse_args()
model, _ = load_exl3(a.model_dir)
info = FG.install(model)
print(f"S5 install: {info['groups']} groups, {info['launches_saved']} launches saved", flush=True)
ids = AutoTokenizer.from_pretrained(a.model_dir).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 40)[:128]
prompt = mx.array(ids)[None]


def step(y, cache):
    return mx.argmax(model(y[:, None], cache=cache)[:, -1], axis=-1)


def run(enabled: bool):
    FG.ENABLED = enabled
    cache = make_prompt_cache(model)
    y = mx.argmax(model(prompt, cache=cache)[:, -1], axis=-1)
    mx.eval(y)
    for _ in range(8):
        y = step(y, cache); mx.eval(y)
    toks, tb, ts = [], 0.0, 0.0
    y = step(y, cache); mx.async_eval(y)
    ru0, tt0 = resource.getrusage(resource.RUSAGE_SELF), time.thread_time()
    t0 = time.perf_counter()
    for _ in range(a.tokens):
        b0 = time.perf_counter(); ny = step(y, cache); b1 = time.perf_counter(); mx.async_eval(ny); b2 = time.perf_counter()
        toks.append(y.item()); y = ny
        tb += b1 - b0; ts += b2 - b1
    dt = time.perf_counter() - t0
    ru1, tt1 = resource.getrusage(resource.RUSAGE_SELF), time.thread_time()
    proc = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    FG.ENABLED = True
    return dict(tps=a.tokens / dt, build_ms=tb / a.tokens * 1e3, submit_ms=ts / a.tokens * 1e3, main_cpu_ms=(tt1 - tt0) / a.tokens * 1e3,
                proc_cpu_ms=proc / a.tokens * 1e3, tokens=toks)


res = {"off": [], "on": []}
order = ["off", "on", "on", "off"]
for rd in range(a.rounds):
    for v in (order[:2] if rd % 2 == 0 else order[2:]):
        r = run(v == "on")
        res[v].append(r)
        print(f"round {rd} {v:3s}: {r['tps']:6.1f} tok/s  build {r['build_ms']:.2f} ms  submit {r['submit_ms']:.2f} ms  cpu: main thread {r['main_cpu_ms']:.2f} ms, process {r['proc_cpu_ms']:.2f} ms", flush=True)

def cpu_isolated(n):
    """Sync decode, variant alternating every step on the same cache (S5 output is bit-exact so the sequence is unchanged): the GPU is idle when a
    step is submitted, so thread CPU time of async_eval = graph walk + command encoding with no GPU back-pressure."""
    cache = make_prompt_cache(model)
    y = mx.argmax(model(prompt, cache=cache)[:, -1], axis=-1)
    mx.eval(y)
    for _ in range(6):
        y = step(y, cache); mx.eval(y)
    rec = {False: ([], [], []), True: ([], [], [])}
    for i in range(n):
        en = bool(i % 2)
        FG.ENABLED = en
        b0 = time.perf_counter(); c0 = time.thread_time(); ny = step(y, cache); c1 = time.thread_time(); b1 = time.perf_counter()
        e0 = time.thread_time(); mx.async_eval(ny); e1 = time.thread_time()
        t0 = time.perf_counter(); mx.eval(ny); t1 = time.perf_counter()
        y = ny
        if i >= 8:
            rec[en][0].append((b1 - b0) * 1e3); rec[en][1].append((e1 - e0) * 1e3); rec[en][2].append((e1 - e0 + b1 - b0) * 1e3 + 0 * (t1 - t0))
    FG.ENABLED = True
    return {("on" if k else "off"): dict(build_ms=statistics.median(v[0]), encode_cpu_ms=statistics.median(v[1]), cpu_ms=statistics.median(v[2])) for k, v in rec.items()}


iso = cpu_isolated(a.cpu_steps) if a.cpu_steps else None
if iso:
    for v in ("off", "on"):
        print(f"isolated CPU, {v:3s}: build {iso[v]['build_ms']:.3f} ms + async_eval encode {iso[v]['encode_cpu_ms']:.3f} ms = {iso[v]['cpu_ms']:.3f} ms/token "
              f"(token-rate ceiling when CPU-bound {1000 / iso[v]['cpu_ms']:.0f} tok/s)")
    print(f"isolated CPU on vs off: build {100 * (iso['on']['build_ms'] / iso['off']['build_ms'] - 1):+.1f} %, encode {100 * (iso['on']['encode_cpu_ms'] / iso['off']['encode_cpu_ms'] - 1):+.1f} %, "
          f"total {100 * (iso['on']['cpu_ms'] / iso['off']['cpu_ms'] - 1):+.1f} %  ({iso['on']['cpu_ms'] - iso['off']['cpu_ms']:+.3f} ms/token)")
same = all(r["tokens"] == res["off"][0]["tokens"] for v in res for r in res[v])
print("greedy tokens identical across all runs (off and on):", same)
out = {}
for v in ("off", "on"):
    row = {k: (statistics.median(r[k] for r in res[v]), min(r[k] for r in res[v]), max(r[k] for r in res[v])) for k in ("tps", "build_ms", "submit_ms", "main_cpu_ms", "proc_cpu_ms")}
    cpu = [r["build_ms"] + r["submit_ms"] for r in res[v]]
    row["cpu_ms"] = (statistics.median(cpu), min(cpu), max(cpu))
    out[v] = row
    print(f"{v:3s}: tok/s median {row['tps'][0]:.1f} (best {row['tps'][2]:.1f})  build {row['build_ms'][0]:.2f}  submit {row['submit_ms'][0]:.2f}  cpu {row['cpu_ms'][0]:.2f} ms/token "
          f"(wall-clock ceiling {1000 / row['cpu_ms'][0]:.0f} tok/s)  CPU-seconds per token: main thread {row['main_cpu_ms'][0]:.2f} ms, whole process {row['proc_cpu_ms'][0]:.2f} ms")
d = lambda k, i=0: out["on"][k][i] / out["off"][k][i] - 1
print(f"on vs off: tok/s {100 * d('tps'):+.1f} % (best {100 * d('tps', 2):+.1f} %), build {100 * d('build_ms'):+.1f} %, submit {100 * d('submit_ms'):+.1f} %, "
      f"main-thread CPU {100 * d('main_cpu_ms'):+.1f} %, process CPU {100 * d('proc_cpu_ms'):+.1f} %")
if a.json:
    json.dump(dict(install=info, same_tokens=same, summary=out, isolated_cpu=iso, runs={v: [{k: x for k, x in r.items() if k != 'tokens'} for r in res[v]] for v in res}), open(a.json, "w"), indent=1)
sys.exit(0 if same else 1)

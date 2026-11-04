"""Serving engine without HTTP: staggered joins, early leaves, prefix hits and sampled sessions against the same requests run alone.
usage: serve_inproc.py MODEL_DIR [--sessions 4] [--chunk 256] [--block 512] [--json OUT]
"Alone" = one request at a time through the same engine code (sessions=1, no prefix cache). Every churn run must give the same token ids
as the solo run of the same request: greedy and seeded sampling. Prints the largest decode-tick gap, which a prefill chunk of a long prompt
makes, and the prefix hits."""
import argparse, json, threading, time, zlib

import mlx.core as mx
import numpy as np

from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--sessions", type=int, default=4)
ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--block", type=int, default=512)
ap.add_argument("--scale", type=float, default=1.0, help="prompt length scale (smaller = faster run)")
ap.add_argument("--json")
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
backend, cfg = load_backend(a.model)
vocab = (cfg.get("text_config") or cfg)["vocab_size"]
if backend.kind == "q35":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    EOS = {cfg["text_config"]["eos_token_id"], tok.convert_tokens_to_ids("<|im_end|>")}
else:                                                                  # DS4.1 / GLM slice packs: no real text, random ids, no stop token
    tok, EOS = None, set()

topics = ["The history of the Roman Empire begins with the founding of Rome in 753 BC and the long struggle between patricians and plebeians. ",
          "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))\n\n",
          "Le petit village de montagne se reveillait doucement sous la brume, tandis que les cloches de l'eglise sonnaient sept heures du matin. ",
          "Photosynthesis converts light energy into chemical energy stored in glucose, using carbon dioxide and water, and releases oxygen. "]


def text_ids(i, n):
    if tok is None:
        return np.random.default_rng(100 + i % 4).integers(16, vocab - 16, int(n * a.scale)).tolist()
    return tok.encode(topics[i % 4] * 400)[:int(n * a.scale)]


def tail(text):
    if tok is None:
        return np.random.default_rng(zlib.crc32(text.encode())).integers(16, vocab - 16, 8).tolist()
    return tok.encode(text)


shared = text_ids(0, 1100)
reqs = {
    "A": dict(ids=text_ids(1, 1500), n=60, at=0.0),
    "B": dict(ids=text_ids(2, 600), n=12, at=0.0),                                       # leaves early
    "C": dict(ids=shared + tail(" Task one: summarise the above in detail."), n=40, at=0.4),   # joins while A, B decode
    "D": dict(ids=text_ids(3, 2300), n=30, at=0.5),                                      # long prompt: chunked prefill under the others
    "E": dict(ids=shared + tail(" Task two: list the main causes, with dates."), n=40, at=0.6, samp=Sampling(0.8, 0.9, 40, 0.02, 123)),   # prefix hit, sampled
    "F": dict(ids=text_ids(2, 400), n=40, at=1.5, samp=Sampling(0.7, 1.0, 0, 0.0, 7)),   # sampled, joins late
    "G": dict(ids=shared + tail(" Task one: summarise the above in detail."), n=20, at=2.0),   # same prompt as C: hit at the last block
}


def run(sessions, prefix_bytes, delays=True):
    eng = Engine(backend, sessions=sessions, ctx=32768, chunk=a.chunk, block=a.block, prefix_bytes=prefix_bytes, log=print)
    out, handles, t0 = {}, {}, time.perf_counter()
    done = {}
    result = {}

    def worker(name, r):
        if delays:
            time.sleep(r["at"])
        t_sub = time.perf_counter() - t0
        h = eng.submit(r["ids"], max_new=r["n"], sampling=r.get("samp"), stop_ids=EOS)
        toks, times = [], []
        while True:
            ev = h.events.get()
            if ev[0] == "tok":
                toks.append(ev[1]); times.append(time.perf_counter() - t0)
            else:
                done[name] = dict(tokens=toks, reason=ev[1], stats=ev[2], times=times, t_sub=t_sub)
                return

    def clients():
        if not delays:                                                 # solo: one after the other
            for k, r in reqs.items():
                th = threading.Thread(target=worker, args=(k, r)); th.start(); th.join()
        else:
            ths = [threading.Thread(target=worker, args=(k, r)) for k, r in reqs.items()]
            for th in ths: th.start()
            for th in ths: th.join()
        result["st"], result["gap"] = eng.status(), eng.max_step_gap
        eng.stop()
    driver = threading.Thread(target=clients); driver.start()
    eng.run()                                                          # the engine runs on this thread, the one that loaded the model
    driver.join()
    return done, result["st"], result["gap"]


print("solo runs ...", flush=True)
solo, st0, _ = run(1, 0, delays=False)
for k, d in solo.items():
    print(f"  solo {k}: {len(reqs[k]['ids'])} prompt, {len(d['tokens'])} tokens, {d['reason']}, ttft {d['stats']['ttft_s']:.2f}s")
print(f"churn run (sessions={a.sessions}) ...", flush=True)
mx.reset_peak_memory()
churn, st, gap = run(a.sessions, 1 << 30)
ok = True
rep = {}
for k in reqs:
    same = churn[k]["tokens"] == solo[k]["tokens"]
    ok &= same
    c = churn[k]
    gaps = [b - a_ for a_, b in zip(c["times"], c["times"][1:])]
    rep[k] = dict(same=same, n=len(c["tokens"]), cached=c["stats"]["cached_tokens"], ttft=c["stats"]["ttft_s"], queue=c["stats"]["queue_s"], max_gap=max(gaps) if gaps else 0)
    print(f"  {k}: tokens equal to solo {same}  n={rep[k]['n']} prompt={c['stats']['prompt_tokens']} cached={rep[k]['cached']} ttft={rep[k]['ttft']:.2f}s "
          f"queue={rep[k]['queue']:.2f}s max_inter_token_gap={rep[k]['max_gap'] * 1e3:.0f}ms reason={c['reason']}")
print("engine status:", st)
print(f"largest decode tick {gap * 1e3:.0f} ms; peak memory {mx.get_peak_memory() / 2 ** 30:.2f} GB")
print("ALL EQUAL TO SOLO" if ok else "MISMATCH")
if a.json:
    json.dump(dict(ok=ok, rep=rep, status=st, max_tick=gap), open(a.json, "w"), indent=1, default=str)
raise SystemExit(0 if ok else 1)

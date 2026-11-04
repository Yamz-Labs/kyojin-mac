#!/usr/bin/env python3
"""DS4.1 bar: our served engine against the TensorFold e2e3-p4-h8 numbers (M5 Ultra, self-reported), in the same harness shape.

Everything runs through the serving engine in this process (kyojin_mlx.serve.engine.Engine, no HTTP): continuous batching, chunked
prefill between decode ticks, MTP/DSpark speculative verify of every session in one pass. Greedy. Modes (--modes, comma list):
  solo   one stream per prompt class (prose / code / structured / list), --n tokens (256), EOS ignored: decode tok/s and TTFT per class
  nK     K streams at once (n2, n4) per class, K copies of the same class prompt (TensorFold tools/bench_concurrent.py): aggregate tok/s =
         sum(tokens - 1) / (last token of any stream - first token of any stream), as theirs; plus per-stream tok/s and the wall time
  pf     cold prefill of a --pf length prompt (2K / 8K / 33K / 131K), max_new 1, distinct first token (no prefix hit): prompt tokens / TTFT
  1m     the same at --long tokens (1M on the Ultra): tok/s, minutes, peak memory, and the first token after the prompt
tok/s of a stream = (tokens - 1) / (t_last - t_first): the decode rate after the first token, TTFT apart (TensorFold tools/bench_openai.py).
Prompt ids: the DS4.1 chat template (thinking off) through the pack tokenizer (--tok, default PACK). A slice pack with a cut vocabulary
(vocab < tokenizer) folds the ids into its vocabulary: the numbers there are plumbing, not speed.
usage: ds41_bar.py PACK [--tok DIR] [--modes solo,n2,n4,pf,1m] [--n 256] [--depth 3] [--pf 2048,8192,32768,131072] [--long 1048576]
                        [--chunk 4096] [--ctx C] [--json F]"""
import argparse, json, os, sys, threading, time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx.serve.backends import load_backend  # noqa: E402
from kyojin_mlx.serve.engine import Engine  # noqa: E402
from kyojin_mlx.serve.ds41_chat import render_chat  # noqa: E402

# TensorFold e2e3-p4-h8 on the M5 Ultra (HF BobClawblaw/DeepSeek-V4.1-Flash-1017-TensorFold-e2e3-p4-h8, ~/leadm/INBOX.md 28/10 10:00)
THEIRS = {"solo": dict(prose=70.4, code=90.5, structured=76.8, list=88.6), "ttft_s": (0.11, 0.14),
          "n4": dict(prose=93.5, code=101.4, structured=89.8, list=102.1), "n2": (82.0, 99.0),
          "pf": {2048: 1093.0, 8192: 1240.0, 32768: 1256.0, 131072: 1195.0}, "1m": 565.0}
PROMPTS = {                                   # code / prose / list: TensorFold tools/zig/xpu_draft_receipt.py (72a552e); structured: ours (none there)
    "prose": "Describe a quiet harbour town at dawn in about two hundred words.",
    "code": "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and tests.",
    "structured": "Return a JSON object describing three fictional employees with the fields name, age, department, skills (a list) and salary.",
    "list": "List the planets of the solar system in order, one per line, with one fact about each.",
}
FILL = ("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. "
        "Trade routes carried grain, oil and wine across the Mediterranean, and the roads tied the provinces to the capital. ")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pack")
ap.add_argument("--tok", help="tokenizer dir (tokenizer.json); default: the pack")
ap.add_argument("--modes", default="solo,n2,n4,pf")
ap.add_argument("--classes", default=",".join(PROMPTS))
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--depth", type=int, default=3, help="MTP / DSpark depth (0 = plain decode)")
ap.add_argument("--pf", default="2048,8192,32768,131072")
ap.add_argument("--long", type=int, default=1048576)
ap.add_argument("--long-n", type=int, default=8, help="tokens decoded after the --long prompt (8 = TTFT only; more to see the steady decode rate and the drafter at depth)")
ap.add_argument("--chunk", type=int, default=int(os.environ.get("KYOJIN_PF_CHUNK", "4096")), help="served prefill chunk (tokens)")
ap.add_argument("--ctx", type=int, default=0, help="engine context (0 = the longest prompt of the modes + n + 64)")
ap.add_argument("--prefix-bytes", type=int, default=0, help="engine prefix snapshot budget (0: no snapshot copies inside the cold prefill timing)")
ap.add_argument("--ced", action="store_true", help="CED prefill (default: the exact prefill)")
ap.add_argument("--reps", type=int, default=1, help="solo / nK: runs per cell, interleaved over the classes; best and median reported")
ap.add_argument("--json")
a = ap.parse_args()
modes = [m for m in a.modes.split(",") if m]
classes = [c for c in a.classes.split(",") if c]
pf_lens = [int(x) for x in a.pf.split(",") if x] if "pf" in modes else []
streams = sorted({int(m[1:]) for m in modes if m[:1] == "n" and m[1:].isdigit()})
GiB = 2 ** 30

t0 = time.time()
backend, cfg = load_backend(a.pack, mtp=a.depth > 0, ced=a.ced, log=lambda *x: None)
vocab = cfg.get("text_config", cfg)["vocab_size"]
max_pos = cfg.get("text_config", cfg).get("max_position_embeddings") or 1 << 30
a.long = min(a.long, max_pos - 16)                                              # prompt + the 8 decoded tokens stay inside the RoPE range
from transformers import PreTrainedTokenizerFast  # noqa: E402
tdir = a.tok or a.pack
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tdir, "tokenizer.json"))
from kyojin_mlx import tuning as TU  # noqa: E402
row_cost, draft_cost = TU.serve("row_cost", 0.4), TU.serve("draft_cost", 0.15)
longest = max([0, *pf_lens, *([a.long] if "1m" in modes else [])]) + 1
ctx = min(a.ctx or max(longest, 4096) + max(a.n, a.long_n) + 64, max_pos)
sessions = max([1, *streams])
eng = Engine(backend, sessions=sessions, ctx=ctx, chunk=a.chunk, block=a.chunk, prefix_bytes=a.prefix_bytes, mtp_depth=a.depth, kv_budget=None,
             row_cost=row_cost, draft_cost=draft_cost, log=lambda *x: None)
load_s = time.time() - t0
folded = vocab < tok.vocab_size


def fold(ids):
    return [t % (vocab - 16) + 16 if t >= vocab else t for t in ids] if folded else ids


def chat_ids(user):
    return fold(tok.encode(render_chat([{"role": "user", "content": user}], thinking=False), add_special_tokens=False))


def long_ids(n, salt):
    base = tok.encode(FILL * 64, add_special_tokens=False)
    body = (base * (n // len(base) + 1))[:n - 1]
    return fold([16 + salt % 64] + body)                                        # distinct first token: never a prefix hit


res = dict(pack=a.pack, chunk=a.chunk, depth=a.depth, n=a.n, ctx=ctx, vocab=vocab, folded=folded, load_s=round(load_s, 1), theirs=THEIRS,
           row_cost=row_cost, draft_cost=draft_cost)


def run_batch(reqs):
    """Submit every request at once; per request: ttft, decode tok/s, tokens; plus the wall time of the batch."""
    out = [None] * len(reqs)

    def worker(i, ids, n):
        t = time.perf_counter()
        h = eng.submit(ids, max_new=n, stop_ids=())
        toks, tf, tl = [], None, None
        while True:
            ev = h.events.get()
            if ev[0] != "tok":
                break
            tl = time.perf_counter()
            tf = tf or tl
            toks.append(ev[1])
        out[i] = dict(t0=t, tf=tf, tl=tl, toks=toks, stats=ev[2] if len(ev) > 2 else None)
    s0 = dict(eng.stats)
    ths = [threading.Thread(target=worker, args=(i, ids, n)) for i, (ids, n) in enumerate(reqs)]
    t = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = max(o["tl"] for o in out) - t
    agg = sum(len(o["toks"]) - 1 for o in out) / max(max(o["tl"] for o in out) - min(o["tf"] for o in out), 1e-9)
    rows = [dict(tokens=len(o["toks"]), ttft_s=round(o["tf"] - o["t0"], 3),
                 tok_s=round((len(o["toks"]) - 1) / max(o["tl"] - o["tf"], 1e-9), 1)) for o in out]
    d = {k: eng.stats.get(k, 0) - s0.get(k, 0) for k in ("decode_ticks", "spec_rows", "spec_rounds", "spec_accepted", "spec_drafted")}
    rows[0]["engine"] = d                                                       # ticks: one forward for every live stream per tick
    return rows, wall, out, agg


def client():
    try:
        mx.reset_peak_memory()
        run_batch([(chat_ids("Say hello."), 16)] * sessions)                               # warm: kernels, draft replay
        med = lambda v: sorted(v)[len(v) // 2]
        if "solo" in modes:
            runs = {c: [] for c in classes}
            for _ in range(a.reps):
                for c in classes:
                    runs[c].append(run_batch([(chat_ids(PROMPTS[c]), a.n)])[0][0])
            res["solo"] = {}
            for c in classes:
                best = max(runs[c], key=lambda r: r["tok_s"])
                res["solo"][c] = dict(best, median_tok_s=med([r["tok_s"] for r in runs[c]]), ttft_median_s=med([r["ttft_s"] for r in runs[c]]))
                print(f"solo {c:10s}: {best['tok_s']:7.1f} tok/s (median {res['solo'][c]['median_tok_s']})  ttft {res['solo'][c]['ttft_median_s']:.3f} s  "
                      f"({best['tokens']} tok, {best['engine']['decode_ticks']} ticks)   theirs {THEIRS['solo'][c]}  {THEIRS['ttft_s'][0]}-{THEIRS['ttft_s'][1]} s", flush=True)
        for K in streams:
            runs = {c: [] for c in classes}
            for _ in range(a.reps):
                for c in classes:
                    rows, wall, _, agg = run_batch([(chat_ids(PROMPTS[c]), a.n)] * K)
                    runs[c].append(dict(agg_tok_s=round(agg, 1), wall_s=round(wall, 3), streams=rows))
            res[f"n{K}"] = {}
            for c in classes:
                best = max(runs[c], key=lambda r: r["agg_tok_s"])
                res[f"n{K}"][c] = dict(best, median_agg_tok_s=med([r["agg_tok_s"] for r in runs[c]]))
                rows, e = best["streams"], best["streams"][0]["engine"]
                th = THEIRS.get(f"n{K}")
                th = th.get(c) if isinstance(th, dict) else (f"{th[0]}-{th[1]}" if th else None)
                print(f"n{K}   {c:10s}: aggregate {best['agg_tok_s']:7.1f} tok/s (median {res[f'n{K}'][c]['median_agg_tok_s']}), per stream "
                      f"{'/'.join(str(r['tok_s']) for r in rows)}, ttft max {max(r['ttft_s'] for r in rows):.3f} s, {e['decode_ticks']} ticks for "
                      f"{sum(r['tokens'] for r in rows)} tok ({e['spec_rows'] / max(e['spec_rounds'], 1):.1f} rows/tick)   theirs {th}", flush=True)
        if pf_lens:
            res["pf"] = []
            for i, L in enumerate(pf_lens):
                mx.clear_cache(); mx.reset_peak_memory()
                rows, _, out, _ = run_batch([(long_ids(L, i), 1)])
                tps = L / rows[0]["ttft_s"]
                res["pf"].append(dict(tokens=L, ttft_s=rows[0]["ttft_s"], tok_s=round(tps, 1), peak_gib=round(mx.get_peak_memory() / GiB, 2)))
                print(f"pf {L:>8d}: {tps:8.1f} tok/s  ({rows[0]['ttft_s']:.2f} s, chunk {a.chunk}, peak {mx.get_peak_memory() / GiB:.2f} GiB)   theirs "
                      f"{THEIRS['pf'].get(L)}", flush=True)
        if "1m" in modes:
            mx.clear_cache(); mx.reset_peak_memory()
            rows, _, out, _ = run_batch([(long_ids(a.long, 99), a.long_n)])
            tps = a.long / rows[0]["ttft_s"]
            res["1m"] = dict(tokens=a.long, ttft_s=rows[0]["ttft_s"], minutes=round(rows[0]["ttft_s"] / 60, 2), tok_s=round(tps, 1), decode_tok_s=rows[0]["tok_s"],
                             first_ids=out[0]["toks"], peak_gib=round(mx.get_peak_memory() / GiB, 2))
            print(f"1m {a.long}: {tps:.1f} tok/s ({rows[0]['ttft_s'] / 60:.2f} min, peak {res['1m']['peak_gib']} GiB), then {rows[0]['tokens']} tok at "
                  f"{rows[0]['tok_s']} tok/s   spec {rows[0].get('engine')}   theirs {THEIRS['1m']} tok/s (29.7 min at 1M)", flush=True)
        res["engine"] = dict(eng.stats)
        res["ok"] = True
    except Exception as e:                                                      # noqa: BLE001  (reported, the engine thread must still stop)
        res["ok"], res["error"] = False, repr(e)
        print("ERROR", repr(e), flush=True)
    finally:
        eng.stop()


th = threading.Thread(target=client)
th.start()
eng.run()                                                                       # the engine runs on the thread that loaded the model
th.join()
res["peak_gib"] = round(mx.get_peak_memory() / GiB, 2)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
print("DS41BAR", "OK" if res.get("ok") else "FAIL", f"load {res['load_s']} s, peak {res['peak_gib']} GiB, engine {res.get('engine')}")
sys.exit(0 if res.get("ok") else 1)

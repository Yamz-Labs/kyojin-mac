"""Long-context gate (kyojin_mlx.longctx): decode ms/step and prefill tok/s at each context point, flag OFF vs KYOJIN_LONGCTX=1, greedy md5 per point.
usage: longctx_gate.py PACK [--points 4096,32768,131072,524288] [--chunk 4096] [--n 32] [--reps 2] [--ref ref.json] [--seed 0] [--json F] [--modes off,on]
       [--set key=json ...]
Each mode runs in its own process (the switches are read at import): same loader as tools/decode_curve.py (build_model + patch_model + rows), one cache, the context
grows point to point by chunked prefill (chunk <= 8192: larger chunks panicked the Ultra on 29/10), then 4 + n greedy steps of the batched core
(dsv4_core, the served multi-row path) on a fork, then `reps` x `n` greedy solo steps (vendor attention + fuse_index). A point passes when both greedy md5 match in every mode. The child prints longctx.HITS so a lever that never reached the served path is visible. Run it alone (one heavy job):
  flock /tmp/ultra_heavy.lock env PYTHONPATH=.:vlm_pkg python tools/longctx_gate.py ~/ds41/model"""
import argparse, hashlib, json, os, statistics, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--points", default="4096,32768,131072,524288"); ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--ref"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json"); ap.add_argument("--modes", default="off,on"); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--child")
a = ap.parse_args()
assert a.chunk <= 8192, "prefill chunks > 8192 rows are not allowed on the Ultra"
MODES = {"off": {"KYOJIN_LONGCTX": "0", "KYOJIN_LONGCTX_ROPE": "0", "KYOJIN_LONGCTX_GROW": "0", "KYOJIN_LONGCTX_PFTOPK": "0", "KYOJIN_IDX_FAST": "0", "KYOJIN_ISCORE_KB": "0"},
         "on": {"KYOJIN_LONGCTX": "1", "KYOJIN_ISCORE_KB": "1", "KYOJIN_ISCORE_KB_MIN": "262144"}}       # iscore_kb is its own flag now (flags.py): keep the arm's old gate


def child(out_path):
    import mlx.core as mx
    import numpy as np
    from kyojin_mlx import headlast as HL, longctx as LC, patch as P, rows as R, wcache as WC
    over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    model, spec, cfg = P.build_model(a.pack, over)
    rep = P.patch_model(model, a.pack, spec, cfg)
    print("longctx install:", (rep or {}).get("longctx"), "| trunk:", R.enable(model, spec), flush=True)
    vocab = {**cfg["text_config"], **over}["vocab_size"]
    pts = [int(x) for x in a.points.split(",")]
    total = max(pts)
    if a.ref:
        base = [t for r in json.load(open(a.ref))["prompts"] for t in r["prompt_ids"]]
        ids = (base * (total // len(base) + 1))[:total]
    else:
        ids = np.random.default_rng(a.seed).integers(16, vocab - 96, total).tolist()
    lm = model.language_model
    cache = lm.make_cache()
    done, rows = 0, []
    for p in pts:
        t0 = time.perf_counter()
        lg = None
        with WC.prefill_scope(model, a.chunk, -(-(p - done) // a.chunk)):
            for s in range(done, p, a.chunk):
                with HL.last_row():
                    lg = lm(mx.array(ids[s:min(s + a.chunk, p)])[None], cache=cache).logits[:, -1].astype(mx.float32)
                mx.eval(lg)
        pf = (p - done) / (time.perf_counter() - t0)
        done = p
        tok0 = int(mx.argmax(lg[0]).item())
        bms, btoks = [], [tok0]                                                              # batched core (dsv4_core, the served multi-row path), batch of one forked user
        bc = R.batch([R.fork(cache)])
        tok = tok0
        for i in range(4 + a.n):
            t0 = time.perf_counter()
            tok = int(mx.argmax(R.step(model, bc, mx.array([tok], dtype=mx.int32))[0]).item()); btoks.append(tok)
            if i >= 4:
                bms.append((time.perf_counter() - t0) * 1e3)
        del bc
        tok = tok0
        toks, ms = [tok], []                                                                 # solo cache (vendor attention + fuse_index)
        for _ in range(4):                                                                   # warm up this context size (tokens kept in the md5)
            tok = int(mx.argmax(R.step(model, cache, mx.array([tok]))[0]).item()); toks.append(tok)
        for _ in range(a.reps):
            t0 = time.perf_counter()
            for _ in range(a.n):
                tok = int(mx.argmax(R.step(model, cache, mx.array([tok]))[0]).item()); toks.append(tok)
            ms.append((time.perf_counter() - t0) / a.n * 1e3)
        r = dict(context=p, decode_ms=statistics.median(ms), decode_ms_min=min(ms), batched_ms=statistics.median(bms),
                 batched_md5=hashlib.md5(json.dumps(btoks).encode()).hexdigest(), prefill_tok_s=pf, peak_gib=mx.get_peak_memory() / 2 ** 30,
                 md5=hashlib.md5(json.dumps(toks).encode()).hexdigest(), hits=dict(LC.HITS))
        rows.append(r)
        print(f"  {p:7d}: decode {r['decode_ms']:.2f} ms/step  batched {r['batched_ms']:.2f}  prefill {pf:.0f} tok/s  peak {r['peak_gib']:.1f} GiB  md5 {r['md5'][:12]}  hits {r['hits']}", flush=True)
    json.dump(rows, open(out_path, "w"))


if a.child:
    child(a.child)
    sys.exit(0)

res = {}
for m in a.modes.split(","):
    out = f"/tmp/longctx_gate_{os.getpid()}_{m}.json"
    print(f"== mode {m}: {MODES[m]}", flush=True)
    env = {**os.environ, **MODES[m]}
    rc = subprocess.call([sys.executable] + sys.argv + ["--child", out], env=env)
    assert rc == 0, f"mode {m} failed (rc {rc})"
    res[m] = json.load(open(out))
ok = True
print(f"{'context':>8} " + " ".join(f"{m + ' ms':>9} {m + ' bat':>9} {m + ' pf':>8}" for m in res) + "  greedy")
for i, r0 in enumerate(next(iter(res.values()))):
    same = len({res[m][i]["md5"] for m in res}) == 1 and len({res[m][i]["batched_md5"] for m in res}) == 1
    ok &= same
    print(f"{r0['context']:8d} " + " ".join(f"{res[m][i]['decode_ms']:9.2f} {res[m][i]['batched_ms']:9.2f} {res[m][i]['prefill_tok_s']:8.0f}" for m in res) + f"  {'SAME' if same else 'DIFF'}")
print("GATE", "PASS" if ok else "FAIL (greedy differs)")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
sys.exit(0 if ok else 1)

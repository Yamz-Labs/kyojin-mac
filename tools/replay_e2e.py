"""End to end with the real draft head: greedy speculative decode (spec.stream, depth D) on a slice with a draft head, then the same decode through the multi-session draft
(Drafter.draft_many / append_many, the serving path) for S sessions. Writes a JSON digest: tokens, every draft proposal, confidences, and a hash of the final draft rings.
Run it with KYOJIN_REPLAY=0 and =1 (tools/replay_e2e.sh) and compare the digests: equal bit for bit, with the replay run's replay.STATS showing regions were replayed.
usage: python tools/replay_e2e.py PACK OUT.json [--tokens 120] [--depth 3] [--sessions 3] [--len 135]"""
import argparse, hashlib, json, os, sys, time
import mlx.core as mx
import numpy as np
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--tokens", type=int, default=120); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--sessions", type=int, default=3); ap.add_argument("--len", type=int, default=135); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
tc = cfg["text_config"]; V = tc["vocab_size"]
dr = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0); R.enable_draft(dr.lm, spec)
S.install(model, spec, tap_layers=set(dr.tap_ids))
rng = np.random.default_rng(a.seed)
def h(arrs):
    m = hashlib.sha256()
    for x in arrs:
        mx.eval(x); m.update(np.array(x.view(mx.uint16) if x.dtype in (mx.float16, mx.bfloat16) else x).tobytes())
    return m.hexdigest()[:16]
dig = {"solo": {}, "rows": {}}
# 1. solo stream: compiled draft + compiled append
prompt = rng.integers(16, V - 16, a.len).tolist()
props = []
orig = dr.draft
def spy(pending, p0, depth):
    r = orig(pending, p0, depth); props.append((pending, p0, r, [round(float(c), 6) for c in dr.stats["conf_lens"][-1:]])); return r
dr.draft = spy
out = []; t0 = time.perf_counter()
for new in S.stream(model, dr, prompt, a.depth):
    out += new
    if len(out) >= a.tokens: break
dr.draft = orig
dig["solo"] = dict(tokens=out[:a.tokens], proposals=props, ring=h(dr.ctx.rows), seconds=time.perf_counter() - t0)
# 2. sessions through draft_many / append_many (rounds of mixed subsets and feed sizes)
ctxs, pos = [], []
for n in (141, 150, 133, 160)[:a.sessions]:
    cache, lg, taps, first = S.prefill(model, rng.integers(16, V - 16, n).tolist(), dr.tap_ids); c = dr.new_ctx(); dr.ctx = c; dr.append([taps[l] for l in dr.tap_ids], first); mx.eval(*c.rows); ctxs.append(c); pos.append(n)
H = tc["hidden_size"]; L = dr.block; log = []
for r in range(60):
    k = int(rng.integers(1, len(ctxs) + 1)); sel = sorted(rng.choice(len(ctxs), k, replace=False).tolist())
    pend = [int(rng.integers(16, V - 16)) for _ in sel]; depth = [int(rng.integers(1, L)) for _ in sel]
    res = dr.draft_many([ctxs[i] for i in sel], pend, [pos[i] for i in sel], depth)
    log.append((sel, pend, res, dr.last_raw[1]))
    items = []
    for i in sel:
        n = int(rng.integers(1, L + 1)); tp = [mx.array(rng.standard_normal((n, H)).astype(np.float16)) for _ in dr.tap_ids]; items.append((ctxs[i], tp, pos[i])); pos[i] += n
    dr.append_many(items)
dig["rows"] = dict(log=log, rings=[h(c.rows) for c in ctxs])
json.dump(dig, open(a.out, "w"))
print("digest written", a.out, "| solo tokens", len(dig["solo"]["tokens"]), "proposals", len(props), "| rows rounds", len(log), "| rings", dig["rows"]["rings"])

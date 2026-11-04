"""Live loop gate for verify_replay: a scripted greedy-free loop (random tokens per round, scripted accept counts) through the served path (be.spec_verify_bc + commit + settle), run twice from the
same state: eager (KYOJIN_VERIFY_REPLAY=0) and replay (=1); logits of every round must be bit-identical. usage: tools/rp_verify_live_check.py PACK --rounds 60 [--acc 0,3,1,2]
Run it once per mode, it prints a digest and the per-round CPU; compare digests (--dump FILE saves the logits for np comparison)."""
import argparse, hashlib, os, statistics, sys, time
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rounds", type=int, default=60); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--acc", default="0,3,1,2,3,0,2"); ap.add_argument("--len", type=int, default=140); ap.add_argument("--dump", default=""); ap.add_argument("--depths", default=""); ap.add_argument("--same", type=int, default=0); ap.add_argument("--pipe", type=int, default=0); ap.add_argument("--same-after", type=int, default=-1); ap.add_argument("--same-before", type=int, default=-1)
a = ap.parse_args()
import numpy as np, mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import verify_replay as VR
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; D = a.depth
rng = np.random.default_rng(7); base = rng.integers(16, min(V - 16, 4000), 61).tolist()
c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16] + (base * 4)[:a.len - 1], draft=d); bc = be.batch([c])
accs = [int(x) for x in a.acc.split(",")]
depths = [int(x) for x in a.depths.split(",")] if a.depths else [D]
h = hashlib.sha1(); allp = []; cpu = []
for i in range(a.rounds):
    Di = depths[i % len(depths)]
    toks = [[int(t) for t in rng.integers(16, min(V - 16, 4000), Di + 1)]]
    if a.same: toks = [[a.same] * (Di + 1)]
    if a.same_before >= 0 and i < a.same_before: toks = [[toks[0][0]] * (Di + 1)]
    if a.same_after >= 0 and i >= a.same_after and i % 3 != 0: toks = [[toks[0][0]] * (Di + 1)]
    t0 = time.perf_counter()
    ps = be.spec_verify_bc(bc, toks); t1 = time.perf_counter(); mx.async_eval(ps.logits); t2 = time.perf_counter()
    if a.pipe:
        allp.append(ps.logits); lg = None
    else:
        lg = np.array(ps.logits); h.update(lg.tobytes()); allp.append(lg)
    ps.commit([min(accs[i % len(accs)], Di)]); be.settle(bc, ())
    cpu.append(t2 - t0)
if a.pipe:
    allp = [np.array(x) for x in allp]
    for lg in allp: h.update(lg.tobytes())
print("DIGEST", h.hexdigest()[:16], "rounds", a.rounds, "replay stats", VR.STATS)
print("verify CPU ms (build+encode) median of last half: %.3f  min %.3f" % (1e3 * statistics.median(cpu[a.rounds // 2:]), 1e3 * min(cpu[a.rounds // 2:])))
if a.dump: np.save(a.dump, np.stack(allp))

"""Census of consecutive LIVE verify rounds (served path rows.verify + commit): record rounds with the rp recorder, dump each command stream, so rp_ident.py / rp_affine.py can tell
what a replay must patch between rounds. usage: rp_verify_live.py PACK DUMPDIR [--rounds 6] [--depth 3] [--acc 3,3,3,3,3,3]"""
import argparse, ctypes, os
here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp.dylib")); lib.rp_install(1)
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("dump"); ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--acc", default=""); ap.add_argument("--len", type=int, default=140)
a = ap.parse_args()
import numpy as np, mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; D = a.depth
rng = np.random.default_rng(7); base = rng.integers(16, V - 16, 61).tolist()
c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16] + (base * 4)[:a.len - 1], draft=d); bc = be.batch([c])
accs = [int(x) for x in a.acc.split(",")] if a.acc else [D] * a.rounds
os.makedirs(a.dump, exist_ok=True)
def rnd(i, rec):
    toks = [[int(t) for t in rng.integers(16, V - 16, D + 1)]]
    mx.synchronize()
    if rec: mx.set_cache_limit(0); mx.clear_cache(); lib.rp_rec_begin(i)
    ps = be.spec_verify_bc(bc, toks); mx.eval(ps.logits)
    ps.commit([accs[i % len(accs)]]); be.settle(bc, ())
    mx.synchronize()
    if rec:
        n = lib.rp_rec_end(); print(f"round {i}: cmds {n}"); lib.rp_dump(i, os.path.join(a.dump, f"step{i}.txt").encode()); lib.rp_hostconst_dump(i, os.path.join(a.dump, f"hc{i}.txt").encode())
for i in range(3): rnd(i, False)
for i in range(a.rounds): rnd(i, True)

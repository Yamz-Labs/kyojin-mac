#!/usr/bin/env python3
"""DS4.1 day-1 bench in one process, on the pack as served (HF EXL3 layout restacked at load time, Engram tables when present, DSpark draft head):
  load     build_model + patch_model + Engram + build_drafter: seconds, MLX peak / active memory
  prefill  exact solo prefill (kyojin_mlx.spec.prefill, the server path with the draft-head taps) at each --pf length, chunk --chunk: tok/s
  solo     plain greedy decode (depth 0) of --n tokens after a --ctx-token prompt: tok/s (first token excluded: it comes from the prefill)
  mtp      speculative decode at --depth with the real drafter on the same prompt: tok/s, accepted / drafted, tokens == solo tokens (EXACT)
  priors   server depth-policy priors (kyojin_mlx/serve/engine.py row_cost / draft_cost, in solo steps): draft ms per call / solo step ms, and
           (verify ms at --depth - solo step ms) / solo step ms / drafted rows per round; --tuning stores them in the pack's tuning table
Prompts: the --ref prompts (results_mac/ref_ds41/ref.json, real text: the acceptance means something) tiled to the length, else seeded random ids.
usage: ds41_bench.py PACK [--pf 2048,8192,32768] [--chunk 8192] [--ctx 2048] [--n 128] [--depth 3] [--ref ref.json] [--json F] [--tuning]
Targets (owner, M5 Ultra): solo >= 70 tok/s, MTP >= 110 tok/s, prefill >= 2000 tok/s."""
import argparse, json, os, sys, time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pack")
ap.add_argument("--pf", default="2048,8192,32768")
ap.add_argument("--chunk", type=int, default=int(os.environ.get("KYOJIN_PF_CHUNK", "8192")))
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--ref")
ap.add_argument("--json")
ap.add_argument("--tuning", action="store_true", help="write the measured priors into the entry \"serve\" of <pack>/tuning/<device>.json (the server reads them when no --row-cost / --draft-cost is given)")
a = ap.parse_args()
TARGETS = dict(solo=70.0, mtp=110.0, prefill=2000.0)
GiB = 2 ** 30
res = dict(pack=a.pack, chunk=a.chunk, targets=TARGETS)

t0 = time.time()
model, spec, cfg = P.build_model(a.pack, {})
P.patch_model(model, a.pack, spec, cfg)
engram = False
try:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg)
    if es is not None:
        tables = ER.EngramTables(a.pack, es)
        bad = tables.validate()
        assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
        EM.install_parallel_gather(model, tables)
        engram = True
except ImportError:
    pass
R.enable(model, spec)
t_trunk = time.time() - t0
dr = D.build_drafter(model, a.pack, spec, cfg)
S.install(model, spec, tap_layers=set(dr.tap_ids))
res["load"] = dict(s=round(time.time() - t0, 1), trunk_s=round(t_trunk, 1), engram=engram, stages=dr.n_stages,
                   active_gib=round(mx.get_active_memory() / GiB, 1), peak_gib=round(mx.get_peak_memory() / GiB, 1))
print(f"load: {res['load']}", flush=True)

vocab = cfg.get("text_config", cfg)["vocab_size"]
if a.ref:
    base = [t for r in json.load(open(a.ref))["prompts"] for t in r["prompt_ids"]]
else:
    base = np.random.default_rng(0).integers(0, vocab, 4096).tolist()


def ids(n):
    return (base * (n // len(base) + 1))[:n]


# ---- prefill
S.prefill(model, ids(512), dr.tap_ids, a.chunk)                                  # warm the kernels
res["prefill"] = []
for L in [int(x) for x in a.pf.split(",") if x]:
    mx.clear_cache(); mx.reset_peak_memory()
    t = time.perf_counter()
    cache, lg, taps, first = S.prefill(model, ids(L), dr.tap_ids, a.chunk)
    mx.eval(lg)
    dt = time.perf_counter() - t
    res["prefill"].append(dict(tokens=L, s=round(dt, 3), tok_s=round(L / dt, 1), peak_gib=round(mx.get_peak_memory() / GiB, 2)))
    print(f"prefill {L}: {L / dt:.1f} tok/s ({dt:.2f} s, chunk {min(a.chunk, L)})", flush=True)
    del cache, lg, taps


# ---- solo and MTP decode on one prompt
def run(depth):
    st, out, t = {}, [], None
    for new in S.stream(model, dr, ids(a.ctx), depth, st, a.chunk):
        if t is None:
            mx.eval(mx.array(0)); t = time.perf_counter(); out += new; continue
        out += new
        if len(out) >= a.n + 1:
            break
    dt = time.perf_counter() - t
    k = len(out) - 1
    return out[:a.n + 1], dict(depth=depth, tokens=k, s=round(dt, 3), tok_s=round(k / dt, 1), rounds=st["rounds"], drafted=st["drafted"], accepted=st["accepted"],
                               draft_ms=round(1e3 * st["t_draft"] / max(st["rounds"], 1), 3), verify_ms=round(1e3 * st["t_verify"] / max(st["rounds"], 1), 3))


run(0)                                                                           # warm
solo_ids, solo = run(0)
mtp_ids, mtp = run(a.depth)
mtp["exact"] = mtp_ids == solo_ids
res["solo"], res["mtp"] = solo, mtp
step = solo["verify_ms"]
rows_per = (mtp["drafted"] / max(mtp["rounds"], 1))
res["priors"] = dict(draft_cost=round(mtp["draft_ms"] / step, 3), row_cost=round(max(0.0, (mtp["verify_ms"] - step) / step / rows_per), 3) if rows_per else None,
                     accept_rate=round(mtp["accepted"] / max(mtp["drafted"], 1), 3))
print(f"solo: {solo['tok_s']} tok/s ({solo['verify_ms']} ms/step), ctx {a.ctx}", flush=True)
print(f"mtp depth {a.depth}: {mtp['tok_s']} tok/s, accepted {mtp['accepted']}/{mtp['drafted']}, draft {mtp['draft_ms']} ms, verify {mtp['verify_ms']} ms, "
      f"{'EXACT' if mtp['exact'] else 'MISMATCH'} vs solo", flush=True)
print(f"priors: {res['priors']}", flush=True)
pf_best = max((r["tok_s"] for r in res["prefill"]), default=None)
res["verdict"] = dict(solo=solo["tok_s"] >= TARGETS["solo"], mtp=mtp["tok_s"] >= TARGETS["mtp"], prefill=None if pf_best is None else pf_best >= TARGETS["prefill"], exact=mtp["exact"])
print(f"targets: solo {solo['tok_s']} / {TARGETS['solo']:.0f}, mtp {mtp['tok_s']} / {TARGETS['mtp']:.0f}, prefill best {pf_best} / {TARGETS['prefill']:.0f}", flush=True)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
if a.tuning and mtp["exact"] and res["priors"]["row_cost"] is not None:
    from kyojin_mlx import tuning as TU
    pr = res["priors"]
    print("tuning:", TU.set_entry(a.pack, "serve", dict(row_cost=pr["row_cost"], draft_cost=pr["draft_cost"])), flush=True)
print("DS41BENCH", "EXACT" if mtp["exact"] else "MISMATCH")
sys.exit(0 if mtp["exact"] else 1)

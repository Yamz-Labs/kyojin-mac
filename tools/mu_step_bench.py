"""Verify step ms of N sessions x (D+1) rows on the batched DS4.1 core, routed experts per row (plain) against grouped by expert, interleaved A/B.
usage: mu_step_bench.py PACK [--users 1,2,4,8] [--depths 0,1,3] [--len 64] [--rounds 9] [--modes plain,grouped] [--json F]
Sessions prefill different token streams; the rows of a session are its real greedy continuation (what a fully accepted draft feeds), so the routing is the model's own.
Time = one `rows.verify` + evaluating its logits (no commit: the pass leaves the cache as it was). min / p25 over the rounds, mode order alternating."""
import argparse, json, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import dsv4_core as DC, patch as P, rows as R

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--users", default="1,2,4,8"); ap.add_argument("--depths", default="0,1,3")
ap.add_argument("--len", type=int, default=64); ap.add_argument("--rounds", type=int, default=9); ap.add_argument("--modes", default="plain,grouped")
ap.add_argument("--json"); ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack, {})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
moes = [m.moe for _, m in model.language_model.named_modules() if isinstance(getattr(m, "moe", None), P.M.MoeGather)]
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
Ns = [int(v) for v in a.users.split(",")]
Ds = [int(v) for v in a.depths.split(",")]
modes = a.modes.split(",")
maxD = max(Ds)
caches, conts = [], []
for u in range(max(Ns)):
    base = rng.integers(16, vocab - 16, int(rng.integers(6, 14)))
    prompt = np.concatenate([np.where(rng.random(len(base)) < 0.8, base, rng.integers(16, vocab - 16, len(base))) for _ in range(a.len // len(base) + 1)])[:a.len].tolist()
    c, lg = R.prefill(model, prompt)
    tok = int(mx.argmax(lg, axis=-1).item())
    run = R.batch([R.fork(c)])
    seq = [tok]
    for _ in range(maxD):
        l = R.step(model, run, mx.array([seq[-1]])); mx.eval(l); seq.append(int(mx.argmax(l[0]).item()))
    caches.append(c); conts.append(seq)
res = {}
print(f"# {a.pack}: {len(moes)} MoE layers, rounds {a.rounds}", flush=True)
for N in Ns:
    for D in Ds:
        bc = R.batch([R.fork(c) for c in caches[:N]])
        lists = [conts[u][:D + 1] for u in range(N)]
        outs = {}
        ts = {m: [] for m in modes}
        for r in range(a.rounds + 2):
            for m in (modes if r % 2 == 0 else modes[::-1]):
                for mo in moes:
                    mo.grouped = (m != "plain")
                    mo.hotk = 4 if m == "nax" else 0
                t = time.perf_counter()
                ps = R.verify(model, bc, lists)
                mx.eval(ps.logits)
                dt = time.perf_counter() - t
                st = DC.stack_rows(bc[0]); st._counts = None; st.pend = {}                      # drop the pass: the cache stays as it was
                if r >= 2:
                    ts[m].append(dt * 1e3)
                outs[m] = np.array(ps.logits)
        same = all(np.array_equal(outs[modes[0]].view(np.uint32), outs[m].view(np.uint32)) for m in modes if m != "nax")
        ent = {m: (float(np.min(v)), float(np.percentile(v, 25))) for m, v in ts.items()}
        res[f"N{N}_R{N * (D + 1)}"] = dict(ms=ent, logits_bitwise=same)
        print(f"N={N} D={D} rows={N * (D + 1):2d} | " + " | ".join(f"{m} {ent[m][0]:7.2f}/{ent[m][1]:7.2f}" for m in modes) + " ms (min/p25)"
              + "".join(f" | {m} x{ent['plain'][0] / ent[m][0]:.2f}" for m in modes if m != "plain") + f" | logits bitwise (plain = grouped) {same}", flush=True)
print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

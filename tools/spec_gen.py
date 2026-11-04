"""End to end speculative decode with the real DSpark drafter: greedy tokens with the drafter == greedy tokens without it (token identical),
plus the acceptance and the time split (draft / verify / ring update) of the run.
usage: spec_gen.py PACK [--n 48] [--len 135] [--depths 0,1,2,3] [--seeds 0,1] [--set key=json ...] [--conf 0.5] [--json F] [--ref ref.json [--prompts a,b]]
--ref: take the prompts (token ids) of a reference file (results_mac/ref_ds41/ref.json) instead of random ids: real text, so the acceptance means something
(a full pack with its Engram shards loads as in tools/check_ds41_ref.py; on the Ultra this is the day-1 MTP measurement).
PACK holds the trunk layers and the `mtp.<stage>.*` tensors (tools/shrink_pack.py ... --mtp 0,1,2 --taps ...). The drafter of a slice is
real DS4.1 weights on a 2-layer trunk, so its acceptance says nothing about the full model: only exactness and the timing carry over.
depth 0 = no drafting (the chained pass with one row, a plain step)."""
import argparse, json, sys, time

import mlx.core as mx
import numpy as np

from kyojin_mlx import archs, dspark as D, patch as P, rows as R, spec as S

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--len", type=int, default=135)
ap.add_argument("--depths", default="0,1,2,3"); ap.add_argument("--seeds", default="0"); ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--conf", type=float); ap.add_argument("--json"); ap.add_argument("--ref"); ap.add_argument("--prompts", default="")
ap.add_argument("--interleave", action="store_true", help="after the runs: two sessions on one model and one drafter, rounds alternating, each equal to plain greedy")
ap.add_argument("--chunk", type=int, help="prefill chunk size (default: one pass)")
ap.add_argument("--neg", default="", help="blind: the trunk 'agrees' with every proposal (must fail with --oracle)")
ap.add_argument("--oracle", action="store_true", help="replace the proposals by the true continuation with wrong tokens injected (accepts 0..D per round), "
                "and check the drafter ring against one built from solo-step taps")
a = ap.parse_args()
over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
try:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg)
    if es is not None:
        tables = ER.EngramTables(a.pack, es)
        bad = tables.validate()
        assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
        EM.install_parallel_gather(model, tables)
except ImportError:
    pass
print("trunk:", R.enable(model, spec), flush=True)
tc = {**cfg["text_config"], **over}
cfg = {**cfg, "text_config": tc}
t0 = time.time()
drafter = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=a.conf)
print(f"drafter: {drafter.n_stages} stages, block {drafter.block}, taps {drafter.tap_ids}, window {drafter.window}, loaded in {time.time() - t0:.1f}s", flush=True)
print("spec.install:", S.install(model, spec, tap_layers=set(drafter.tap_ids)), flush=True)
vocab = tc["vocab_size"]
if a.neg == "blind":
    _verify = S.verify

    def blind(model, base, tokens, link=True):
        lg, tp, ch = _verify(model, base, tokens, link)
        fake = mx.concatenate([mx.eye(lg.shape[-1], dtype=lg.dtype)[mx.array(tokens[1:])], lg[-1:]], axis=0) if len(tokens) > 1 else lg
        return fake, tp, ch
    S.verify = blind


def ring_ok(prompt, ref_long, p_end):
    """The drafter ring after a speculative run equals (to fp16 rounding) the ring built from the taps of plain solo steps."""
    got = [None if r is None else np.array(r.astype(mx.float32)) for r in drafter.ctx.rows]
    cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids, a.chunk)
    drafter.reset()
    drafter.append([taps[l] for l in drafter.tap_ids], first)
    for j in range(p_end - len(prompt)):
        cache[0].taps = {}
        R.step(model, cache, mx.array([ref_long[j]]))
        tp = cache[0].taps
        del cache[0].taps
        drafter.append([tp[l][0] for l in drafter.tap_ids], len(prompt) + j)
    mx.eval(drafter.ctx.rows)
    worst = 0.0
    for g, r in zip(got, drafter.ctx.rows):
        r = np.array(r.astype(mx.float32))
        if g.shape != r.shape:
            print(f"  ring shape {g.shape} != {r.shape}", flush=True)
            return False
        worst = max(worst, float(np.abs(g - r).max()) / (float(np.abs(r).max()) + 1e-6))
    print(f"  ring vs solo-step taps: shape {got[0].shape}, max relative diff {worst:.2e}", flush=True)
    return worst < 5e-2


res, ok = [], True
batched_core = (getattr(model.language_model, '_kyojin_rows', None) or {}).get('adapter') == 'dsv4_core'
if a.ref:
    work = [(r["name"], r["prompt_ids"]) for r in json.load(open(a.ref))["prompts"] if not a.prompts or r["name"] in a.prompts.split(",")]
else:
    work = [(f"seed{seed}", np.random.default_rng(seed).integers(16, vocab - 96, a.len).tolist()) for seed in [int(x) for x in a.seeds.split(",")]]
for seed, prompt in work:
    S.plain_greedy(model, prompt, 4, a.chunk)                                                                  # warm up
    t = time.perf_counter()
    ref_long = S.plain_greedy(model, prompt, a.n + 8, a.chunk)
    t_plain = time.perf_counter() - t
    if batched_core:                                                    # the plain steps run on the batched core: another rounding than the chain
        ref_step, ref_long = ref_long, S.chain_greedy(model, prompt, a.n + 8)
        d = next((i for i, (x, y) in enumerate(zip(ref_step, ref_long)) if x != y), None)
        print(f"{seed}: plain steps (batched core) vs chain path: " + ("same tokens" if d is None else f"first different token {d} (rounding of near tied logits)"), flush=True)
    ref = ref_long[:a.n]
    for depth in [int(x) for x in a.depths.split(",")]:
        st = {}
        real_draft = drafter.draft
        if a.oracle:
            cnt = [0]

            def oracle(pending, p0, dep, _real=real_draft):
                _real(pending, p0, dep)                                                   # run the real drafter (timing, shapes), then override
                i = p0 - len(prompt)
                D = 1 + cnt[0] % dep
                if i >= len(ref_long) or pending != ref_long[i]:                           # the run left the plain greedy path
                    return [pending] * D
                k = [-1, 0, 1, 2, -1, 1, 0, 2][cnt[0] % 8]
                cnt[0] += 1
                d = ref_long[i + 1:i + 1 + D]
                if 0 <= k < len(d):
                    d[k] = (d[k] + 1 + cnt[0]) % vocab
                return d
            drafter.draft = oracle
        if depth > 0 or not a.oracle:
            t = time.perf_counter()
            got = S.generate(model, drafter, prompt, a.n, depth, st, a.chunk)
            dt = time.perf_counter() - t
        drafter.draft = real_draft
        same = got == ref
        if a.oracle and depth > 0:
            same &= ring_ok(prompt, ref_long, st["p_end"])
        ok &= same
        rate = st["accepted"] / max(st["drafted"], 1)
        print(f"{seed} depth {depth}: tokens equal to {'chain-path' if batched_core else 'plain'} greedy: {same}; rounds {st['rounds']}, drafted {st['drafted']}, accepted {st['accepted']} "
              f"({rate:.2f}), skipped {st['skipped']}, tok/round {(a.n - 1) / max(st['rounds'], 1):.2f}; draft {1e3 * st['t_draft'] / max(st['rounds'], 1):.0f} ms, "
              f"verify {1e3 * st['t_verify'] / max(st['rounds'], 1):.0f} ms, ring {1e3 * st['t_append'] / max(st['rounds'], 1):.0f} ms per round; "
              f"{a.n / dt:.1f} tok/s (plain solo {a.n / t_plain:.1f})", flush=True)
        if not a.oracle and depth > 0 and st["drafted"] >= 8 and rate < 0.3:
            print(f"  WARNING: only {rate:.2f} of the drafted tokens are accepted (a healthy head gives 0.6-0.9 on text, see ULTRA.md section 11). The output is still exact; "
                  f"check the draft head: tools/draft_dump.py on the Mac + tools/draft_ref.py on the box compare one draft call with the checkpoint's model.py.", flush=True)
        if not a.oracle and depth > 0 and st["rounds"] >= 8 and st["skipped"] > st["rounds"] // 2:
            print(f"  WARNING: the confidence gate skipped {st['skipped']} of {st['rounds']} rounds (threshold {drafter.conf_threshold}, KYOJIN_DSPARK_CONF).", flush=True)
        res.append(dict(seed=seed, depth=depth, same=same, **{k: v for k, v in st.items()}, tok_s=a.n / dt, tok_s_plain=a.n / t_plain))
if a.interleave and len(work) >= 1:
    pa = work[0][1]
    pb = work[-1][1] if len(work) > 1 else list(reversed(pa))
    refs = [S.reference_greedy(model, p, a.n, a.chunk) for p in (pa, pb)]
    real_draft, log = drafter.draft, {}

    def logged(pending, p0, dep):
        d = real_draft(pending, p0, dep)
        log.setdefault(id(drafter.ctx), []).append((pending, p0, d))
        return d
    drafter.draft = logged

    def run(gens):
        outs = [[] for _ in gens]
        while min(len(o) for o in outs) < a.n:
            for i, g in enumerate(gens):
                outs[i] += next(g)
        return outs
    solo = []
    for p in (pa, pb):
        log.clear()
        o = run([S.stream(model, drafter, p, 3, None, a.chunk)])
        solo.append((o[0][:a.n], next(iter(log.values()))))
    log.clear()
    outs = run([S.stream(model, drafter, p, 3, None, a.chunk) for p in (pa, pb)])
    drafter.draft = real_draft
    inter = list(log.values())
    same = all(outs[i][:a.n] == refs[i] for i in (0, 1))
    n_cmp = min(len(inter[0]), len(solo[0][1])), min(len(inter[1]), len(solo[1][1]))
    prop = inter[0][:n_cmp[0]] == solo[0][1][:n_cmp[0]] and inter[1][:n_cmp[1]] == solo[1][1][:n_cmp[1]]
    ok &= same and prop
    print(f"two interleaved sessions, one drafter: tokens equal to {'chain-path' if batched_core else 'plain'} greedy: {same}; proposals equal to each session run alone: {prop} ({sum(n_cmp)} rounds compared)", flush=True)
print("EXACT" if ok else "MISMATCH", flush=True)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
sys.exit(0 if ok else 1)

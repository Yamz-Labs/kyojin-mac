"""End to end: stacked pack vs per-expert raw pack (load-time restack). Trunk prefill logits, 4 speculative rounds (draft tokens + verify
logits) and plain decode logits must be bit-identical. Covers MLA/DSA attention, MoE, shared experts, grouped wo_a, DSpark draft head."""
import gc, sys, time
import mlx.core as mx, numpy as np
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S


def run(pack):
    t0 = time.time()
    model, spec, cfg = P.build_model(pack, {})
    P.patch_model(model, pack, spec, cfg)
    R.enable(model, spec)
    dr = D.build_drafter(model, pack, spec, cfg)
    S.install(model, spec, tap_layers=set(dr.tap_ids))
    lt = time.time() - t0
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, cfg["text_config"]["vocab_size"], 135).tolist()
    out = {}
    dr.ctx = dr.new_ctx()
    cache, lg, taps, first = S.prefill(model, prompt, dr.tap_ids)
    dr.append([taps[l] for l in dr.tap_ids], first)
    out["prefill"] = np.array(lg)
    pending = int(mx.argmax(lg[0]).item())
    drafts_all = []
    for r in range(4):
        p0 = cache[0].offset
        drafts = dr.draft(pending, p0, 3)
        lgs, tp, chain = S.verify(model, cache, [pending] + drafts)
        out[f"verify{r}"] = np.array(lgs.astype(mx.float32))
        am = mx.argmax(lgs, axis=-1).tolist()
        a = 0
        while a < len(drafts) and am[a] == drafts[a]:
            a += 1
        cache = S.commit(chain, a)
        dr.append([tp[l][:a + 1, 0] for l in dr.tap_ids], p0)
        mx.eval(dr.ctx.rows)
        drafts_all.append(drafts)
        pending = am[a]
    out["drafts"] = np.array(drafts_all)
    out["greedy"] = np.array(S.plain_greedy(model, prompt[:40], 8))
    print(f"{pack}: load {lt:.1f}s drafts {drafts_all} peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    del model, dr, cache
    gc.collect(); mx.clear_cache()
    return out


a, b = run(sys.argv[1]), run(sys.argv[2])
ok = True
for k in a:
    same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
    ok &= same
    print(f"{k:10s} {'BITIDENT' if same else 'DIFF max ' + str(np.abs(a[k].astype(np.float64) - b[k]).max())}")
print("E2E", "EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

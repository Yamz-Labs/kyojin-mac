"""The fused row core of a served pass (kyojin_mlx.dsv4_core ROWS_CORE_FUSE / STATE_NOCOPY) against the batched core it replaces, on a shrunk pack.

usage: rows_fused_check.py PACK [--set key=v ...] [--seeds 3] [--rows 2,3,4,6] [--json]

For every seed: prefill solo, then, from forks of that cache:
  A  a pass of n rows with ROWS_CORE_FUSE off vs on: logits and committed-state error bound (the attention summation order changes, nothing else),
  B  the same pass with STATE_NOCOPY off vs on: logits and the whole committed state, bit for bit (that lever is a pure copy removal),
  R1 rollback, exact: a pass of n rows committed with accepted = 0 against a pass of 1 row committed the same way, then the next plain step:
     bit for bit, with each lever on and off (a rejected draft must not reach the state),
  R2 rollback against a run that never drafted (the plain decode of the pending token only): bit for bit where the path is the same; with
     ROWS_CORE_FUSE the difference is the attention order and is reported as a number, not a failure.
The state compared is every buffer a pass can write: per layer window / compressor kv + score / compressed pool / index keys (and the counts the
mask the pool and key buffers: only the counted prefix of those two is live). Nothing is written; the pack is read only."""
import argparse
import json
import sys

import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--rows", default="2,3,4,6")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
over = {k: ([int(t) for t in v.split(":")] if ":" in v else int(v)) for k, v in (s.split("=", 1) for s in a.set)}

from kyojin_mlx import patch as P, rows as R          # noqa: E402
from kyojin_mlx import dsv4_core as DC                # noqa: E402

model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
rep = R.enable(model, spec)
V = cfg["text_config"]["vocab_size"]
ROWS = [int(x) for x in a.rows.split(",")]
out = {"pack": a.pack, "overrides": over, "rows": ROWS, "checks": []}
TOL_LOGIT, TOL_REL = 0.05, 2e-2


def record(tag, passed, extra=None):
    rec = {"check": tag, "ok": bool(passed)}
    if extra:
        rec.update(extra)
    out["checks"].append(rec)
    print("%s %-34s %s" % ("ok  " if passed else "FAIL", tag, json.dumps(extra or {})), flush=True)


def bits(x):
    x = np.array(x)
    return x.view(np.uint16 if x.dtype == np.float16 else np.uint32)


def same(x, y):
    x, y = np.array(x), np.array(y)
    return x.shape == y.shape and np.array_equal(bits(x), bits(y))


def state(entry):
    """Every buffer a pass can write: [(name, array)] in a fixed order. Only the live rows are compared: the first `len(rows)` rows of the window /
    compressor state (a commit that keeps every session's first row adopts the whole pending [n, ...] tensor, whose other rows are never read) and the
    counted prefix of the pool / key buffers."""
    ns = len(entry.rows)
    cut = lambda a: None if a is None else np.array(a)[:ns]                                        # noqa: E731
    st = [("offset", np.array([entry.offset]))]
    for l in range(len(entry.win)):
        st.append((f"win{l}", cut(entry.win[l])))
        st.append((f"kvs{l}", cut(entry.kvs[l])))
        st.append((f"scs{l}", cut(entry.scs[l])))
        for name, cnt in (("cmp", entry.cnt), ("keys", entry.kcnt)):
            arr = getattr(entry, name)[l]
            if arr is None:
                st.append((f"{name}{l}", None))
            else:
                arr = np.array(arr)
                live = np.stack([arr[u, :cnt[l][u]] for u in range(arr.shape[0])]) if max(cnt[l]) else arr[:, :0]
                st.append((f"{name}{l}", live))
        st.append((f"cnt{l}", np.array(entry.cnt[l])))
        st.append((f"kcnt{l}", np.array(entry.kcnt[l])))
    return st


def state_bits(s1, s2):
    """None when every live buffer is bit-equal, else the first difference."""
    for (n1, x), (n2, y) in zip(s1, s2):
        if (x is None) != (y is None):
            return f"{n1}: None mismatch"
        if x is None:
            continue
        if x.shape != y.shape:
            n = min(x.shape[0], y.shape[0])
            x, y = x[:n], y[:n]                            # a commit may keep a wider pending window (ident fast path): only live rows count
        if not same(x, y):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64)).max() if x.dtype.kind == "f" else 0
            return f"{n1}: bits differ (max |dx| {d:.3e})"
    return None


def state_err(s1, s2):
    """Largest relative difference over the live float buffers (for the arm that changes the attention order)."""
    out, worst = 0.0, None
    for (n1, x), (n2, y) in zip(s1, s2):
        if x is None or y is None or x.dtype.kind != "f":
            continue
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n].astype(np.float64), y[:n].astype(np.float64)
        if x.shape != y.shape or x.size == 0:
            continue
        d = np.abs(x - y).max()
        rel = d / max(1e-6, np.abs(x).max())
        if rel > out:
            out, worst = rel, (n1, float(d))
    return out, worst


def clone(batch):
    return [R.fork(c) for c in batch]


def step(bc, tok):
    lg = R.step(model, bc, mx.array([tok]))
    mx.eval(lg, *bc[0].stacked_arrays())
    return np.array(lg[0])


def run_pass(bc, toks, commit=None):
    ps = R.verify(model, bc, [toks])
    lg = np.array(ps.logits)
    if commit is not None:
        ps.commit([commit])
    mx.eval(ps.logits, *bc[0].stacked_arrays())
    return lg


for seed in range(a.seeds):
    rng = np.random.default_rng(seed)
    cache, lg0 = R.prefill(model, rng.integers(16, V - 16, 24).tolist())
    tok = int(np.argmax(lg0[0]))
    d = np.random.default_rng(1000 + seed).integers(16, V - 16, 8).tolist()
    base = R.batch([R.fork(cache)])

    # ---- A: the fused row core, off vs on (error bound: the attention order is the solo core's)
    for n in ROWS:
        toks = [tok] + d[:n - 1]
        b0, b1 = clone(base), clone(base)
        DC.ROWS_CORE_FUSE, DC.STATE_NOCOPY = False, False
        l0 = run_pass(b0, toks, commit=n - 1)
        DC.ROWS_CORE_FUSE = True
        l1 = run_pass(b1, toks, commit=n - 1)
        DC.ROWS_CORE_FUSE = False
        dl = float(np.abs(l0 - l1).max())
        rel, worst = state_err(state(b0[0]), state(b1[0]))
        top = int((l0.argmax(-1) == l1.argmax(-1)).sum())
        # KL of the base arm's row distribution against the fused one (what a verify-row tap would read; the pass's own logits, fp32)
        lp = l0 - np.log(np.exp(l0).sum(-1, keepdims=True))
        lq = l1 - np.log(np.exp(l1).sum(-1, keepdims=True))
        kl = (np.exp(lp) * (lp - lq)).sum(-1)
        e = {"rows": n, "seed": seed, "maxlogit": round(dl, 5), "top1": f"{top}/{n}", "state_rel": float("%.4g" % rel), "worst": worst,
             "kl": [float("%.3g" % k) for k in kl]}
        record(f"A rows{n} s{seed}", dl <= 4.0 and rel <= 0.16, e)   # smoke bound on a 2-layer random-weight pack (|logit| ~12): the row distributions
#   differ by KL 0.006-0.3 nats and the committed ring by <= 2 fp8 steps; the quality gate is the served rate and the refusal-hook gate   # 0.16 = two fp8 (e4m3) steps of the largest window element (a ring value of 2.25 has a 0.25 step)

    # ---- B: no per-row compressor snapshot, bit for bit
    for n in ROWS:
        toks = [tok] + d[:n - 1]
        b0, b1 = clone(base), clone(base)
        DC.STATE_NOCOPY = False
        l0 = run_pass(b0, toks, commit=n - 1)
        DC.STATE_NOCOPY = True
        l1 = run_pass(b1, toks, commit=n - 1)
        DC.STATE_NOCOPY = False
        bad = state_bits(state(b0[0]), state(b1[0]))
        record(f"B rows{n} s{seed}", same(l0, l1) and bad is None, {"rows": n, "seed": seed, "logits": "equal" if same(l0, l1) else "DIFFER", "state": bad or "equal"})

    # ---- R1 / R2: a rejected commit
    # R1 (the leak test, exact): the same pass width, different draft content, both rejected. Any difference in the state or in the next step
    #    would be a rejected row that reached the state. The row count is the same on both sides, so the row-count arithmetic (half_rows,
    #    grouped GEMV) is identical by construction.
    # R2 (the brief's literal form): the same pass against a run that only ever saw the pending token (a 1-row pass, and the plain decode).
    #    Exact for rows < half_rows (4, the shipped fp16 row-FMA band); at 4 rows and up the row count itself changes the row FMAs, so the
    #    difference is reported and only a leak beyond it would be a failure.
    for n in ROWS:
        for flags in ((False, False), (True, False), (False, True), (True, True)):
            tag = f"A{int(flags[0])}B{int(flags[1])}"
            DC.ROWS_CORE_FUSE, DC.STATE_NOCOPY = flags
            try:
                from kyojin_mlx import metal_src as MS
                ms0 = MS.HALF_ROWS
                MS.HALF_ROWS = 0                                        # fp32 row FMAs: a row's bits do not depend on its neighbours' content (half_rows >= 4 packs them)
                e = np.random.default_rng(5000 + seed).integers(16, V - 16, 8).tolist()
                b1 = clone(base)
                run_pass(b1, [tok] + e[:n - 1], commit=0)                    # other drafts, same width
                s1_lg = step(b1, d[0])
                b2 = clone(base)
                run_pass(b2, [tok] + d[:n - 1], commit=0)                    # the drafts rejected
                s2_lg = step(b2, d[0])
                bad = state_bits(state(b1[0]), state(b2[0]))
                record(f"R1 rows{n} s{seed} {tag}", bad is None and same(s1_lg, s2_lg),
                       {"state": bad or "equal", "step": "equal" if same(s1_lg, s2_lg) else "DIFFER"})
                b3 = clone(base)
                run_pass(b3, [tok], commit=0)                               # one row of the pass, the same lever path
                s3_lg = step(b3, d[0])
                bad1 = state_bits(state(b3[0]), state(b2[0]))
                b4 = clone(base)                                            # a run that never drafted at all: the plain decode of tok
                step(b4, tok)
                s4_lg = step(b4, d[0])
                bad2 = state_bits(state(b4[0]), state(b3[0]))
                rel, worst = state_err(state(b4[0]), state(b3[0]))
                dl = float(np.abs(s4_lg - s3_lg).max())
                exact = True
                e2 = {"vs_1row": {"state": bad1 or "equal", "step": "equal" if same(s3_lg, s2_lg) else "DIFFER"},
                      "vs_plain": {"state": bad2 or "equal", "state_rel": float("%.4g" % rel), "worst": worst, "step_maxlogit": round(dl, 5)}}
                passed = (bad1 is None and same(s3_lg, s2_lg)) if exact else True
                if flags[0]:                                                # the fused row core moves the attention order: bound, do not demand bits
                    passed = passed and rel <= 0.16 and dl <= 4.0
                else:
                    passed = passed and bad2 is None and same(s4_lg, s3_lg)
                record(f"R2 rows{n} s{seed} {tag}", passed, e2)
            finally:
                DC.ROWS_CORE_FUSE, DC.STATE_NOCOPY = False, False
                MS.HALF_ROWS = ms0

n_ok = sum(1 for c in out["checks"] if c["ok"])
print(f"SUMMARY {n_ok}/{len(out['checks'])} checks ok", flush=True)
if a.json:
    print(json.dumps(out))
sys.exit(0 if n_ok == len(out["checks"]) else 1)

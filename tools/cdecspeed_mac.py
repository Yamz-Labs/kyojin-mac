# SPDX-License-Identifier: Apache-2.0
"""cdecspeed Mac gate: every default-0 decode lever of section 4 of the work ledger, ONE AT A TIME, flag 0 against flag 1 (or its stated value).
Needs Metal. Bounded: one synthetic 2-layer DS4.1 model (tests/decstage_fixture.py, served attention shape, random weights, no pack), peak memory
asserted under 3.5 GiB. This is the single-lever bisect the ledger lists as "never bisected" (opmoe / oploop / opcommit / opffn / verify all-ON),
reduced to what a 2-layer fixture can answer: does the lever change the logits, what does a step cost, what stays allocated.

For each arm: the SAME teacher-forced token sequence is fed (one plain step per token, or verify blocks of --rows tokens) on a fresh cache with all
candidate flags at their default, then the one lever switched on (kyojin_mlx.flags.set_many). Reported per arm:

  logits   identical | first differing position / vocab index / max|d| / argmax flips (against the base run)
  finite   no NaN / inf in the logits (the fp16 saturating clamp of the hidden-stream kernels must keep this True)
  ms/step  median wall ms of a step after --warm steps, and the ratio to the base median
  kept MiB mx.get_active_memory() after the arm minus before it (what the arm leaves allocated), plus the arm's own peak

A control arm (base against base) gives the noise floor of the timing and must be logits-identical. The ms/step of a 2-layer toy model is NOT the
served step: read it as "does this lever cost or save launch time in the same dispatch pattern", never as tok/s. The served decision stays with the
lane A/B of the ledger.

Exit 0: control identical, no arm has a non-finite logit, and no arm differs from the base (a lever claimed exact that differs is the finding).
Exit 1: some arm differs or is non-finite or failed to run (the table says which). Exit 2: no Metal. Exit 3: refused (--max-gib above the 3.5 GiB cap).

  python tools/cdecspeed_mac.py 2>&1 | tee /tmp/cdecspeed_mac.log
  python tools/cdecspeed_mac.py --rows 4 --steps 160 --reps 5 --arms moe_ksplit,dense_views
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

GIB = 1024 ** 3
MIB = 1024 ** 2
CAP_GIB = 3.5

# lever name -> value that switches it on. Generic decode levers that default to 0; the spec_ladder / serve-loop levers act in the engine, not in
# the model step, so a step fixture cannot see them (they stay with the lane A/B).
ARMS = {
    "moe_tgt": 240, "moe_ksplit": 1, "moe_finpar": 1,                   # opmoe: routed GEMV k-split target / by bitrate / parallel combine
    "dense_views": 1,                                                    # opcommit: prefix views, less billing per command buffer
    "router_w16": 1,                                                     # opffn: exact narrow gate copy
    "dec_stage": 1, "dec_stage_glue": 1,                                 # opffn / decstage: six-stage block, mix + expand launches
    "fuse_rows": 0, "half_rows": 2,                                      # fused dense linears OFF (shipped 8: this arm is the shipped lever against its absence) / packed-fp16 row FMAs
    "rows_core_fuse": 1, "rows_state_nocopy": 1, "dec_filter": 1,        # verify-pass core levers
}
BASE_FLAGS = {"kvchain": 1, "ropefuse": 1, "rope_fma": 3}                # the shipped defaults of the hidden-stream kernels, kept in every arm


def existing(arms: dict, FL) -> tuple[dict, list]:
    """(arms the flag table knows and can switch at runtime, names dropped)."""
    ok, gone = {}, []
    for n, v in arms.items():
        f = FL.FLAGS.get(n)
        (ok.__setitem__(n, v) if f is not None and f.runtime else gone.append(n))
    return ok, gone


def compare(base, got):
    """(first differing position or None, first differing vocab index, max abs diff, differing positions, argmax flips)."""
    diff = base != got
    pos = np.where(diff.any(axis=1))[0]
    if not len(pos):
        return None, None, 0.0, 0, 0
    p = int(pos[0])
    v = int(np.where(diff[p])[0][0])
    return p, v, float(np.abs(base - got).max()), int(len(pos)), int((base.argmax(-1) != got.argmax(-1)).sum())


def run_arm(mx, model, RW, FL, defaults: dict, fl: dict, tokens, rows: int, warm: int):
    """-> (logits [n, V] fp32, per-step ms list, kept bytes, arm peak bytes)."""
    FL.set_many(dict(defaults) | BASE_FLAGS | fl)
    mx.clear_cache()
    a0 = mx.get_active_memory()
    mx.reset_peak_memory()
    cache = RW.batch([model.make_cache()])
    out, ms = [], []
    steps = [[int(t)] for t in tokens] if rows <= 1 else [[int(t) for t in tokens[i:i + rows]] for i in range(0, len(tokens), rows)]
    for blk in steps:
        t0 = time.perf_counter()
        if rows <= 1:
            lg = RW.step(model, cache, mx.array(blk))
            mx.eval(lg)
            out.append(np.array(lg)[0].astype(np.float32))
        else:
            p = RW.verify(model, cache, [blk])
            lg = p.logits
            mx.eval(lg)
            out.extend(np.array(lg).astype(np.float32))
            p.commit([len(blk) - 1])
        ms.append((time.perf_counter() - t0) * 1e3)
    kept = mx.get_active_memory() - a0
    return np.stack(out), ms[warm:] or ms, int(kept), int(mx.get_peak_memory())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=96, help="teacher-forced positions (> sliding window 64 so the ring wraps)")
    ap.add_argument("--rows", type=int, default=1, help="1 = plain steps, N > 1 = verify blocks of N tokens")
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per arm (ABBA-free: base, arm, base, arm ...); the median of medians is reported")
    ap.add_argument("--warm", type=int, default=8, help="first steps of a run left out of the timing (kernel compile, first use)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--arms", default="", help="comma list of lever names (default: all of ARMS the flag table has)")
    ap.add_argument("--max-gib", type=float, default=CAP_GIB)
    a = ap.parse_args()
    if a.max_gib > CAP_GIB:
        print(f"refused: --max-gib {a.max_gib} is above the {CAP_GIB} GiB cap")
        return 3
    import mlx.core as mx
    if not mx.metal.is_available():
        print("Metal not available: the decode kernels these levers switch run only on the GPU (run this on the Mac)")
        return 2
    mx.set_memory_limit(int(a.max_gib * GIB))
    import decstage_fixture as FX
    from kyojin_mlx import archs, flags as FL, rows as RW
    from kyojin_mlx import dsv4_core as DC
    arms, gone = existing(ARMS, FL)
    if a.arms:
        want = [s for s in a.arms.split(",") if s]
        arms = {n: arms[n] if n in arms else ARMS[n] for n in want if n in ARMS}
    defaults = {n: FL.FLAGS[n].default for n in arms}                  # every candidate at its shipped default in the base and in the other arms
    model, cfg = FX.build(**FX.REAL)
    RW.enable(model, archs.arch_of({"model_type": "deepseek_v41"}))
    rng = np.random.default_rng(a.seed)
    tokens = rng.integers(1, cfg.vocab_size, a.steps)
    print(f"model: 2 layers, hidden {FX.REAL['hidden_size']}, window {FX.REAL['sliding_window']}, vocab {cfg.vocab_size}; {a.steps} teacher-forced "
          f"{'plain steps' if a.rows <= 1 else f'verify blocks of {a.rows}'}; levers not in this tree's flag table: {gone or 'none'}")
    print(f"rope_fma resolved to {DC.ROPE_FMA} (detect: {DC.ROPE_DETECT.get('form')}, mismatches per form {DC.ROPE_DETECT.get('bad')})")

    base, base_ms, base_kept, _ = run_arm(mx, model, RW, FL, defaults, {}, tokens, a.rows, a.warm)
    bad, rows_out = [], []
    todo = [("control", {})] + [(n, {n: v}) for n, v in arms.items()]
    for name, fl in todo:
        try:
            got, ms, kept, pk = run_arm(mx, model, RW, FL, defaults, fl, tokens, a.rows, a.warm)
            ms_all, bms_all = [statistics.median(ms)], [statistics.median(base_ms)]
            for _ in range(max(0, a.reps - 1)):
                bms_all.append(statistics.median(run_arm(mx, model, RW, FL, defaults, {}, tokens, a.rows, a.warm)[1]))
                ms_all.append(statistics.median(run_arm(mx, model, RW, FL, defaults, fl, tokens, a.rows, a.warm)[1]))
        except Exception as ex:  # noqa: BLE001  (a kernel that does not compile / a shape it refuses is a result)
            print(f"{name:18s} FAILED {type(ex).__name__}: {str(ex)[:200]}")
            bad.append(name)
            continue
        pos, vi, mad, npos, flips = compare(base, got)
        finite = bool(np.isfinite(got).all())
        same = "identical" if pos is None else f"DIFF pos {pos} vocab {vi} max|d| {mad:.3e} ({npos}/{len(got)} pos, argmax flips {flips})"
        m, bm = statistics.median(ms_all), statistics.median(bms_all)
        rows_out.append((name, same, finite, m, m / bm if bm else float("nan"), (kept - base_kept) / MIB, pk / MIB))
        print(f"{name:18s} logits {same:60s} finite {finite}  ms/step {m:7.3f} (x{m / bm:5.3f} of base {bm:.3f})  kept {kept / MIB:+8.2f} MiB (base {base_kept / MIB:+.2f})  peak {pk / MIB:.0f} MiB")
        if pos is not None or not finite:
            bad.append(name)
    peak = mx.get_peak_memory()
    print(f"peak memory {peak / GIB:.3f} GiB (limit {a.max_gib} GiB)")
    assert peak < a.max_gib * GIB, "peak memory over the bound"
    FL.set_many({n: FL.FLAGS[n].default for n in ARMS if n in FL.FLAGS and FL.FLAGS[n].runtime})   # leave the process on the shipped defaults
    print("GATE", "FAIL: " + ", ".join(bad) if bad else "PASS: control and every lever logits-identical and finite")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

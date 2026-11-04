# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced identity gate for kvchain / ropefuse (cmd5b). Needs Metal. Bounded: one synthetic 2-layer DS4.1 model (tests/decstage_fixture.py, the served
attention shape, random weights, no pack), peak memory asserted under 3.5 GiB, nothing timed.

Every arm is fed the SAME token sequence (teacher forcing, no sampling), one plain decode step per token on a fresh cache, so the logits of two arms are comparable
position by position. The base arm runs the stock ops (kvchain 0, ropefuse 0, rope_fma 0). A second base run is the control (must be 0 difference: the gate is
deterministic or it says nothing). Each other arm switches flags with kyojin_mlx.flags.set_many (GEN bumps, rope_fma=3 resolves on the device) and is compared with
the base:

  arm                         flags
  kvchain/rope_fma=0|1|3      kvchain=1 ropefuse=0
  ropefuse/rope_fma=0|1|3     kvchain=0 ropefuse=1
  both/rope_fma=0|1|3         kvchain=1 ropefuse=1

For each arm it prints the FIRST position whose logits differ from the base, the first differing vocab index, the max abs difference, the number of differing
positions, and the number of positions where the argmax changes. Positions: the sequence crosses the sliding window (W = 64 in the fixture, so the ring wraps and
the slot map `wmap` is exercised) and the layer-1 compressor boundaries run every position (ratio 1). Exit code 0 iff the control is 0 and every arm with rope_fma
1 or 3 is 0 (rope_fma=0 is expected to differ: that is the cause the gate demonstrates, cmd5a_REPORT).

  python tools/cmd5_identity_gate.py                    # 96 plain steps, all 9 arms + control
  python tools/cmd5_identity_gate.py --steps 160 --rows 4   # verify blocks of 4 consecutive tokens instead of plain steps
  python tools/cmd5_identity_gate.py --arms both/1,kvchain/3

--rows N > 1 runs RW.verify over N-token blocks (row path of the batched core, bound on the same cache), --rows 1 is the plain step.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mlx.core as mx  # noqa: E402

GIB = 1024 ** 3
ARMS = {"kvchain": {"kvchain": 1, "ropefuse": 0}, "ropefuse": {"kvchain": 0, "ropefuse": 1}, "both": {"kvchain": 1, "ropefuse": 1}}
BASE = {"kvchain": 0, "ropefuse": 0, "rope_fma": 0}


def peak() -> int:
    f = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return int(f())


def run_arm(model, RW, flags, fl, tokens, rows):
    """logits [len(tokens), V] fp32 of a fresh cache fed `tokens` (teacher forced): plain steps, or verify blocks of `rows` tokens committed whole."""
    flags.set_many(dict(BASE) | fl)
    cache = RW.batch([model.make_cache()])
    out = []
    if rows <= 1:
        for t in tokens:
            lg = RW.step(model, cache, mx.array([int(t)]))
            mx.eval(lg)
            out.append(np.array(lg)[0])
    else:
        for i in range(0, len(tokens), rows):
            blk = [int(t) for t in tokens[i:i + rows]]
            p = RW.verify(model, cache, [blk])
            lg = p.logits
            mx.eval(lg)
            out.extend(np.array(lg))
            p.commit([len(blk) - 1])
    return np.stack(out)



def compare(base, got):
    """(first differing position or None, first differing vocab index, max abs diff, differing positions, argmax flips)."""
    diff = base != got
    pos = np.where(diff.any(axis=1))[0]
    if not len(pos):
        return None, None, 0.0, 0, 0
    p = int(pos[0])
    v = int(np.where(diff[p])[0][0])
    return p, v, float(np.abs(base - got).max()), int(len(pos)), int((base.argmax(-1) != got.argmax(-1)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=96, help="teacher-forced positions (> sliding window 64 so the ring wraps)")
    ap.add_argument("--rows", type=int, default=1, help="1 = plain decode steps, N > 1 = verify blocks of N tokens")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--arms", default="", help="comma list of name/rope_fma, e.g. both/1,kvchain/3 (default: all)")
    ap.add_argument("--max-gib", type=float, default=3.5)
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available: the fused kernels run only on the GPU, this gate has nothing to compare on this machine")
        return 2
    mx.set_memory_limit(int(a.max_gib * GIB))
    import decstage_fixture as FX
    from kyojin_mlx import archs, flags, rows as RW
    from kyojin_mlx import dsv4_core as DC
    model, cfg = FX.build(**FX.REAL)
    RW.enable(model, archs.arch_of({"model_type": "deepseek_v41"}))
    W = FX.REAL["sliding_window"]
    rng = np.random.default_rng(a.seed)
    tokens = rng.integers(1, cfg.vocab_size, a.steps)
    print(f"model: 2 layers, hidden {FX.REAL['hidden_size']}, window {W}, vocab {cfg.vocab_size}; {a.steps} teacher-forced "
          f"{'plain steps' if a.rows <= 1 else f'verify blocks of {a.rows}'}; the ring wraps at position {W}")
    arms = [("control", {}, None)]
    want = [s for s in a.arms.split(",") if s] or [f"{n}/{f}" for n in ARMS for f in (0, 1, 3)]
    for s in want:
        n, f = s.split("/")
        arms.append((s, ARMS[n] | {"rope_fma": int(f)}, int(f)))

    base = run_arm(model, RW, flags, {}, tokens, a.rows)
    bad = 0
    for name, fl, fma in arms:
        got = run_arm(model, RW, flags, fl, tokens, a.rows)
        pos, vi, mad, npos, flips = compare(base, got)
        resolved = f" (rope_fma resolved to {DC.ROPE_FMA})" if fma == 3 else ""
        if pos is None:
            print(f"{name:14s} IDENTICAL over {a.steps} positions{resolved}")
        else:
            where = "before the ring wraps" if pos < W else "after the ring wrapped"
            print(f"{name:14s} first diff at position {pos} ({where}), vocab index {vi}, max|d| {mad:.3e}, {npos}/{a.steps} positions differ, "
                  f"argmax flips {flips}{resolved}")
        expect_zero = name == "control" or (fma in (1, 3))
        if expect_zero and pos is not None:
            bad += 1
    pk = peak()
    print(f"peak memory {pk / GIB:.3f} GiB (limit {a.max_gib} GiB)")
    assert pk < a.max_gib * GIB, "peak memory over the bound"
    flags.set_many({"kvchain": 1, "ropefuse": 1, "rope_fma": 3})        # leave the process on the shipped defaults
    print("GATE", "FAIL: " + str(bad) + " arm(s) that must be identical differ" if bad else "PASS: control and every rope_fma 1 / 3 arm identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

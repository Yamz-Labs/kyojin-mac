# SPDX-License-Identifier: Apache-2.0
"""Which rounding does mx.fast.rope use for the rope pair, and does the fused kernels' form agree? (kvchain / ropefuse md5 question)

The fused DS4.1 kernels (dsv4_core: kv_chain, q rope in the attention load, inverse rope in the merge) compute `x1 * c - x2 * s` under
`#pragma METAL fp contract(off)`. If the stock op contracts the products into an fma the two differ in the last fp32 bit and a fp16 result
flips on about 1 element in 10^4, which is enough to move an emitted-token md5 over a few hundred tokens.

This runs mx.fast.rope on random fp16 data with the SAME cos / sin table the fused kernels read (Plan.attn_rope), for the forward and the inverse
rope, and counts the elements where numpy emulations of the three forms (plain / fma first product / fma second product) differ from the stock
output. The form with 0 mismatches is the one to set KYOJIN_ROPE_FMA to (0 = plain). All three non-zero: the cause is not the contraction.

  python tools/rope_contract_probe.py [--n 2048] [--heads 64] [--reps 4] [--yarn]

Needs Metal (fp16 stock op on the GPU). Peak memory < 1 GiB.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx  # noqa: E402

from kyojin_mlx import dsv4_core as C  # noqa: E402
from kyojin_mlx.ref_numpy import rope_pairs_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048, help="positions per rep")
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--p0", type=int, default=3000, help="first position of rep 0")
    ap.add_argument("--yarn", action="store_true", help="yarn-scaled frequencies (the compressing layers) instead of plain")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available: this probe compares against the GPU stock op")
        return 1
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    scaling = {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1} if a.yarn else None
    rope = DeepseekV4RoPE(64, 160000.0 if a.yarn else 10000.0, scaling)
    rng = np.random.default_rng(1)
    D, H, RD = a.dim, a.heads, 64
    bad = {(d, f): 0 for d in ("fwd", "inv") for f in (0, 1, 2)}
    total = 0
    for rep in range(a.reps):
        pos = list(range(a.p0 + rep * a.n * 7, a.p0 + rep * a.n * 7 + a.n))
        plan = C.Plan(pos)
        rt, it = plan.attn_rope(rope)
        mx.eval(rt, it)
        x = (rng.standard_normal((a.n, H, 1, D)) * 0.5).astype(np.float16)
        pa = mx.array(pos, dtype=mx.int32)
        fwd = np.array(rope(mx.array(x), pa))[..., D - RD:]
        inv = np.array(rope(mx.array(x), pa, inverse=True))[..., D - RD:]
        tail = x[..., D - RD:]
        for d, stock, tab in (("fwd", fwd, np.array(rt)), ("inv", inv, np.array(it))):
            t = tab.reshape(a.n, 1, 1, RD)
            for f in (0, 1, 2):
                bad[(d, f)] += int((rope_pairs_ref(tail, t, f) != stock).sum())
        total += tail.size
    print(f"elements per direction: {total}")
    for d in ("fwd", "inv"):
        print(d, "  ".join(f"form{f}: {bad[(d, f)]} ({bad[(d, f)] / total:.2e})" for f in (0, 1, 2)))
    best = [f for f in (0, 1, 2) if bad[("fwd", f)] == 0 and bad[("inv", f)] == 0]
    print("KYOJIN_ROPE_FMA =", best[0] if best else "none matches: the md5 drift is not the rope contraction")
    return 0


if __name__ == "__main__":
    sys.exit(main())

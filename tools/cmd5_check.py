# SPDX-License-Identifier: Apache-2.0
"""cmd5 Mac check (needs Metal; synthetic arrays of the tests' shapes, no model load, peak memory asserted under 3.5 GiB, nothing timed).

1. Probe: for R in 1, 2, 5, 6, 24 the max |error| against float64 of (a) the stock GPU fp32 `x @ w.T` and (b) stagek2.exact_f32_matmul_t, on the 5120-wide
   shapes of test_xgemv_matches_the_f32_matmul. Hypothesis (ULTRA.md:210): (a) ~1e-6 at R=1 and ~1e-3 for R>=2; (b) <= 3e-5 everywhere. Printed, not gated.
2. Kernel: stagek2.xgemv against float64 for fp16 / bf16 / fp32 weights at every R, bound 1e-5 (same bound as the test).
3. pytest on tests/test_dec_stage_k2.py, test_dec_stage_k.py, test_dec_stage.py (needs the mlx_vlm fork for the last) and the CPU tests of cmd5, -rA --tb=long.

Exit 0 only if (b) <= 3e-5 at every R, (2) holds, pytest returns 0 and the two k2 tests of the task are in the PASSED list.
Exit 1 and the failing items named otherwise; exit 2 without Metal. A test_dec_stage test that still fails after this is NOT the fp32 reference: another cause, read its --tb.

  python tools/cmd5_check.py 2>&1 | tee /tmp/cmd5_check.log
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import mlx.core as mx  # noqa: E402

from kyojin_mlx import stagek2 as SK2  # noqa: E402

GIB = 1024 ** 3
RS = (1, 2, 5, 6, 24)
FILES = ["tests/test_dec_stage_k2.py", "tests/test_dec_stage_k.py", "tests/test_dec_stage.py", "tests/test_cmd5_ref.py", "tests/test_cmd5_kernel_refs.py"]
MUST_PASS = ["test_xgemv_reference_is_the_f32_matmul"] + [f"test_xgemv_matches_the_f32_matmul[{d}-{r}]" for d in ("float16", "float32", "bfloat16") for r in RS]


def main() -> int:
    if not mx.metal.is_available():
        print("Metal not available")
        return 2
    mx.set_memory_limit(int(3.5 * GIB))
    bad = []
    print("== probe: max |error| vs float64, K=5120, 512 outputs")
    print(f"{'R':>3} {'stock @':>10} {'exact ref':>10}")
    for R in RS:
        rng = np.random.default_rng(R)
        x = (rng.standard_normal((R, 5120)) * 0.7).astype(np.float16)
        w = (rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)
        want = x.astype(np.float64) @ w.astype(np.float64).T
        stock = mx.array(x).astype(mx.float32) @ mx.array(w).T
        exact = SK2.exact_f32_matmul_t(mx.array(x), mx.array(w))
        mx.eval(stock, exact)
        es, ee = float(np.abs(np.array(stock) - want).max()), float(np.abs(np.array(exact) - want).max())
        print(f"{R:>3} {es:>10.2e} {ee:>10.2e}")
        if ee > 3e-5:
            bad.append(f"exact reference R={R} {ee:.2e}")
    print("== kernel: |xgemv - float64|, bound 1e-5")
    for R in RS:
        for wdt in ("float16", "float32", "bfloat16"):
            rng = np.random.default_rng(R)
            dt = getattr(mx, wdt)
            x = mx.array((rng.standard_normal((R, 5120)) * 0.7).astype(np.float16))
            w0 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
            w1 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
            y0, y1 = SK2.xgemv(x, w0, w1)
            mx.eval(y0, y1)
            xe = np.array(x.astype(mx.float32)).astype(np.float64)
            e = max(float(np.abs(np.array(y).astype(np.float64) - xe @ np.array(w.astype(mx.float32)).astype(np.float64).T).max()) for y, w in ((y0, w0), (y1, w1)))
            print(f"  R={R:>2} {wdt:>8} {e:.2e}{'' if e < 1e-5 else '   <-- KERNEL'}")
            if e >= 1e-5:
                bad.append(f"kernel xgemv R={R} {wdt} {e:.2e}")
    peak = mx.get_peak_memory() / GIB
    print(f"peak memory {peak:.2f} GiB")
    if peak > 3.5:
        bad.append(f"peak {peak:.2f} GiB")
    print("== pytest")
    r = subprocess.run([sys.executable, "-m", "pytest", *FILES, "-q", "-p", "no:cacheprovider", "-rA", "--tb=long"], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-12000:])
    passed = {ln.split("::", 1)[1].split(" ")[0] for ln in out.splitlines() if ln.startswith("PASSED ") and "::" in ln}
    bad += [f"pytest FAILED {ln.split(' ', 1)[1][:150]}" for ln in out.splitlines() if ln.startswith(("FAILED ", "ERROR "))]
    bad += [f"not passed: {t}" for t in MUST_PASS if t not in passed]
    if r.returncode != 0 and not bad:
        bad.append(f"pytest returncode {r.returncode}")
    print("EXIT 0: every check holds" if not bad else "EXIT 1:\n  " + "\n  ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""Which side is wrong in the Metal reds of tests/test_dec_stage_k.py (test_kernels_match_the_references) and tests/test_dec_stage_k2.py (test_xgemv_matches_the_f32_matmul)?
Needs Metal; synthetic inputs of the tests' shapes, peak memory asserted under 3.5 GiB, nothing timed.

For every case it prints three errors against the float64 result: the KERNEL (stagek mix z / stagek2 xgemv), the `mx` matmul REFERENCE the old tests used, and the CPU emulation of the
kernel's reduction order. Reading: kernel ~1e-6 and reference ~1e-3 -> the reference (tensor-unit fp32 matmul with 10-bit operands) was the inexact side and the tests were wrong;
kernel well above ~1e-5 -> the kernel is wrong (the first row printed with that mark is the case to debug).

  python tools/cmd5_kernel_triage.py
  python -m pytest tests/test_dec_stage_k.py tests/test_dec_stage_k2.py tests/test_dec_stage.py tests/test_topk_tree.py -q -p no:cacheprovider -rf --tb=short 2>&1 | tail -60
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mlx.core as mx  # noqa: E402

import kernel_order as KO  # noqa: E402
from kyojin_mlx import stagek as SK, stagek2 as SK2  # noqa: E402

GIB = 1024 ** 3
BOUND = 1e-5


def main() -> int:
    if not mx.metal.is_available():
        print("Metal not available")
        return 2
    mx.set_memory_limit(int(3.5 * GIB))
    spec = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "tests", "test_dec_stage_k.py"))
    tk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tk)
    bad = 0
    print(f"{'case':26s} {'kernel':>10s} {'mx ref':>10s} {'cpu emul':>10s}")
    for D, R in [(512, 1), (512, 6), (4096, 1), (4096, 6), (4096, 24)]:
        x, cpre, fn, scale, base, w, y = tk.inputs(R, D, seed=D + R)
        a = [mx.array(v) for v in (x, cpre, fn, scale, base, w)]
        _, z, _ = SK.mix_stage(*a, tk.EPS_FLAT, tk.EPS_NORM, tk.HC_EPS)
        _, zr, _ = SK.mix_ref(*a, tk.EPS_FLAT, tk.EPS_NORM, tk.HC_EPS)
        mx.eval(z, zr)
        xf = KO.f32(x.reshape(R, 4 * D))
        ex = KO.exact_z(xf, fn)
        ek, er = np.abs(np.array(z) - ex).max(), np.abs(np.array(zr) - ex).max()
        ee = np.abs(KO.mix_z_order(xf, fn) - ex).max()
        flag = "  <-- KERNEL" if ek > BOUND else ""
        bad += ek > BOUND
        print(f"{'mix z D=%d R=%d' % (D, R):26s} {ek:10.2e} {er:10.2e} {ee:10.2e}{flag}")
    for R in (1, 2, 5, 6, 24):
        for wdt in ("float16", "float32", "bfloat16"):
            rng = np.random.default_rng(R)
            dt = getattr(mx, wdt)
            x = mx.array((rng.standard_normal((R, 5120)) * 0.7).astype(np.float16))
            w0 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
            w1 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
            y0, y1 = SK2.xgemv(x, w0, w1)
            r0, r1 = SK2.xgemv_ref(x, w0, w1)
            mx.eval(y0, y1, r0, r1)
            xe = np.array(x.astype(mx.float32)).astype(np.float64)
            ek = er = ee = 0.0
            for yy, rr, ww in ((y0, r0, w0), (y1, r1, w1)):
                wv = np.array(ww.astype(mx.float32))
                ex = xe @ wv.astype(np.float64).T
                ek = max(ek, np.abs(np.array(yy) - ex).max())
                er = max(er, np.abs(np.array(rr) - ex).max())
            ee = np.abs(KO.xgemv_order(KO.f32(xe), KO.f32(np.array(w0.astype(mx.float32)[:16]))) - xe @ np.array(w0.astype(mx.float32)[:16]).astype(np.float64).T).max()
            flag = "  <-- KERNEL" if ek > BOUND else ""
            bad += ek > BOUND
            print(f"{'xgemv R=%d %s' % (R, wdt):26s} {ek:10.2e} {er:10.2e} {ee:10.2e}{flag}")
    f = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    print(f"peak memory {f() / GIB:.3f} GiB (bound 3.5)")
    print("TRIAGE:", "kernel error above 1e-5 in %d case(s): the code is wrong there" % bad if bad else "kernels within 1e-5 of exact everywhere: the old mx-matmul reference was the inexact side")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

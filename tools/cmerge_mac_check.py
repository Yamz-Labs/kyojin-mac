#!/usr/bin/env python3
"""cmerge Mac check: what the merged test branch needs a Metal GPU for. Synthetic arrays only, no model load, nothing timed, memory capped (peak asserted under 3.5 GiB).

1. Lever combinations of the routed prefill GEMM (gate + up, persist form, as pf_moe_nosync builds it) on small synthetic stacks: stock, pf_unroll2, pf_nosync_rowend,
   both, and the 16-row form (pf_bmu16) alone and with the others. unroll / rowend must be bit-identical to stock (exit 1 otherwise); bmu16 is REPORTED (tensor-op tile M
   changes, tools/cpftrellis_gemm.py). A kernel that does not compile is a result: printed, exit 1.
2. The saturation tests that need a GPU: pytest tests/test_expand_sat.py (hidden-stream expand / collapse clamp at the fp16 range) and the Metal tests of tests/test_cpftrellis.py.
3. Nothing about speed: for timing use the per-lever tools in the report's table (they load the model or a 1.7 GiB synthetic stack).

  python tools/cmerge_mac_check.py 2>&1 | tee /tmp/cmerge_mac_check.log

Exit 0: every bit-identity holds and pytest returns 0. Exit 1: a named item failed. Exit 2: no Metal. Exit 3: refused (the memory bound is above --cap-gib).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bench"))

import mlx.core as mx  # noqa: E402

GIB = 2 ** 30
H, I = 1024, 512


def build(E, M, K, pool, seed):
    from bench_moe import synth_stack
    rng = np.random.default_rng(seed)
    pr = rng.dirichlet(np.full(E, 0.4))                                 # skewed routing: some experts empty, some hot
    counts = rng.multinomial(E * M, pr)
    n = int(counts.sum())
    g, u = (synth_stack(H, I, [K] * pool, "mul1", s) for s in (1, 2))
    mx.eval(g.words, u.words)
    phys = np.arange(E) % pool

    def virtual(st):
        off = np.array(st.offset)[:pool][phys]
        off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
        return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]), mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]), st.kt, st.nt, st.cb)
    return virtual(g), virtual(u), counts, n


def md5(y) -> str:
    return hashlib.md5(np.ascontiguousarray(np.array(y)).view(np.uint16).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=64)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--cap-gib", type=float, default=3.5)
    ap.add_argument("--skip-pytest", action="store_true")
    a = ap.parse_args()
    n_pairs = a.e * a.m
    bound = 2 * a.pool * (H * I * a.k // 8) + 2 * n_pairs * H * 2 + 2 * n_pairs * I * 2 * 2 + (64 << 20)
    print(f"E={a.e} mean rows {a.m} K={a.k} pool={a.pool} pairs {n_pairs}  memory bound {bound / GIB:.3f} GiB (cap {a.cap_gib})", flush=True)
    if bound > a.cap_gib * GIB:
        print("refused: the bound is above the cap; lower --e / --m")
        return 3
    if not mx.metal.is_available():
        print("Metal not available (run this on the Mac)")
        return 2
    mx.set_memory_limit(int(3.0 * GIB))
    mx.set_cache_limit(1 << 29)
    from kyojin_mlx import metal_nax as NX
    g, u, counts, n = build(a.e, a.m, a.k, a.pool, 1)
    rng = np.random.default_rng(0)
    eids = mx.array(np.arange(a.e, dtype=np.int32))
    A = mx.array(NX.permute_k(rng.standard_normal((2 * n, H)).astype(np.float16)))
    cl = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    ro_glued = mx.array(np.concatenate([cl[:-1], n + cl]))              # pf_moe_nosync's range: the up block glued to the gate block
    ro_end = mx.array(np.concatenate([cl, n + cl]))                     # pf_nosync_rowend: each range closed by its own end
    mx.eval(A, ro_glued, ro_end)
    cfg32, cfg16 = (32, 2, 32, 64), (16, 4, 32, 64)
    arms = {"stock": (cfg32, False, False), "unroll": (cfg32, True, False), "rowend": (cfg32, False, True), "unroll+rowend": (cfg32, True, True),
            "bmu16": (cfg16, False, False), "bmu16+unroll": (cfg16, True, False), "bmu16+rowend": (cfg16, False, True), "bmu16+unroll+rowend": (cfg16, True, True)}
    old = NX.UNROLL2
    ref, bad = None, []
    print(f"{'arm':22s} {'md5':34s} bit-identical to stock")
    for name, (cfg, un, re_) in arms.items():
        NX.UNROLL2 = un
        try:
            y = NX.expert_gemm((g, u), eids, ro_end if re_ else ro_glued, A, 2 * n, 0, cfg, persist=True, rowend=re_)
            h = md5(y)
            yv = np.array(y).view(np.uint16)
        except Exception as ex:  # noqa: BLE001
            print(f"{name:22s} FAILED {type(ex).__name__}: {str(ex)[:240]}")
            bad.append(name)
            continue
        finally:
            NX.UNROLL2 = old
        if ref is None:
            ref = yv
            print(f"{name:22s} {h}  (reference)")
            continue
        eq = bool(np.array_equal(yv, ref))
        print(f"{name:22s} {h}  {eq}")
        if not eq and not name.startswith("bmu16"):
            bad.append(name)                                            # unroll / rowend are claimed bit-identical; bmu16 is only reported
    rc = 1 if bad else 0
    if not a.skip_pytest:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_expand_sat.py", "tests/test_cpftrellis.py", "tests/test_cmerge.py"], cwd=HERE)
        if r.returncode:
            bad.append("pytest")
            rc = 1
    peak = mx.get_peak_memory() / GIB
    print(f"peak {peak:.3f} GiB (cap {a.cap_gib})")
    assert peak < a.cap_gib, f"peak {peak:.2f} GiB above the cap"
    print("FAILED: " + ", ".join(bad) if bad else "ALL OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())

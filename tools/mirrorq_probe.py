# SPDX-License-Identifier: Apache-2.0
"""mirrorq probe: what one kv-source layer's per-step slab decode costs on this device, against the dense gather the attention does today and against the
MLX-ops decode (cdecpack.compact's form), and what one whole-layer unpack costs per key row (the per-turn price of the stack build).

Bounds: 64K keys per array and 1 GiB per array (the probe's own limit, the task's limit). The pool layers of DS4.1 are D = 512 (block 16, e4m3), so a
64K-key layer is 16 MB of codes + 2 MB of scales + 64 MB dense: the numbers scale linearly in the key count (a decode reads exactly the rows it names, an
unpack reads every row).

    python3 tools/mirrorq_probe.py            # this Mac, Metal
    python3 tools/mirrorq_probe.py --json out.json

What it prints per arm (median of --reps, ms):
  decode_kernel   mirrorq.decode_rows: the K named rows from the codes, one fused launch
  decode_mlx      the same rows through MLX ops (the shape cdecpack.compact has)
  gather_dense    the K named rows out of the dense table the flag 0 path reads (what the attention's own gather costs in bytes moved)
  unpack_rows     one whole-layer unpack, per row (the option-a / prefill / second-turn cost)
Exit code 0 always: this is a probe, it decides nothing on its own.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

KFULL = 65536                                       # keys per layer the probe builds (the task's cap: <= 64K keys)
D, BLOCK = 512, 16


def _layer(seed=0):
    from kyojin_mlx import fp4pack
    x = mx.array((np.random.default_rng(seed).standard_normal((1, KFULL, D)) * 3.0).astype(np.float16))
    c, s = fp4pack.pack(x, BLOCK, True)
    mx.eval(c, s)
    return c, s


def _ev(r):
    if isinstance(r, (tuple, list)):
        mx.eval(*r)
    elif r is not None:
        mx.eval(r)


def _time(fn, reps):
    _ev(fn())
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ev(fn())
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--keys", type=int, default=KFULL)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--rows", type=int, default=1, help="rows of the pass (1 = plain decode, 6 = a verify round)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if not mx.metal.is_available():
        print("no Metal: nothing to measure", file=sys.stderr)
        return 0
    from kyojin_mlx import cdecpack as CD, fp4pack, fp4pool, mirrorq as MQ
    from kyojin_mlx.fp4pool import PoolSlots

    codes, scales = _layer()
    s = PoolSlots([None])
    s.set_raw(0, fp4pool._Pk(codes, scales, a.keys, mx.float16))
    v = MQ.pool_view(s, 0)
    dense = fp4pack.unpack(codes[:, :a.keys], scales[:, :a.keys], BLOCK, True, mx.float16)
    mx.eval(dense)
    rng = np.random.default_rng(1)
    idx = mx.array(np.sort(rng.choice(a.keys, (a.rows, a.topk), replace=False), axis=1).astype(np.int32))
    nsel = int(idx.size)

    res = {
        "keys": a.keys, "rows": a.rows, "topk": a.topk, "selected": nsel,
        "decode_kernel_ms": _time(lambda: MQ.decode_rows(v, idx), a.reps),
        "compact_ms": _time(lambda: CD.compact(v, idx), a.reps),
        "gather_dense_ms": _time(lambda: mx.take(dense, idx, axis=1), a.reps),
        "unpack_rows_ms": _time(lambda: fp4pack.unpack(codes[:, :a.keys], scales[:, :a.keys], BLOCK, True, mx.float16), a.reps),
    }
    # the served attention kernel on the same pooled rows: the dense pool (flag 0) against the decoded slab (mirrorq)
    from kyojin_mlx import dsv4_core as DC
    H, Wc = 4, 128
    q = mx.array((rng.standard_normal((a.rows, H, D)) * 0.1).astype(np.float16))
    win = mx.array((rng.standard_normal((a.rows, Wc, D)) * 0.1).astype(np.float16))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    pos = mx.array(np.r_[np.arange(5, 5 + a.rows * 2, 2)], dtype=mx.int32)
    sess = mx.zeros((a.rows,), dtype=mx.uint32)
    slab, remap = MQ.decode_rows(v, idx)
    mx.eval(slab, remap)

    def attn_dense():
        return DC.attend(q, win, dense, idx, pos, sink, D ** -0.5, a.topk, sess=sess)

    def attn_slab():
        return DC.attend(q, win, slab, remap, pos, sink, D ** -0.5, a.topk, sess=sess)

    def attn_both():                                    # what a step costs with the flag on: decode then attend
        r, m = MQ.decode_rows(v, idx)
        return DC.attend(q, win, r, m, pos, sink, D ** -0.5, a.topk, sess=sess)

    res["attend_dense_pool_ms"] = _time(attn_dense, a.reps)
    res["attend_slab_ms"] = _time(attn_slab, a.reps)
    res["attend_decode_plus_slab_ms"] = _time(attn_both, a.reps)
    res["attend_delta_ms"] = res["attend_decode_plus_slab_ms"] - res["attend_dense_pool_ms"]
    # the floor of ONE evaluated op on this device: every arm above is a handful of these, so the delta is a launch count here, not a kernel time
    one = mx.array([[0]], dtype=mx.int32)
    res["eval_floor_ms"] = _time(lambda: mx.take(dense, one, axis=1), a.reps)
    res["slab_is_the_same_rows"] = bool(mx.array_equal(
        mx.take(dense, mx.where(remap >= 0, idx, 0), axis=1),
        mx.take(slab, mx.maximum(remap, 0), axis=1)).item())
    res["decode_kernel_per_row_us"] = res["decode_kernel_ms"] * 1e3 / nsel
    res["unpack_per_row_us"] = res["unpack_rows_ms"] * 1e3 / a.keys
    # the key shape (D = 128, block 32, ue8m0): only the whole-layer unpack matters for a turn (the stack build), and it must agree with the pool row
    kx = mx.array((np.random.default_rng(2).standard_normal((1, a.keys, 128)) * 3.0).astype(np.float16))
    kc, ks = fp4pack.pack(kx, 32, False)
    mx.eval(kc, ks)
    res["unpack_key_rows_ms"] = _time(lambda: fp4pack.unpack(kc, ks, 32, False, mx.float16), a.reps)
    res["slab_mb"] = nsel * D * 2 / 2 ** 20
    res["dense_layer_mb"] = a.keys * D * 2 / 2 ** 20
    res["codes_layer_mb"] = a.keys * (D // 2 + D // BLOCK) / 2 ** 20
    res["hits"] = dict(MQ.HITS)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

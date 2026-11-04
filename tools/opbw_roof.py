#!/usr/bin/env python3
"""opbw: the real GPU read roof of this box with small bounded Metal kernels, plus the fp16 matmul peak.

Every kernel below is a read-only stream: the sum of the words each thread loaded is written once per thread, so no load can be removed.
All buffers live in one allocation (--gb, default 3.2, the 4 GiB process cap), filled with a multiplicative hash; the first run of every arm is
dropped (page-in, clock ramp) and the median of --n is reported.

Arms:
  seq        thread-strided uint4 stream (the classic read roof)
  tile       one threadgroup reads a contiguous 16 KiB tile (the shape the GEMV/fused kernels use per threadgroup)
  stride96   per-lane 16-byte loads 96 B apart, 768 B per simdgroup group of 8 lanes (the EXL3 decode GEMV's 12..96 B run stride)
  stride768  per-lane 16-byte loads 768 B apart (one trellis word per block column)
  gather     one 96 B row out of a random row id per lane group (an expert bank gather)
  offsets    the seq kernel at +0 / +128 B / +4 KiB / +2 MiB inside the buffer (128-byte line and page placement)
  mm16       fp16 matmul peak (8192x5120 @ 5120x2304), TFLOPS
Sweeps: threadgroup 64/128/256/512/1024 rows x grid 4k/64k/1M threads for seq; the other arms run at the best seq geometry.

usage: opbw_roof.py [--gb 3.2] [--n 9] [--json F] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

HDR = "uint tid = thread_position_in_grid.x, nt = threads_per_grid.x; uint4 acc = uint4(0);\n"
P = "device const uint4* p = (device const uint4*)((device const char*)inp + OFF);\n"   # OFF is a template constant (an input would be a buffer)
# `off` is a byte offset into the buffer
SEQ = HDR + P + "for (ulong i = tid; i < N4; i += nt) acc ^= p[i]; out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;"
# one threadgroup walks a contiguous 16 KiB tile at a time (the per-threadgroup slab shape of the weight kernels)
TILE = ("uint tid = thread_position_in_grid.x, nt = threads_per_grid.x; uint tg = tid / tgn, lt = tid % tgn; uint4 acc = uint4(0);\n"
        "device const uint4* p = (device const uint4*)((device const char*)inp + OFF); ulong tiles = N4 / tpn;\n"
        "for (ulong t = tg; t < tiles; t += nt / tgn) { ulong base = t * tpn; for (uint j = lt; j < tpn; j += tgn) acc ^= p[base + j]; }\n"
        "out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;")
# 16-byte loads at a fixed byte stride (STRIDE / 16 uint4 apart)
STRIDE = ("uint tid = thread_position_in_grid.x, nt = threads_per_grid.x; uint4 acc = uint4(0);\n"
          "device const uint4* p = (device const uint4*)((device const char*)inp + OFF); ulong n = N4 / (STRIDE / 16);\n"
          "for (ulong i = tid; i < n; i += nt) acc ^= p[i * (STRIDE / 16)];\n"
          "out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;")
# one 96 B row (6 uint4) per index, row id = hash(i) % rows: the expert-bank gather shape
GATHER = ("uint tid = thread_position_in_grid.x, nt = threads_per_grid.x; uint4 acc = uint4(0);\n"
          "device const uint4* p = (device const uint4*)((device const char*)inp + OFF); ulong rows = N4 / 6;\n"
          "for (ulong i = tid; i < rows; i += nt) { ulong r = (i * 2654435761ul) % rows; for (uint j = 0; j < 6; ++j) acc ^= p[r * 6 + j]; }\n"
          "out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;")


def build(name, src, **tpl):
    return mx.fast.metal_kernel(name=name, input_names=["inp"], output_names=["out"], source=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=3.2)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.gb > 3.5:
        raise SystemExit("opbw_roof: --gb > 3.5 refused (the 4 GiB process cap includes MLX temporaries)")
    nbytes = int(a.gb * 1e9) // 16 * 16
    n4 = nbytes // 16
    # pseudo-random words, built in 512 MiB pieces so the fill temporaries stay small
    piece = 1 << 27
    parts = []
    for s in range(0, nbytes // 4, piece):
        e = min(s + piece, nbytes // 4)
        p = (mx.arange(s, e, dtype=mx.uint32) * mx.array(2654435761, dtype=mx.uint32)) ^ mx.array(0x9E3779B9, dtype=mx.uint32)
        mx.eval(p)
        parts.append(p)
    buf = mx.concatenate(parts)
    del parts
    mx.eval(buf)
    mx.clear_cache()
    zeros = mx.zeros((0,), dtype=mx.uint32)
    res = {"gb": nbytes / 1e9, "device": mx.device_info().get("device_name"), "arms": {}}

    k = {n: build(n, s) for n, s in (("seq", SEQ), ("tile", TILE), ("s96", STRIDE), ("s768", STRIDE), ("gath", GATHER))}
    del zeros

    def run(kern, threads, tg, off=0, **tpl):
        out = kern(inputs=[buf.view(mx.uint32)], template=list(tpl.items()) + [("OFF", int(off))], grid=(threads, 1, 1),
                   threadgroup=(tg, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.uint32])[0]
        mx.eval(out)
        return out

    def time_arm(fn, n=None):
        fn()                                                      # warm-up (also pages in)
        ts = []
        for _ in range(n or a.n):
            t0 = time.perf_counter()
            fn()
            mx.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    # ---- seq sweep: threadgroup x grid
    tgs = [64, 256] if a.quick else [64, 128, 256, 512, 1024]
    grids = [4096, 65536] if a.quick else [4096, 65536, 1 << 20]
    best = None
    for tg in tgs:
        for th in grids:
            dt = time_arm(lambda: run(k["seq"], th, tg, 0, N4=n4))
            gbs = nbytes / dt / 1e9
            res["arms"][f"seq_tg{tg}_th{th}"] = gbs
            print(f"seq  tg {tg:4d} threads {th:8d}: {gbs:7.0f} GB/s")
            if best is None or gbs > best[0]:
                best = (gbs, tg, th)
    _, tg, th = best
    print(f"best seq: tg {tg}, threads {th} -> {best[0]:.0f} GB/s\n", flush=True)
    res["best"] = {"tg": tg, "threads": th, "gbs": best[0]}

    # ---- tile: one threadgroup per contiguous 16 KiB tile
    for tpn in (1024, 256):                                       # 16 KiB and 4 KiB per threadgroup
        dt = time_arm(lambda: run(k["tile"], th, tg, 0, N4=n4, tgn=tg, tpn=tpn))
        gbs = nbytes / dt / 1e9
        res["arms"][f"tile_tgn{tg}_tpn{tpn}"] = gbs
        print(f"tile tgn {tg:4d} tile {tpn * 16 // 1024:3d} KiB: {gbs:7.0f} GB/s")

    # ---- fixed strides (sector/line efficiency of the decode pattern)
    for stride in (96, 768):
        kern = k["s96"] if stride == 96 else k["s768"]
        dt = time_arm(lambda: run(kern, th, tg, 0, N4=n4, STRIDE=stride))
        touched = (n4 // (stride // 16)) * 16                 # bytes the loads ask for
        lines = (n4 // (stride // 16)) * 128                  # bytes those loads drag in (one 128 B line each)
        res["arms"][f"stride{stride}"] = {"useful_gbs": touched / dt / 1e9, "line_gbs": lines / dt / 1e9}
        print(f"stride {stride:4d} B: {touched / dt / 1e9:7.0f} GB/s asked, {lines / dt / 1e9:7.0f} GB/s dragged "
              f"({touched / lines * 100:.0f} % of a line used)")

    # ---- gather: one 96 B row per lane group out of a random row
    dt = time_arm(lambda: run(k["gath"], th, tg, 0, N4=n4))
    res["arms"]["gather96"] = 96.0 * (n4 // 6) / dt / 1e9
    print(f"gather 96 B rows: {96.0 * (n4 // 6) / dt / 1e9:7.0f} GB/s useful, "
          f"{128.0 * (n4 // 6) / dt / 1e9:7.0f} GB/s of lines")

    # ---- buffer placement: the same stream at four offsets (128 B line and 2 MiB page scale)
    for off in (0, 128, 4096, 1 << 21):
        dt = time_arm(lambda: run(k["seq"], th, tg, off, N4=(nbytes - off - 16) // 16))
        gbs = (nbytes - off - 16) / dt / 1e9
        res["arms"][f"off{off}"] = gbs
        print(f"seq at +{off:8d} B: {gbs:7.0f} GB/s")

    # ---- fp16 matmul peak (the prefill roof the MoE GEMM is compared with)
    x = mx.random.normal((8192, 5120)).astype(mx.float16)
    w = mx.random.normal((5120, 2304)).astype(mx.float16)
    mx.eval(x, w)
    y = x @ w
    mx.eval(y)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = x @ w
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    dt = ts[len(ts) // 2]
    tflops = 2 * 8192 * 5120 * 2304 / dt / 1e12
    res["matmul_tflops"] = tflops
    print(f"\nfp16 matmul 8192x5120x2304: {tflops:.1f} TFLOPS ({dt * 1e3:.2f} ms)")
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()

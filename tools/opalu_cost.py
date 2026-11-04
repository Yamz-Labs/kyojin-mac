#!/usr/bin/env python3
"""opalu: GPU instruction cost per decoded weight in the prefill expert GEMM (metal_nax.coop_source), from the generated source.

Counts the scalar instructions the Metal compiler is handed per 16-column trellis tile (8 weights per lane) and per decoded weight,
for every bit width of the pack (2..6) and every block config in use. The decoder bodies themselves are counted from the header text
(one line per emitted instruction in the MSL), so the table is an instruction count of the source, not a disassembly: the compiled
kernel is checked separately by tools/opalu_dec.py (time) and the GPU counter probe if one is available.

usage: tools/opalu_cost.py [--json F] [--src]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from kyojin_mlx import metal_nax as NX  # noqa: E402
from kyojin_mlx import metal_src as S    # noqa: E402

# Instruction count of one decode of one 16-bit state (the `dec_<cb>(s)` body), counted op by op on the MSL.
# Decoder bodies: metal_src.HEADER (dec_mul1) and HEADER_MOE_ROOF (dec_mul1r1 / dec_mul1r2).
DEC_OPS = {
    # name: imad / shr / and-bfi / add / fp16 ops, one instruction each (counted on the MSL of metal_src.HEADER and HEADER_MOE_ROOF)
    "mul1":   {"imad": 1, "shr": 2, "and": 3, "add": 3, "fp": 1},   # h=s*C; t=(h&m)+((h>>8)&m); ushort(t)+ushort(t>>16)+0x6400; fma
    "mul1r1": {"imad": 2, "shr": 1, "and": 3, "add": 1, "fp": 1},   # + lane sum folded into one imad (t*0x10001+0x64000000)
    "mul1r2": {"imad": 2, "shr": 1, "and": 2, "add": 0, "fp": 1},   # + byte pairs by one subtract-imad (h - 255*((h>>8)&m))
}
# static operand extraction per weight (extract_bits of a 32-bit word at a compile-time offset) and the per-tile window build
EXTRACT_OPS = 1
WINDOW_OPS = 11     # shi (shl,shr,shr,or), slo (shl,shr,shr,or), smid (shr + insert_bits) per tile of 8 weights


def dec_ops(cb: str) -> int:
    o = DEC_OPS[cb]
    return o["imad"] + o["shr"] + o["and"] + o["add"] + o["fp"]


def count_src(src: str, cb: str) -> dict:
    """Counts of the token classes in one generated kernel body (whole body, all K bodies) and the per-weight arithmetic."""
    body = src[src.index("if (K ==") if "if (K ==" in src else 0:]
    return dict(dec_calls=len(re.findall(rf"dec_{cb}\(", body)),
                extract=len(re.findall(r"extract_bits\(", body)),
                insert=len(re.findall(r"insert_bits\(", body)),
                shifts=len(re.findall(r"<<|>>", body)),
                ands=len(re.findall(r"&\s*0x", body)),
                runs=len(re.findall(r"op\.run\(", body)))


def tile_cost(cb: str) -> dict:
    """Instructions per (lane, 16-column tile, k step) = 8 decoded weights, and per weight."""
    d = dec_ops(cb)
    per_tile = WINDOW_OPS + 8 * (EXTRACT_OPS + d)
    return dict(decoder_ops_per_weight=d, extract_ops_per_weight=EXTRACT_OPS, window_ops_per_tile=WINDOW_OPS,
                ops_per_tile=per_tile, weights_per_tile=8, ops_per_weight=per_tile / 8.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--src", action="store_true", help="print the decode block of K=3, 64-row cfg")
    a = ap.parse_args()
    cfgs = {"16 rows (16,1,32,64)": (16, 1, 32, 64), "32 rows (32,1,32,64)": (32, 1, 32, 64),
            "64 rows (32,2,32,64)": (32, 2, 32, 64), "128 rows (32,4,16,64)": (32, 4, 16, 64)}
    res = {"dec_ops": DEC_OPS, "tiles": {}, "kernels": {}}
    print(f"{'decoder':8s} {'int ops':>7s} {'fp':>3s} {'total':>5s} {'per tile (8 w)':>14s} {'per weight':>10s}")
    for cb in ("mul1", "mul1r1", "mul1r2"):
        t = tile_cost(cb)
        res["tiles"][cb] = t
        print(f"{cb:8s} {dec_ops(cb) - 1:7d} {1:3d} {dec_ops(cb):5d} {t['ops_per_tile']:14.0f} {t['ops_per_weight']:10.2f}")
    print()
    # one k step of one lane = bns/16 tiles = 8 weights each; the kernel does nk x nt_sg tiles per loop iteration
    print(f"{'bit width':>9s} {'tiles/step':>10s} {'w/step':>6s} {'ops/step':>8s} | " +
          " | ".join(f"{k.split()[0]:>6s} {v[2]:>2d}" for k, v in cfgs.items()))
    rows = {}
    for K in (2, 3, 4, 5, 6):
        line = []
        for name, (bmu, nmb, bns, bks) in cfgs.items():
            nt_sg, nk = bns // 16, bks // 16
            tiles = nk * nt_sg
            t = tile_cost("mul1")
            line.append(f"{t['ops_per_weight'] * 8 * tiles:7.0f}")
            rows.setdefault(name, {})[K] = dict(tiles_per_step=tiles, weights_per_step=8 * tiles,
                                                ops_per_step=t["ops_per_weight"] * 8 * tiles, bmu=bmu, nmb=nmb, bns=bns, bks=bks)
        print(f"K={K:7d} {rows[list(cfgs)[2]][K]['tiles_per_step']:10d} {rows[list(cfgs)[2]][K]['weights_per_step']:6d}"
              f" {rows[list(cfgs)[2]][K]['ops_per_step']:8.0f} | " + " | ".join(f"{x:>9s}" for x in line))
    res["kernels"] = rows
    # cross-check against the generated source
    for K in (2, 3):
        for name, (bmu, nmb, bns, bks) in cfgs.items():
            src = NX.coop_source((K,), "mul1", bmu, nmb, bns, bks)
            c = count_src(src, "mul1")
            nk, nt_sg = bks // 16, bns // 16
            per_iter = c["dec_calls"]
            print(f"src K={K} {name:22s} dec_calls {c['dec_calls']:3d} extract {c['extract']:3d} "
                  f"shift/rot {c['shifts']:4d} ands {c['ands']:3d} op.run {c['runs']:2d}"
                  f"   (one loop iteration: {nk * nt_sg * 8} weights/lane)")
            res["kernels"][name][K]["src_counts"] = c
    print("\nper weight (mul1, 64-row block): "
          f"{tile_cost('mul1')['ops_per_weight']:.2f} scalar instr; mul1r2 {tile_cost('mul1r2')['ops_per_weight']:.2f}")
    # what one decoded byte costs at each bit width of the pack (the decode does not depend on K; the weight it produces does)
    kinds = [("routed hot", 3), ("routed cold", 2), ("mtp", 4), ("shared expert", 5), ("attention/dense", 5), ("lm head", 6)]
    print(f"\n{'kind':16s} {'bits':>4s} {'instr/weight':>12s} {'weights/byte':>12s} {'int instr/GB':>12s} (mul1) {'(mul1r2)':>9s}")
    res["kinds"] = {}
    for name, K in kinds:
        wb = K / 8.0
        res["kinds"][name] = dict(bits=K, instr_per_weight=tile_cost("mul1")["ops_per_weight"], weights_per_byte=1.0 / wb,
                                  int_instr_per_gb=tile_cost("mul1")["ops_per_weight"] / wb * 1e9)
        print(f"{name:16s} {K:4d} {tile_cost('mul1')['ops_per_weight']:12.2f} {1.0 / wb:12.2f} "
              f"{tile_cost('mul1')['ops_per_weight'] / wb * 1e9:12.2e} {tile_cost('mul1r2')['ops_per_weight'] / wb * 1e9:9.2e}")
    if a.src:
        src = NX.coop_source((3,), "mul1", 32, 2, 32, 64)
        i = src.index("const uint shi =")
        print(src[i:i + 900])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

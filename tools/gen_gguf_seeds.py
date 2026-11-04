#!/usr/bin/env python3
"""Extract the seed lists of ggml's iq2xs_init_impl / iq3xs_init_impl (the `kgrid_*` uint16 arrays) from ggml-quants.c and write
kyojin_mlx/foreign_seeds.py.  Mechanical text extraction, nothing typed by hand.  usage: gen_gguf_seeds.py <ggml-quants.c> <llama.cpp commit> > out"""
import re
import sys

NAMES = ["kgrid_2bit_256", "kgrid_2bit_512", "kgrid_2bit_1024", "kgrid_1bit_2048", "kgrid_256", "kgrid_512"]


def main(path, commit):
    src = open(path).read()
    out = ['# SPDX-License-Identifier: MIT',
           f'"""Seed lists of ggml iq2xs_init_impl / iq3xs_init_impl (ggml/src/ggml-quants.c, llama.cpp commit {commit}, MIT), extracted by tools/gen_gguf_seeds.py."""',
           f'COMMIT = "{commit}"']
    for n in NAMES:
        m = re.search(r"static const uint16_t " + n + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        v = [int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\d+", body)]
        out.append(f"{n.upper()} = (")
        for i in range(0, len(v), 16):
            out.append("    " + ", ".join(map(str, v[i:i + 16])) + ",")
        out.append(")")
    print("\n".join(out))


main(sys.argv[1], sys.argv[2])

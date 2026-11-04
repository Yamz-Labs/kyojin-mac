"""Hash the Metal kernel sources the engine builds with every lever at its default, over a grid of configurations. CPU only, no model, no GPU.

Run it from two trees and diff the output: `python tools/cmerge_srchash.py > a.txt` in a perf/integ checkout and in the merged checkout. Equal lines mean the
default-off kernels are byte-identical (the compile input of the served path did not move). Only arguments both trees accept are passed.
Also hashes every module-level string constant of the hidden-stream writers (fuse, pf_glue, suglue, stagek, stagek2, glue_fuse, gluefuse, ds_pf_mhc), so a
merge that loses a clamp shows as a changed line.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("PYTHONHASHSEED", None)

from kyojin_mlx import metal_nax as NX, metal_nax_gu as GU  # noqa: E402


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _try(fn) -> str:
    try:
        return h(fn())
    except (AssertionError, KeyError, ValueError) as e:
        return "ERR " + type(e).__name__


def main() -> int:
    n = 0
    for Ks, cb, cfg, out_half, mode, nproj, persist, pipe in itertools.product(
            [(3,), (2, 3, 5), (4, 6)], ["mul1r2", "mul1"], [(32, 2, 32, 64), (16, 4, 32, 64), (32, 1, 32, 32), (32, 2, 16, 64)],
            [True, False], ["full", "ks"], [1, 2], [False, True], [False, True]):
        bmu, nmb, bns, bks = cfg
        try:
            src = NX.coop_source(Ks, cb, bmu, nmb, bns, bks, out_half, mode, nproj, True, "", False, persist, pipe)
        except (AssertionError, KeyError, ValueError) as e:
            print(f"coop {Ks} {cb} {cfg} oh={out_half} {mode} p{nproj} pl={persist} pp={pipe} ERR {type(e).__name__}")
            continue
        print(f"coop {Ks} {cb} {cfg} oh={out_half} {mode} p{nproj} pl={persist} pp={pipe} {h(src)}")
        n += 1
    for Ks, cfg, nproj in itertools.product([(3,), (2, 3, 5)], [(32, 2, 32, 64), (16, 4, 32, 64)], [1, 2]):
        print(f"coop_rm {Ks} {cfg} p{nproj} {_try(lambda: NX.coop_rm_source(Ks, 'mul1r2', *cfg, True, nproj))}")
    for Kg, Ku, cfg, rm in itertools.product([3, 5], [3, 4], [(32, 2, 32, 64), (16, 4, 32, 64)], [False, True]):
        print(f"gu {Kg},{Ku} {cfg} rm={rm} {_try(lambda: GU.gu_source(Kg, Ku, 'mul1r2', *cfg, 7.0, True, True, 1.702, 1.0, True, rm))}")
    import importlib
    for mod in ("fuse", "pf_glue", "suglue", "stagek", "stagek2", "glue_fuse", "gluefuse", "ds_pf_mhc", "ablit"):
        try:
            m = importlib.import_module("kyojin_mlx." + mod)
        except Exception as e:  # noqa: BLE001
            print(f"mod {mod} IMPORT {type(e).__name__}")
            continue
        strs = sorted((k, v) for k, v in vars(m).items() if isinstance(v, str) and len(v) > 80)
        print(f"mod {mod} {len(strs)} strings {h(repr(strs))} clamp65504={sum(v.count('65504') for _, v in strs)}")
    print(f"coop sources hashed: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

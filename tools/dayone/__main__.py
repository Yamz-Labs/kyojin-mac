# SPDX-License-Identifier: Apache-2.0
"""python -m tools.dayone [NAME|CONFIG.json ...] [--all] [--flags off|on|both] [--json OUT]

NAME is a synthetic family of tools/dayone/shapes.py; CONFIG.json is any config.json (its sizes are used as given: pass a small one).
Prints one table: phase x family, pass / FAIL / n/a with the first error line."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import engine, shapes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.dayone")
    ap.add_argument("what", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--flags", default="both", choices=("off", "on", "both"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    todo = {}
    for w in (list(shapes.SHAPES) if a.all or not a.what else a.what):
        if w in shapes.SHAPES:
            todo[w] = shapes.SHAPES[w]
        else:                                                   # a config.json path: the family is named after the file (a path cannot be a temp-dir prefix)
            todo[os.path.splitext(os.path.basename(w))[0]] = json.load(open(w))
    out = {}
    for name, cfg in todo.items():
        for fl in (("off", "on") if a.flags == "both" else (a.flags,)):
            out[f"{name}|{fl}"] = engine.run(name, cfg, fl)
    for k, r in out.items():
        print(f"== {k}")
        for ph, (st, d) in r.items():
            print(f"  {ph:10s} {st:5s} {d}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 1 if any(s == "FAIL" for r in out.values() for s, _ in r.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

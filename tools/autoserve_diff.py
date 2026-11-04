#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diff the effective serve options of an old explicit command line against the pack-derived defaults.

    python tools/autoserve_diff.py --model ~/ds41/model \
        --old "--sessions 4 --ctx 1048576 --prefill-chunk 8192 ... --kv-disk /old/dir"

The disk directory itself is deployment-specific (the default lives in the user cache, keyed by the pack); it is compared as
on/off, everything else value by value. Exit 0 = empty diff.
"""
import argparse
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kyojin_mlx.serve import autoconfig as AC, server   # noqa: E402


def resolve(argv):
    ns = server.build_parser().parse_args(argv)
    if ns.model is None:
        ns.model = ns.model_opt
    return AC.effective_config(AC.resolve_serve_args(ns))


def comparable(cfg):
    out = {k: v for k, v in cfg.items() if k != "kv_disk"}
    out["kv_disk"] = "on" if cfg.get("kv_disk") else "off"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--old", required=True, help="the old flag string (without the model path)")
    a = ap.parse_args()
    old = resolve([a.model] + shlex.split(a.old))
    new = resolve([a.model, "--print-config"])               # just flags the resolver; no model load either way
    old, new = comparable(old), comparable(new)
    rows = [(k, old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    for k, o, n in rows:
        print(f"{k}: old={o!r} new={n!r}")
    if rows:
        print(f"diff: {len(rows)} value(s) differ")
        return 1
    print("diff: empty (all effective values equal; kv disk directory is compared on/off by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

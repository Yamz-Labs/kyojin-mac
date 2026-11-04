#!/usr/bin/env python3
"""Kernel launches per decoded token, from the lazy graph of one decode step (mx.export_to_dot), S5 off vs on.
usage: count_launches.py MODEL_DIR [--rows 1,2,3,4] [--json OUT.json]
Every primitive node is one dispatch except the view primitives (Reshape, Transpose, Broadcast, Flatten, Unflatten, Squeeze, ExpandDims, Split, Slice,
AsStrided, StopGradient, Depends), which only change strides / offsets. Contiguous and AsType are counted (they copy when the input is not dense).
Concatenate is counted once (it may issue one copy per input). The tool also lists the copy-like nodes per variant: S5 must not add any."""
import argparse
import collections
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from kyojin_mlx import fused_group as FG
from kyojin_mlx.exl3_linear import load_exl3

VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided", "StopGradient", "Depends"}
COPYLIKE = ("Contiguous", "Copy", "AsType", "Concatenate", "Slice", "SliceUpdate")


def graph_census(out, rows: int = 1) -> dict:
    """Dispatch census of the lazy graph that ends in `out` (an mx.array or a list of them)."""
    path = os.path.join(tempfile.gettempdir(), f"count_launches_{os.getpid()}.dot")
    mx.export_to_dot(path, out)
    txt = open(path).read()
    os.unlink(path)
    labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt)
    c = collections.Counter(labels)
    views = sum(v for k, v in c.items() if k in VIEWS)
    return dict(rows=rows, nodes=len(labels), views=views, dispatches=len(labels) - views, by_type=dict(c.most_common()),
                copylike={k: c[k] for k in COPYLIKE if c.get(k)})


def census(model, rows: int) -> dict:
    cache = make_prompt_cache(model)
    pre = model(mx.array([list(range(1, 33))]), cache=cache)[:, -1]
    mx.eval(pre)
    out = mx.argmax(model(mx.array([list(range(40, 40 + rows))]), cache=cache)[:, -1], axis=-1)
    return graph_census(out, rows)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--rows", default="1,2,3,4"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    model, _ = load_exl3(a.model_dir)
    info = FG.install(model)
    print(f"S5 install: {info['groups']} groups, {info['members']} members, {info['launches_saved']} launches saved (3 per extra member)", flush=True)
    res = {}
    for rows in [int(r) for r in a.rows.split(",")]:
        for variant, en in (("off", False), ("on", True)):
            FG.ENABLED = en
            res[f"{variant}_rows{rows}"] = census(model, rows)
        FG.ENABLED = True
        off, on = res[f"off_rows{rows}"], res[f"on_rows{rows}"]
        print(f"rows {rows}: dispatches/token off {off['dispatches']} on {on['dispatches']} ({on['dispatches'] - off['dispatches']:+d}, "
              f"{100 * (on['dispatches'] - off['dispatches']) / off['dispatches']:+.1f} %); graph nodes {off['nodes']} -> {on['nodes']}; "
              f"CustomKernel {off['by_type'].get('CustomKernel', 0)} -> {on['by_type'].get('CustomKernel', 0)}", flush=True)
        new = {k: on["copylike"].get(k, 0) - off["copylike"].get(k, 0) for k in set(on["copylike"]) | set(off["copylike"])}
        print("   copy-like nodes off", off["copylike"], "on", on["copylike"], "-> new copy nodes:", {k: v for k, v in new.items() if v > 0} or "none", flush=True)
        res[f"new_copy_nodes_rows{rows}"] = {k: v for k, v in new.items() if v > 0}
    r1 = res.get("off_rows1")
    if r1:
        print("decode step (rows 1) by type, off -> on:")
        for k, v in r1["by_type"].items():
            w = res["on_rows1"]["by_type"].get(k, 0)
            if v != w or v >= 12:
                print(f"   {k:28s} {v:5d} -> {w:5d}{'  (view)' if k in VIEWS else ''}")
    if a.json:
        json.dump(dict(install=info, results=res), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where do the dispatches of one DS4.1 decode step go? Per module (block, attention, FFN/MoE, indexer, compressor) of a patched slice pack: the module's inputs
are evaluated first (so they are graph leaves) and the graph of its output is counted with tools/count_launches.graph_census.
usage: s5_modules.py PACK_DIR [--s5 0|1] [--rows 1]"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from count_launches import graph_census
from kyojin_mlx import fused_group as FG
from kyojin_mlx import patch as P

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--s5", type=int, default=1); ap.add_argument("--rows", type=int, default=1); ap.add_argument("--top", type=int, default=14)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
lm = model.language_model
vocab = cfg["text_config"]["vocab_size"]
seq = np.random.default_rng(0).integers(3, vocab, 40).tolist()
cache = lm.make_cache()
mx.eval(lm(mx.array([seq[:5]]), cache=cache).logits)
FG.install(model)
FG.ENABLED = bool(a.s5)
rec = collections.defaultdict(list)


def detach(v):
    if isinstance(v, mx.array):
        mx.eval(v)
    elif isinstance(v, (list, tuple)):
        for i in v:
            detach(i)
    return v


def hook(obj, key):
    cls = obj.__class__

    def __call__(self, *args, **kw):
        for v in args:
            detach(v)
        for v in kw.values():
            detach(v)
        out = cls.__call__(self, *args, **kw)
        outs = [o for o in (out if isinstance(out, (list, tuple)) else [out]) if isinstance(o, mx.array)]
        if outs:
            rec[key].append(graph_census(outs))
            mx.eval(*outs)
        return out

    obj.__class__ = type(cls.__name__, (cls,), {"__call__": __call__})


for i, layer in enumerate(lm.layers):
    hook(layer, f"L{i} block")
    hook(layer.attn, f"L{i} attn")
    hook(layer.ffn, f"L{i} ffn")
    if getattr(layer.attn, "indexer", None) is not None:
        hook(layer.attn.indexer, f"L{i} indexer")
    if getattr(layer.attn, "compressor", None) is not None:
        hook(layer.attn.compressor, f"L{i} compressor")
mx.eval(lm(mx.array([seq[5:5 + a.rows]]), cache=cache).logits) if False else None
rec.clear()
out = lm(mx.array([seq[5:5 + a.rows]]), cache=cache).logits
mx.eval(out)
print(f"S5 {'on' if a.s5 else 'off'}, rows {a.rows}: dispatches per module call (views excluded); top types")
tot = collections.Counter()
for k, v in rec.items():
    c = v[-1]
    tot[k] = c["dispatches"]
    top = ", ".join(f"{t} {n}" for t, n in list(c["by_type"].items()) if t not in ("Broadcast", "Reshape", "ExpandDims", "Squeeze", "Transpose", "Flatten", "Unflatten", "Slice", "Split"))
    print(f"{k:16s} {c['dispatches']:4d}   {top}")

"""Vendor-quantized source checkpoints for tools/conv_mlx.py (FP8, MXFP4, NVFP4; BF16 passes through untouched).

`load_source(src, files)` returns the dict conv_mlx already uses (name -> mx array, lazy), except that every linear
stored in a vendor quantization is a LAZY entry: it is dequantized to float16 only when the converter asks for it
(`W[name]`, or `W.items_prefix(prefix)` for one decoder block), one linear at a time, so the footprint stays at the
block being converted plus one fp32 numpy temporary of the largest linear. The scale / global-scale / activation-scale
companions are dropped from the mapping (conv_mlx copies every unquantized key of a block into the output pack).

The format knowledge is `vendor_input.py`, a byte-identical copy of exllamav3/foreign_quant/vendor_input.py of the
CUDA/ROCm tree (numpy only); this file is the MLX glue.
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from . import vendor_input as vi

_MX_OK = {"F32", "F16", "BF16", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "BOOL", "F64", "F8_E4M3", "F8_E8M0"}


def _load_shard(path, st):
    try:
        return mx.load(path)
    except Exception:                       # a dtype mlx cannot load (F8_E5M2 / F4) somewhere in the shard: tensor by tensor
        out = {}
        for k, (tag, shape) in st.info().items():
            if tag not in _MX_OK:
                continue
            _, _, a = st.raw(k)
            m = mx.array(np.ascontiguousarray(a))
            out[k] = m.view(mx.bfloat16) if tag == "BF16" else m
        return out


class SrcDict(dict):
    def __init__(self, raw, ck, plan):
        super().__init__(raw)
        self._ck, self._plan = ck, plan                       # plan: weight tensor name -> linear prefix
        for p in plan.values():
            for sfx in vi_companions(ck.info(), p):
                self.pop(sfx, None)
        for wn in plan:
            self.setdefault(wn, None)                         # placeholder key (E5M2 weights are not in the mlx dict)

    def __getitem__(self, k):
        p = self._plan.get(k)
        if p is None:
            return super().__getitem__(k)
        w = vi.load_linear(self._ck, p)                       # float32 numpy [out, in]
        return mx.array(w).astype(mx.float16)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def values(self):
        return (self[k] for k in self.keys())

    def items_prefix(self, prefix):
        return ((k, self[k]) for k in self.keys() if k.startswith(prefix))


def vi_companions(info, prefix):
    c = vi.classify_linear(info, prefix)
    names = [c.get("scale"), c.get("scale2")]
    # activation scales ride along with the weight and have no use in a conversion
    names += [prefix + s for s in (".input_scale", ".input_global_scale") if prefix + s in info]
    return [n for n in names if n]


def src_items(W, prefix):
    """(name, array) of the source tensors under `prefix`; lazy entries are only dequantized when they match."""
    if hasattr(W, "items_prefix"):
        return W.items_prefix(prefix)
    return ((k, v) for k, v in W.items() if k.startswith(prefix))


def load_source(src, files, log=print):
    ck = vi.Checkpoint(src)
    info = ck.info()
    rep = vi.detect_checkpoint(info, ck.config)
    if rep["primary"] == "unsupported-integer-quant":
        raise SystemExit(f"source {src}: integer-quantized linears with no known scale layout (GPTQ / AWQ?): {rep['unsupported_examples']}")
    log(f"source format: {rep['primary']} {rep['counts']}" + (f" config {rep['quantization_config']}" if rep["quantization_config"] else ""))
    raw = {}
    for f in files:
        raw.update(_load_shard(os.path.join(src, f), ck.files[f]) if f in ck.files else mx.load(os.path.join(src, f)))
    plan = {}
    for p in vi.linear_prefixes(info):
        c = vi.classify_linear(info, p)
        if c and not c["fmt"].startswith("dense_"):
            if c["fmt"] == "mxfp4_blocks":
                continue                                      # fused gpt-oss experts: not a Qwen3.5 layout, left as stored
            plan[c["weight"]] = p
    if not plan:
        return raw
    return SrcDict(raw, ck, plan)

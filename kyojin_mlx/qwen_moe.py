# SPDX-License-Identifier: Apache-2.0
"""Routed experts of a Qwen3.5-MoE / Qwen3-Next pack on the fused Metal MoE path (moe.MoeGather), for `load_exl3` (mlx-lm model, no mlx-vlm).
The converter writes one trellis per expert (`<layer>.mlp.experts.<e>.<proj>.{trellis,suh,svh,mul1}`); `stack_experts` concatenates them into the
ragged layout MoeGather reads (same arrays as tools/convert_stack.py, built in memory), `patch_sparse_blocks` swaps the block's `switch_mlp` call for
the gather (routing, shared expert and its gate stay the model's own code)."""
from __future__ import annotations

import re

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import moe as M

PROJ = ("gate_proj", "up_proj", "down_proj")
EXP = re.compile(r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1|mcg)$")


def pop_experts(weights: dict) -> dict:
    """Removes the per-expert pack tensors from `weights`; {mlp prefix: {proj: {e: {trellis, suh, svh}}}}."""
    out: dict = {}
    for k in list(weights):
        m = EXP.match(k)
        if m:
            v = weights.pop(k)
            if m[4] in ("trellis", "suh", "svh"):
                out.setdefault(m[1], {}).setdefault(m[3], {}).setdefault(int(m[2]), {})[m[4]] = v
    return out


def stack_experts(per_e: dict, cb: str = "mul1") -> M.Stack:
    """per_e {e: {trellis [kt, nt, 16K] int16, suh, svh}} -> Stack (ragged: K may differ between experts)."""
    n = len(per_e)
    assert sorted(per_e) == list(range(n)), "expert ids are not contiguous"
    tr = [np.array(per_e[e]["trellis"]) for e in range(n)]
    kt, nt = tr[0].shape[:2]
    assert all(t.shape[:2] == (kt, nt) and t.dtype == np.int16 for t in tr)
    off = np.concatenate([[0], np.cumsum([t.size // 2 for t in tr])]).astype(np.uint32)
    assert off[-1] < 2 ** 32
    words = np.concatenate([np.ascontiguousarray(t).reshape(-1).view(np.uint32) for t in tr])
    ks = np.array([t.shape[2] // 16 for t in tr], np.uint8)
    suh = np.stack([np.array(per_e[e]["suh"]) for e in range(n)]).astype(np.float16)
    svh = np.stack([np.array(per_e[e]["svh"]) for e in range(n)]).astype(np.float16)
    return M.Stack(mx.array(words), mx.array(off), mx.array(ks), mx.array(suh), mx.array(svh), int(kt), int(nt), cb)


def _call(self, x):
    from .patch import _moe_rows
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    h = x.shape[-1]
    y = _moe_rows(self.kmoe, x.reshape(-1, h).astype(mx.float16), inds.reshape(-1, k), scores.astype(mx.float32).reshape(-1, k)).reshape(*x.shape[:-1], h)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return (y + shared.astype(mx.float32)).astype(x.dtype)


def patch_sparse_blocks(model, experts: dict, cb: str = "mul1") -> int:
    """`experts` from pop_experts (names as in the sanitized model: `language_model.model.layers.<i>.mlp`). Returns the number of blocks swapped."""
    root = model.language_model.model if hasattr(model.language_model, "model") else model.language_model
    n = 0
    for prefix, projs in experts.items():
        li = int(re.search(r"layers\.(\d+)\.mlp$", prefix)[1])
        blk = root.layers[li].mlp
        assert hasattr(blk, "switch_mlp"), f"layer {li}: no switch_mlp to replace"
        assert set(projs) == set(PROJ), (prefix, sorted(projs))
        st = {p: stack_experts(projs[p], cb) for p in PROJ}
        assert st["gate_proj"].experts == blk.num_experts, (prefix, st["gate_proj"].experts, blk.num_experts)
        blk.kmoe = M.MoeGather(st["gate_proj"], st["up_proj"], st["down_proj"], int(blk.top_k), limit=1e30)
        blk.switch_mlp = nn.Module()                                      # no random fp experts left to materialise
        blk.__class__ = type("KyojinQwenMoe", (type(blk),), {"__call__": _call})
        mx.eval(*[a for s in st.values() for a in (s.words, s.offset, s.k, s.suh, s.svh)])
        n += 1
    return n

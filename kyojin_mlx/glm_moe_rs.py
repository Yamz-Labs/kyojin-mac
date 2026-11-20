# SPDX-License-Identifier: Apache-2.0
"""BP-MOE2: router rowdot + top-K select in ONE launch (flag KYOJIN_GLM_MOE_E_RS, default off, read per call, only inside the glm_moe E branch).

On the synthetic GLM-5.3 MoE chain the router costs two dependent launches (rowdot 9.4 us, select 9.2 us on the critical path, exact stamps): a dispatch
boundary (~7 us) is paid twice for 288 dot products and one simdgroup of top-8. Here the rowdot kernel is the SAME source text (`rowinv._ROWDOT`, one output
per threadgroup, KS = 1) and the select is the SAME source text (`glm_moe._SELECT_BODY`): the threadgroup that finishes last (device atomic counter) runs
the select on the logits the others stored. Same arithmetic => logits, indices and weights are bit-identical to rowdot + select.

Synchronisation: store of the logit -> device fence -> relaxed atomic increment of the counter; the last arriver fences again before it reads the logits.
The counter lives in one persistent buffer, passed as an input and reset to 0 by the last arriver (no init launch): launches that use it must be ordered
(they are: every layer's router input depends on the previous layer's output).

Eligible: KS == 1 (N >= 192 outputs), T <= ROWDOT_MAX rows, E == router outputs, K <= 16.
"""
from __future__ import annotations

import functools
import os
import re

import mlx.core as mx

from . import glm_moe as GM
from . import rowinv as RI


def on() -> bool:
    return os.environ.get("KYOJIN_GLM_MOE_E_RS", "0") not in ("0", "")


_TAIL = r"""
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint is_last;
    if (sg == 0 && lane == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint prev = atomic_fetch_add_explicit(((device atomic_uint*)cnt) + threadgroup_position_in_grid.y, 1u, memory_order_relaxed);
        is_last = (prev == N - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (sg == 0 && lane == 0) atomic_store_explicit(((device atomic_uint*)cnt) + threadgroup_position_in_grid.y, 0u, memory_order_relaxed);
    for (uint rr = 0; rr < MR; ++rr) {
      const uint t = rb + rr;
      if (t < ROWS) {
        device const float* L = ((device const float*)y) + t * uint(NE);
        SELBODY
      }
    }
"""


def _sel_body() -> str:
    """The select source with its template names renamed (E -> NE, K -> TOPK: the rowdot body already owns `K`), the head (t / lane / L) supplied by the tail."""
    b = GM._SELECT_BODY
    b = re.sub(r"\bE\b", "NE", b)
    b = re.sub(r"\bK\b", "TOPK", b)
    return b


@functools.lru_cache(maxsize=None)
def _kern(E: int, K: int, norm: bool, scale: float):
    # the select body is written with uint(E) / uint(K) casts: NE / TOPK are template constants of this kernel
    row = RI._ROWDOT
    st = "y[ulong(rb + r) * N + o] = static_cast<YT>(s);"
    assert st in row
    # logits are stored / loaded as relaxed device atomics (bypass the per-core caches: the last arriver sees what the others stored)
    row = row.replace(st, "atomic_store_explicit(((device atomic_uint*)y) + ulong(rb + r) * N + o, as_type<uint>(float(s)), memory_order_relaxed);")
    body = _sel_body()
    assert "L[i]" in body
    body = body.replace("L[i]", "as_type<float>(atomic_load_explicit(((const device atomic_uint*)L) + i, memory_order_relaxed))")
    src = row + _TAIL.replace("SELBODY", body.replace("SCALE", repr(float(scale)) + "f"))
    return mx.fast.metal_kernel(name=f"kyojin_glm_rowsel_{E}_{K}_{GM._scale_tag(scale)}", input_names=["x", "w", "params", "bias", "cnt"], output_names=["y", "inds", "wts"],
                                header=RI._LD8, source=src)


_CNT: dict = {}


def eligible(T: int, E: int, K: int, K_in: int) -> bool:
    return 1 <= T <= RI.ROWDOT_MAX and RI.pick_ks(K_in, E) == 1 and K <= 16 and E <= 1024 and K_in % 8 == 0


def rowsel(x2, w, bias, T, E, K, norm, scale):
    """x2 [T, H], router weight w [E, H] (fp16 / bf16 / fp32), bias [E] f32 -> (logits f32 [T, E], inds uint32 [T * K], weights f32 [T * K]); bit-identical to rowdot + glm_moe.select."""
    B, Kin = x2.shape
    mr = min(B, RI.ROWDOT_MAX)
    ng = (B + mr - 1) // mr
    cnt = _CNT.get(ng)
    if cnt is None:
        cnt = _CNT[ng] = mx.zeros((4096,), dtype=mx.uint32)      # large enough that mlx binds it as a device buffer (small inputs go to the constant space)
        mx.eval(cnt)
    p = RI._params.get((Kin, E, B))
    if p is None:
        p = RI._params[(Kin, E, B)] = mx.array([Kin, E, B], dtype=mx.uint32)
    y, inds, wts = _kern(E, K, bool(norm), float(scale))(
        inputs=[x2, w, p, bias, cnt],
        template=[("MR", mr), ("KS", 1), ("YT", mx.float32), ("V8", int(Kin % 8 == 0)), ("NE", E), ("TOPK", K), ("EL", -(-E // 32)), ("NORM", bool(norm) and K > 1)],
        grid=(E * 32, ng, 1), threadgroup=(32, 1, 1), output_shapes=[(B, E), (T * K,), (T * K,)], output_dtypes=[mx.float32, mx.uint32, mx.float32])
    return y, inds, wts

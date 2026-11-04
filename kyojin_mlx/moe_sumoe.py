# SPDX-License-Identifier: Apache-2.0
# Portions adapted from beamivalice/sushi (MIT / Apache-2.0), src/exl3/rate_groups.zig@0f4f00b8a166fe8dc67bca05b34c732df7c7d176
# (moeDecodeClampedSharedFused: the shared expert rides the routed launches; see THIRD_PARTY_NOTICES.md).
"""sumoe: the routed MoE decode AND the shared expert in five launches per layer (stock: five routed + the shared expert's own chain).

The shared expert becomes one more pair per row, expert ESH = the last expert of the three stacks (`with_shared` appends it). The rot launch
writes the extended routing, the gather / union GEMV launches read it unchanged, the mid launch skips the fp8 fake quant for the shared slot and
the final launch rounds the shared slot to bf16 before the fp32 sum (the stock chain's output rounding), then adds it last, as the model does
(`routed.sum + shared`). Output: routed + shared in fp32 [rows, hidden].

Not bit-identical to the stock shared chain (that one rounds the gate / up / swiglu in bf16, this one keeps fp32 until the bf16 output); routed
parts are bit-identical to the plain gather path. Behind KYOJIN_PORT_SUMOE=1 (default 0)."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import metal_src as S
from . import moe as M
from . import fired as FD

SUMOE = os.environ.get("KYOJIN_PORT_SUMOE", "0") == "1"


def _append(st: M.Stack, trellis_u32: mx.array, k: int, suh: mx.array, svh: mx.array) -> M.Stack:
    """Stack `st` plus one more expert (words of a [kt, nt, 16k] int16 trellis viewed as u32, bitrate k, signs) as its last expert."""
    off = np.array(st.offset).astype(np.uint64)
    n = int(trellis_u32.size)
    return M.Stack(mx.concatenate([st.words, trellis_u32.reshape(-1)]),
                   mx.array(np.concatenate([off, [off[-1] + n]]).astype("<u4")),
                   mx.concatenate([st.k, mx.array([k], dtype=st.k.dtype)]),
                   mx.concatenate([st.suh, suh.reshape(1, -1).astype(st.suh.dtype)]),
                   mx.concatenate([st.svh, svh.reshape(1, -1).astype(st.svh.dtype)]), st.kt, st.nt, st.cb)


def with_shared(g: M.Stack, u: M.Stack, d: M.Stack, shared) -> tuple[M.Stack, M.Stack, M.Stack]:
    """Stacks with the shared expert appended as expert E. `shared` = three (trellis int16 [kt, nt, 16k], suh, svh) for gate, up, down (the
    Exl3Linear arrays of the shared expert). Same intermediate width and codebook as the routed experts. Load-time copy: free the dense
    shared linears afterwards."""
    out = []
    for st, (tr, suh, svh) in zip((g, u, d), shared):
        kt, nt, w = tr.shape
        assert (kt, nt) == (st.kt, st.nt), "the shared expert must have the routed experts' shapes"
        out.append(_append(st, tr.view(mx.uint32), w // 16, suh, svh))
    return tuple(out)


class MoeSharedFused(M.MoeGather):
    """MoeGather over stacks that end with the shared expert (`with_shared`). `topk` is the routed top-k; the layer runs topk + 1 pairs per row.
    __call__(x, ids [rows, topk], wts [rows, topk], fq8) -> routed + shared, fp32 [rows, hidden]. Decode rows only (rows * (topk + 1) <= MAX_PAIRS)."""

    def __init__(self, gate, up, down, topk, shared_fq=True, **kw):
        kw.setdefault("union", True)
        kw["grouped"] = False
        kw["hotk"] = 0
        super().__init__(gate, up, down, topk + 1, **kw)
        self.tk = topk                                                      # routed top-k (self.topk = pairs per row)
        self.ktop = topk                                                    # the k-split of the routed-only call (pin_rows * routed top-k pairs), so a row's partial sums equal the plain gather's
        self.esh = gate.experts - 1
        self.shared_fq = shared_fq                                          # False: the shared slot skips the fp8 fake quant (stock shared chain without it)
        self._is_fused_inner = True

    def _extra(self, rows: int) -> dict:
        P = self._params(rows)
        if "rot_sh" not in P:
            u32 = lambda *v: mx.array(list(v), dtype=mx.uint32)
            P["rot_sh"] = u32(self.g.kin, self.topk, self.tk, self.esh)
            P["mid_sh"] = u32(P["mid"][0].item(), self.g.nout, 0xFFFFFFFF if self.shared_fq else self.esh)
            mx.eval(P["rot_sh"], P["mid_sh"])
        return P

    def rot_sh(self, x: mx.array, ids: mx.array, wts: mx.array, fq8: bool):
        rows, kd = x.shape
        P = self._extra(rows)
        ih = x.dtype == mx.float16
        kern = self._kern("rotsh", (ih, fq8), f"kyojin_rotsh_{'h' if ih else 'f'}{'_fq' if fq8 else ''}", ["x", "ids", "wts", "suh0", "suh1", "params"],
                          ["xh", "ids2", "wts2"], lambda: S.rot_gather_sh_source(2, ih, fq8))
        gt = P.get("hg_rot")
        if gt is None:
            gt = P["hg_rot"] = S.had_grid(kd // 128, 2 * P["npairs"])
        grid, tg = gt
        return kern(inputs=[x, ids, wts, self.g.suh, self.u.suh, P["rot_sh"]], grid=grid, threadgroup=tg,
                    output_shapes=[(2 * P["npairs"], kd), (P["npairs"],), (P["npairs"],)], output_dtypes=[mx.float32, mx.int32, mx.float32])

    def mid_sh(self, part: mx.array, ids2: mx.array, wts2: mx.array, rows: int) -> mx.array:
        P = self._extra(rows)
        kern = self._kern("midsh", 0, f"kyojin_midsh_{int(self.wpre)}{int(self.fq)}", ["part", "ids", "wts", "svh0", "svh1", "suhd", "params"], ["xh2"],
                          lambda: S.mid_sh_source(self.limit, self.wpre, self.fq, self.alpha, self.beta))
        gt = P.get("hg_mid")
        if gt is None:
            gt = P["hg_mid"] = S.had_grid(self.g.nout // 128, P["npairs"])
        grid, tg = gt
        return kern(inputs=[part, ids2, wts2, self.g.svh, self.u.svh, self.d.suh, P["mid_sh"]], grid=grid, threadgroup=tg,
                    output_shapes=[(P["npairs"], self.g.nout)], output_dtypes=[mx.float32])[0]

    def final_sh(self, part: mx.array, ids2: mx.array, wts2: mx.array, rows: int) -> mx.array:
        if M.FINPAR:
            y = self.final_par(part, ids2, wts2, rows, bf16_last=True)
            if y is not None:
                return y
        P = self._params(rows)
        kern = self._kern("finsh", 0, f"kyojin_finsh_{int(not self.wpre)}", ["part", "ids", "wts", "svh", "params"], ["y"],
                          lambda: S.final_gather_sh_source(not self.wpre))
        gt = P.get("hg_fin")
        if gt is None:
            gt = P["hg_fin"] = S.had_grid(self.d.blocks, rows)
        grid, tg = gt
        return kern(inputs=[part, ids2, wts2, self.d.svh, P["fin"]], grid=grid, threadgroup=tg,
                    output_shapes=[(rows, self.d.nout)], output_dtypes=[mx.float32])[0]

    def __call__(self, x: mx.array, ids: mx.array, wts: mx.array, fq8: bool = False) -> mx.array:
        rows = x.shape[0]
        assert rows * self.topk <= M.MAX_PAIRS and ids.shape == (rows, self.tk) and wts.shape == (rows, self.tk)
        ids, wts = ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32)
        FD.hit("port_sumoe")                                          # per call of the fused routed + shared launches
        xh, ids2, wts2 = self.rot_sh(x, ids, wts, fq8)
        part = self.gemv_gu(xh, ids2, rows)
        xh2 = self.mid_sh(part, ids2, wts2, rows)
        part2 = self.gemv_dn(xh2, ids2, rows)
        return self.final_sh(part2, ids2, wts2, rows)


def _lin_parts(lin):
    """(u32 words, K, suh, svh) of an Exl3Linear without bias, or None."""
    g = getattr(lin, "_g", None)
    if g is None or getattr(lin, "_bias", None) is not None:
        return None
    b = g[1]
    return b.words, b.K, b.suh, b.svh, lin._kt, lin._nt, lin.cb


def wire_shared(g: M.Stack, u: M.Stack, d: M.Stack, shared_mod, limit: float, alpha: float, beta: float):
    """Stacks with the model's shared expert appended as the last expert, or None when it cannot ride the routed launches: `shared_mod` needs
    gate_proj / up_proj / down_proj Exl3Linear (no bias, no fused gate_up), the routed stacks' shapes and codebook, the routed swiglu (limit,
    alpha 1, beta 0)."""
    if alpha != 1.0 or beta != 0.0:
        return None
    sl = getattr(shared_mod, "swiglu_limit", None)
    if sl is not None and float(sl) != float(limit):
        return None
    parts = [_lin_parts(getattr(shared_mod, n, None)) for n in ("gate_proj", "up_proj", "down_proj")]
    if any(p is None for p in parts):
        return None
    for st, p in zip((g, u, d), parts):
        if (p[4], p[5], p[6]) != (st.kt, st.nt, st.cb):
            return None
    return tuple(_append(st, p[0], p[1], p[2], p[3]) for st, p in zip((g, u, d), parts))


class SharedStub(nn.Module):
    """Replaces the parent's `shared_experts`: the rows the fused launches serve get nothing (the routed call adds the shared expert itself),
    other rows (prompts, chunks) run the dense shared expert as before. `takes(rows)` is set by the owner."""

    def __init__(self, dense, none):
        super().__init__()
        self.dense = dense
        self._none = none
        self.takes = lambda rows: False

    def __call__(self, x):
        if self.takes(x.size // x.shape[-1]):
            return None if self._none else mx.zeros((), dtype=mx.float32)
        return self.dense(x)

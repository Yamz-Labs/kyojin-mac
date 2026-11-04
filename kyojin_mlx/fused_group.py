# SPDX-License-Identifier: Apache-2.0
"""S5 dispatch cut: linears that read the SAME input (MLP gate/up, GDN in_proj_qkv/in_proj_z, attention q/k/v) run as ONE group of three kernel
launches instead of three launches per member (rot + GEMV + finalize):

  rot       xh[m] = H128(x * suh_m) / sqrt(128)         grid z = member, suh stacked [M, in], xh [M * rows, in]
  gemv      the per-member GEMV with member m's own NT / tgk, one launch: the flat grid x enumerates (member, k-split, 128-column block);
            every member keeps ITS OWN k-split count (its own tgk), so the summation order of each output element is exactly the one of the
            separate launch. The member trellis buffers are separate kernel inputs (no concatenated weight copy); partials [max tgk, rows, N_total]
  finalize  y_m = H128(sum of partials) / sqrt(128) * svh_m, one launch, M separate contiguous outputs (no slice / copy node after it)

The group kernels are the single-linear kernels with asserted text substitutions (like grouped.py), so every output is BIT-IDENTICAL to the separate
Exl3Linear path, for equal-size members (gate/up, qkv/z) and for ragged ones (q/k/v).

Wiring (no change to Exl3Linear / the model code): `install(model)` traces one 1-row forward, finds the Exl3Linear calls that receive the same input array
object, and replaces those modules with `FusedMember`s. The first member called computes the whole group and caches (input, outputs); the other members
return their cached output when called with the same input object. More rows than the GEMV handles (prefill) or `ENABLED = False` (env KYOJIN_S5=0, or
set the module attribute at run time to A/B in one process) go to the original Exl3Linear. Launches per decoded token on Qwen3.5-2B: see tools/count_launches.py."""
from __future__ import annotations

from . import archs as _A
import functools
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import cbill as CB
from . import fired as FD
from . import pf_ring as _RING
from . import exl3_linear as EL
from . import fqfuse as FQF
from . import kernels as KN
from . import metal_src as S

ENABLED = os.environ.get("KYOJIN_S5", "1") != "0"
MAX_MEMBERS = 4


def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"template changed, {old!r} found {text.count(old)} times"
    return text.replace(old, new)


@functools.lru_cache(maxsize=None)
def rot_multi_source(in_half: bool, fq8: bool = False) -> str:
    """rot_source with grid z = member: suh [M, in] stacked, output xh [M, rows, in]. params u32 [in, rows]. fq8: the fp8 round trip of x folded in."""
    s = S.rot_source(in_half, False, fq8)
    s = _sub(s, "const uint KD = params[0];", "const uint KD = params[0], RR = params[1];\n    const uint zm = threadgroup_position_in_grid.z;")
    s = _sub(s, "(suh + blk * 128 + lane * 4)", "(suh + ulong(zm) * KD + blk * 128 + lane * 4)")
    s = _sub(s, "*((device float4*)(xh + o))", "*((device float4*)(xh + ulong(zm) * RR * KD + o))")
    return s


@functools.lru_cache(maxsize=None)
def gemv_multi_source(K: int, rows: int, cb: str, M: int, prefetch: bool = False) -> str:
    """gemv_source over M members in one launch. params u32 [KT, N_total, M, (NT_j, TGK_j, count_j, col0_j) * M], count_j = (NT_j / 8) * TGK_j
    threadgroups; grid x = sum of counts. Output part [max tgk, rows, N_total]: member j writes columns col0_j .. col0_j + 16 NT_j for k-splits < TGK_j."""
    s = S.gemv_source(K, rows, prefetch, cb)

    def pick(j):
        return f"trellis{j}" if j == M - 1 else f"(mm == {j} ? trellis{j} : {pick(j + 1)})"

    decode = ("const uint KT = params[0], NTOT = params[1], M = params[2];\n"
              "    const uint f = threadgroup_position_in_grid.x;\n"
              "    uint mm = 0, base = 0;\n"
              "    while (mm + 1 < M && f >= base + params[3 + 4 * mm + 2]) { base += params[3 + 4 * mm + 2]; ++mm; }\n"
              "    const uint NT = params[3 + 4 * mm], TGK = params[3 + 4 * mm + 1], COFF = params[3 + 4 * mm + 3];\n"
              "    const uint loc = f - base, nblk = NT >> 3;\n"
              "    const uint blk = loc % nblk, kblock = loc / nblk;")
    s = _sub(s, "const uint blk = threadgroup_position_in_grid.x;        // 128-column output block", decode)
    s = _sub(s, "const uint kblock = threadgroup_position_in_grid.y;", "")
    s = _sub(s, "const uint KT = params[0], NT = params[1], TGK = params[2];", "")
    s = _sub(s, "const uint KD = KT * 16, N = NT * 16;", "const uint KD = KT * 16;")
    s = _sub(s, "const device uint* tp = trellis + (ulong(ks0) * NT + nt) * TW;",
             f"const device uint* tp = {pick(0)} + (ulong(ks0) * NT + nt) * TW;")
    s = _sub(s, "(const device float4*)(xh + ks0 * 16);", "(const device float4*)(xh + ulong(mm) * MR * KD + ks0 * 16);")
    s = _sub(s, "part[(ulong(kblock) * MR + r) * N + blk * 128 + c] = v;", "part[(ulong(kblock) * MR + r) * NTOT + COFF + blk * 128 + c] = v;")
    return s


@functools.lru_cache(maxsize=None)
def fused_multi_source(K: int, rows: int, cb: str, M: int, in_half: bool, fq8: bool, xsr: int, reads: str = S.TAIL_ATOM) -> str:
    """The M-member group as ONE launch (rot + GEMV + finalize, metal_src.fuse_gemv over gemv_multi_source). Inputs x, suh [M, in], svh [N_total], trellis0.., params
    (the gemv params + the counter base at [3 + 4 M]), ctr; outputs part, y0.. . reads: how the finalize reads the partials (bit-identical, metal_src.tail_reads)."""
    ym = lambda j: f"y{j}" if j == M - 1 else f"(mm == {j} ? y{j} : {ym(j + 1)})"
    return S.fuse_gemv(gemv_multi_source(K, rows, cb, M, True), rows, xsr, in_half, fq8, reads=reads,
                       xo="ulong(rr) * KD + bb * 128 + lane * 4", suh="suh + ulong(mm) * KD + bb * 128 + lane * 4", ctri="params[3 + 4 * M] + (COFF >> 7) + blk",
                       pb="ulong(warp) * NTOT + COFF + blk * 128 + lane * 4", ks="ulong(MR) * NTOT", svh="svh + COFF + blk * 128 + lane * 4",
                       yout="ym + ulong(warp) * (NT * 16) + blk * 128 + lane * 4", tail_pre=f"        device OT* ym = {ym(0)};",
                       xp_old="    const device float4* xp = (const device float4*)(xh + ulong(mm) * MR * KD + ks0 * 16);\n")


@functools.lru_cache(maxsize=None)
def fin_multi_source(M: int) -> str:
    """final_source over the concatenated columns of M members, M separate outputs y0.. . params u32 [M, N_total, rows, (col0_j, N_j, TGK_j) * M]."""
    s = S.final_source(False)

    def pick(j):
        return f"y{j}" if j == M - 1 else f"(mm == {j} ? y{j} : {pick(j + 1)})"

    s = _sub(s, "const uint TGK = params[0], N = params[1], MRT = params[2];", "const uint M = params[0], NTOT = params[1], MRT = params[2];")
    s = _sub(s, "if (blk * 128 >= N) return;",
             "if (blk * 128 >= NTOT) return;\n"
             "    uint mm = 0;\n"
             "    while (mm + 1 < M && blk * 128 >= params[3 + 3 * (mm + 1)]) ++mm;\n"
             "    const uint COFF = params[3 + 3 * mm], N = params[3 + 3 * mm + 1], TGK = params[3 + 3 * mm + 2];\n"
             f"    device OT* ym = {pick(0)};")
    s = _sub(s, "(part + (ulong(kb) * MRT + r) * N + blk * 128 + lane * 4)", "(part + (ulong(kb) * MRT + r) * NTOT + blk * 128 + lane * 4)")
    s = _sub(s, "(y + ulong(r) * N + blk * 128 + lane * 4)", "(ym + ulong(r) * N + (blk * 128 - COFF) + lane * 4)")
    return s


def _u32(*v) -> mx.array:
    return mx.array(list(v), dtype=mx.uint32)


class Exl3Group:
    """M same-input Exl3Linears (equal K, codebook, in_features, out dtype, no bias) evaluated with 3 launches."""

    def __init__(self, members: list):
        assert 2 <= len(members) <= MAX_MEMBERS, len(members)
        m0 = members[0]
        for m in members:
            assert (m.K, m.cb, m.in_features, m.out_dtype) == (m0.K, m0.cb, m0.in_features, m0.out_dtype) and m._bias is None, "not groupable"
        self.members, self.M = members, len(members)
        self.K, self.cb, self.kin, self.out_dtype = m0.K, m0.cb, m0.in_features, m0.out_dtype
        self.dcb = S.dense_cb(self.cb)                                   # KYOJIN_DENSE_DEC, refreshed per call
        gs = [m._g[1] for m in members]
        self.tgk = [g.tgk for g in gs]
        self.blocks = [g.blocks for g in gs]
        self.nouts = [m.out_features for m in members]
        self.ntot = sum(self.nouts)
        self.col0 = [sum(self.nouts[:j]) for j in range(self.M)]
        self.bviews = [g.bview for g in gs]
        self.words = [g.words for g in gs]                                          # the members' own trellis buffers: no weight copy
        self.suh = mx.concatenate([g.suh.astype(mx.float16) for g in gs])          # [M * in]
        self.svh = mx.concatenate([g.svh.astype(mx.float16) for g in gs])          # [N_total]
        kt, nt_all = self.kin // 16, [n // 16 for n in self.nouts]
        gp = [kt, self.ntot, self.M]
        for j in range(self.M):
            gp += [nt_all[j], self.tgk[j], self.blocks[j] * self.tgk[j], self.col0[j]]
        self.p_gemv = _u32(*gp)
        self.count = sum(b * t for b, t in zip(self.blocks, self.tgk))
        self.max_tgk = max(self.tgk)
        self.fusable = all(t & (t - 1) == 0 for t in self.tgk)
        self.xsr = max(S.fused_dims(kt, t) for t in self.tgk)
        self.nsets, self._fnext = KN.Mul1Gemv.FUSED_SETS, 0
        self.ctr = mx.array(np.zeros(self.nsets * (self.ntot // 128), dtype=np.uint32))
        self.p_fused = KN.param_sets(gp, self.ntot // 128, self.nsets)
        self._rows: dict = {}
        mx.eval(self.suh, self.svh, self.p_gemv, *self.words, self.ctr, *self.p_fused)
        self.cx, self.cy = None, None                                                  # last input object and its outputs

    def _params(self, rows: int):
        if rows not in self._rows:
            fp = [self.M, self.ntot, rows]
            for j in range(self.M):
                fp += [self.col0[j], self.nouts[j], self.tgk[j]]
            self._rows[rows] = (_u32(self.kin, rows), _u32(*fp))
            mx.eval(*self._rows[rows])
        return self._rows[rows]

    def rot(self, x2: mx.array, fq8: bool = False) -> mx.array:
        rows, M = x2.shape[0], self.M
        p_rot = self._params(rows)[0]
        ih = x2.dtype == mx.float16
        kr = KN._kern(("s5rot", ih, fq8), name=f"kyojin_s5_rot_{'h' if ih else 'f'}{'_fq' if fq8 else ''}", input_names=["x", "suh", "params"], output_names=["xh"],
                      source=rot_multi_source(ih, fq8))
        grid, tg = S.had_grid(self.kin // 128, rows)
        return kr(inputs=[x2, self.suh, p_rot], grid=(grid[0], grid[1], M), threadgroup=tg, output_shapes=[(M * rows, self.kin)],
                  output_dtypes=[mx.float32])[0]

    def gemv(self, xh: mx.array, rows: int) -> mx.array:
        M = self.M
        dcb = self.dcb = S.dense_cb(self.cb)
        kg = KN._kern(("s5gemv", self.K, rows, dcb, M), name=f"kyojin_s5_{dcb}_gemv_k{self.K}_m{rows}_g{M}",
                      input_names=["xh"] + [f"trellis{j}" for j in range(M)] + ["params"], output_names=["part"],
                      header=S.dense_header(dcb),
                      source=gemv_multi_source(self.K, rows, dcb, M))
        return kg(inputs=[xh, *[CB.bind(w, v) for w, v in zip(self.words, self.bviews)], self.p_gemv], template=[("BITS", self.K), ("MR", rows)], grid=(self.count * 256, 1, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(self.max_tgk, rows, self.ntot)], output_dtypes=[mx.float32])[0]

    def fin(self, part: mx.array, rows: int) -> list:
        M = self.M
        p_fin = self._params(rows)[1]
        kf = KN._kern(("s5fin", M), name=f"kyojin_s5_final_g{M}", input_names=["part", "svh", "params"], output_names=[f"y{j}" for j in range(M)],
                      source=fin_multi_source(M))
        grid, tg = S.had_grid(self.ntot // 128, rows)
        return kf(inputs=[part, self.svh, p_fin], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
                  output_shapes=[(rows, n) for n in self.nouts], output_dtypes=[self.out_dtype] * M)

    def can_fuse(self, rows: int) -> bool:
        return self.fusable and rows <= KN.fuse_rows() and max(rows * self.xsr, 512 * rows) <= KN.Mul1Gemv.FUSED_SMEM_FLOATS

    def fused(self, x2: mx.array, fq8: bool = False) -> list:
        """The whole group in one launch: bit-identical to `run`."""
        rows, M = x2.shape[0], self.M
        if rows > KN.MAX_ROWS:
            FD.hit("fuse_rows_group")
        ih = x2.dtype == mx.float16
        reads = KN.TAIL_READS
        dcb = self.dcb = S.dense_cb(self.cb)                      # KYOJIN_DENSE_DEC, read per call
        kg = KN._kern(("s5fused", self.K, rows, dcb, M, ih, fq8, self.xsr, reads), name=f"kyojin_s5_{dcb}_fused_k{self.K}_m{rows}_g{M}_{'h' if ih else 'f'}{'_fq' if fq8 else ''}_x{self.xsr}{KN._readstag(reads)}",
                      input_names=["x", "suh", "svh"] + [f"trellis{j}" for j in range(M)] + ["params", "ctr"], output_names=["part"] + [f"y{j}" for j in range(M)],
                      header=S.dense_header(dcb),
                      source=fused_multi_source(self.K, rows, dcb, M, ih, fq8, self.xsr, reads))
        params = self.p_fused[self._fnext]
        self._fnext = (self._fnext + 1) % self.nsets
        return kg(inputs=[x2, self.suh, self.svh, *[CB.bind(w, v) for w, v in zip(self.words, self.bviews)], params, self.ctr], template=[("BITS", self.K), ("MR", rows), ("OT", self.out_dtype)],
                  grid=(self.count * 256, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(self.max_tgk, rows, self.ntot)] + [(rows, n) for n in self.nouts],
                  output_dtypes=[mx.float32] + [self.out_dtype] * M)[1:]

    def run(self, x2: mx.array, fq8: bool = False) -> list:
        """x2 [rows, in] fp16 / fp32, rows <= KN.MAX_ROWS -> list of M outputs [rows, N_j] (out dtype). fq8: x2 is the INPUT of an fp8 fake quant, applied inside the rot."""
        rows = x2.shape[0]
        if EL.LINEAR_FUSED and EL.LF_MASK & 2 and self.can_fuse(rows):
            return self.fused(x2, fq8)
        return self.fin(self.gemv(self.rot(x2, fq8), rows), rows)


class FusedMember(nn.Module):
    """Stand-in for one Exl3Linear of a group. Same call signature and output as the original."""

    def __init__(self, group: Exl3Group, idx: int, orig):
        super().__init__()
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "in_features", orig.in_features)
        object.__setattr__(self, "out_features", orig.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        g = self._group
        if g.cx is x and ENABLED:
            return g.cy[self._idx].reshape(*x.shape[:-1], self.out_features)
        shape = x.shape
        rows = 1
        for d in shape[:-1]:
            rows *= d
        if not ENABLED or ((rows > EL.GEMV_MAX_ROWS or rows > KN.MAX_ROWS) and not (KN.MAX_ROWS < rows <= KN.fuse_rows() and EL.LINEAR_FUSED and EL.LF_MASK & 2 and g.can_fuse(rows))):
            return self._orig(x)           # KYOJIN_FUSE_ROWS: 5 / 6 rows stay in the group only through its one-launch fused kernel
        src = FQF.source(x)                                                             # x = fp8 fake quant of src: fold it into the rot
        x2 = (x if src is None else src).reshape(rows, shape[-1])
        if x2.dtype not in (mx.float16, mx.float32):
            x2 = x2.astype(mx.float16)
        g.cy, g.cx = g.run(x2, src is not None), x
        return g.cy[self._idx].reshape(*shape[:-1], self.out_features)


def discover(model, forward) -> list:
    """Run `forward()` (one decode step, graph build only: nothing is evaluated) and return the groups [(name, Exl3Linear), ...] of linears that were
    called with the same input array object. Members are distinct modules of equal (K, cb, in_features, out dtype), no bias, at most MAX_MEMBERS."""
    names = {id(m): n for n, m in model.named_modules() if isinstance(m, EL.Exl3Linear)}
    calls = []
    orig = EL.Exl3Linear.__call__

    def rec(self, x):
        calls.append((names[id(self)], self, x))                                         # keeps x alive: ids stay unique
        return orig(self, x)

    EL.Exl3Linear.__call__ = rec
    try:
        forward()
    finally:
        EL.Exl3Linear.__call__ = orig
    by_x: dict = {}
    for name, mod, x in calls:
        by_x.setdefault(id(x), []).append((name, mod))
    out = []
    for lst in by_x.values():
        seen, uniq = set(), []
        for name, mod in lst:
            if id(mod) not in seen and mod._bias is None:
                seen.add(id(mod)); uniq.append((name, mod))
        buckets: dict = {}
        for name, mod in uniq:
            buckets.setdefault((mod.K, mod.cb, mod.in_features, mod.out_dtype), []).append((name, mod))
        out += [b[:MAX_MEMBERS] for b in buckets.values() if len(b) >= 2]
    # the same group is seen once per traced call (prefill and decode step): keep each module in ONE group, larger groups first
    final, taken = [], set()
    for g in sorted(out, key=len, reverse=True):
        ids = {id(m) for _, m in g}
        if ids & taken:
            continue
        taken |= ids
        final.append(g)
    return final


def _set(model, path: str, new) -> None:
    parent = model
    parts = path.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if isinstance(parent, list) else getattr(parent, p)
    last = parts[-1]
    if isinstance(parent, list):
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _memo(f, depth: int = 4, note=None):
    """Pure array function f(x) -> y memoised on the input OBJECT (the last `depth` inputs are kept alive so ids stay unique). Returns the very same
    output array for the same input array, so callers that apply f to one x twice (DS4.1 attention: wq_a(fq(x)) then wkv(fq(x))) share one node
    and the group detection sees one input. Bypassed when S5 is disabled (the baseline keeps both calls)."""
    ring: list = []

    def g(x, *args, **kw):
        if not ENABLED or args or kw:
            return f(x, *args, **kw)
        for xi, yi in ring:
            if xi is x:
                return yi
        y = f(x)
        ring.append((x, y))
        del ring[:-depth]
        if note is not None:
            note(x, y)
        return y

    g._s5_memo = True
    g.__wrapped__ = f
    return g


def memoize_fake_quant() -> bool:
    """DS4.1 (mlx-vlm deepseek_v41.language): fake_quant_fp8_ue8m0(x) is evaluated once per input array instead of once per consuming linear."""
    try:
        import importlib
        m = _A.lang()
    except ImportError:
        return False
    f = getattr(m, "fake_quant_fp8_ue8m0", None)
    if f is None:
        return False
    if not getattr(f, "_s5_memo", False):
        m.fake_quant_fp8_ue8m0 = _memo(f, note=FQF.note)
    return True


_WINDOW_PART_SHA = "181647ecfd8c236489aec35eaeb9e11c8da45c472e15d2c62124269ef4f9eabb"        # mlx-vlm 0.7.10 DeepseekV41Attention._window_part
_win_memo: dict = {}


def share_window_math() -> bool:
    """DS4.1 attention builds the ring-buffer indices and the sliding-window mask (6 arange / remainder / compare nodes ... about 10 dispatches) in EVERY
    layer from Python ints that are the same for all layers of a token (start_pos, seqlen, window). They are built once per token and shared by the
    layers (same arrays, same values). Only applied when `_window_part` is the audited mlx-vlm 0.7.10 source; otherwise the model is left as is."""
    try:
        import hashlib
        import inspect
        import importlib
        L = _A.lang()
        cls = _A.cls("attn")
    except ImportError:
        return False
    orig = cls._window_part
    if getattr(orig, "_s5_window", False):
        return True
    if hashlib.sha256(inspect.getsource(orig).encode()).hexdigest() != _WINDOW_PART_SHA:
        return False

    def idx(start_pos, seqlen, win):
        key = (start_pos, seqlen, win)
        e = _win_memo.get(key)                                  # several positions per layer in a chained verify: keep the last few
        if e is None:
            base = max(0, start_pos - win + 1)
            end_pos = start_pos + seqlen
            retained = min(seqlen, win)
            hist = mx.arange(base, start_pos) % win if start_pos else None
            ring = mx.arange(end_pos - retained, end_pos) % win
            positions = mx.arange(base, end_pos)
            queries = mx.arange(start_pos, end_pos)
            valid = (positions[None, :] <= queries[:, None]) & (positions[None, :] > queries[:, None] - win)
            if len(_win_memo) >= 16:
                _win_memo.pop(next(iter(_win_memo)))
            e = _win_memo[key] = (hist, ring, valid[None, None])
        return e

    def _window_part(self, x, start_pos, cache):
        if not ENABLED:
            return orig(self, x, start_pos, cache)
        batch, seqlen = x.shape[0], x.shape[1]
        win = self.window_size
        fq = L.fake_quant_fp8_ue8m0
        kv = self.kv_norm(self.wkv(fq(x)))
        kv = self.rope(kv[:, None], start_pos).reshape(batch, seqlen, self.head_dim)
        kv = fq(kv)
        buffer = cache.window[self.layer_idx]
        if buffer is None:
            buffer = mx.zeros((batch, win, self.head_dim), dtype=kv.dtype)
        elif buffer.shape[0] < batch:
            buffer = mx.pad(buffer, [(0, batch - buffer.shape[0]), (0, 0), (0, 0)])
        hist, ring, valid = idx(start_pos, seqlen, win)
        retained = min(seqlen, win)
        if _RING.ENABLED:                                       # pf_win_slice: slices instead of a Gather and a Scatter, same rows
            base = max(0, start_pos - win + 1)
            part = mx.concatenate([_RING.read(buffer, batch, base, start_pos, win), kv], axis=1) if start_pos else kv
            buffer = _RING.write(buffer, batch, kv[:, -retained:], start_pos + seqlen, win)
        else:
            if start_pos:
                part = mx.concatenate([mx.take(buffer[:batch], hist, axis=1), kv], axis=1)
            else:
                part = kv
            buffer[:batch, ring] = kv[:, -retained:]
        cache.window[self.layer_idx] = buffer
        return part, valid

    _window_part._s5_window = True
    cls._window_part = _window_part
    return True


def default_forward(model):
    """One graph-building forward that exercises every path: mlx-lm models: a 1-token step; mlx-vlm DS4.1 / GLM text models (compressor and indexer layers
    only run once the cache holds a few tokens): a 9-token prefill, then a 1-token step on the same (lazy) cache."""
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "make_cache") and type(lm).__module__.startswith("mlx_vlm"):
        ids = list(range(1, 10))

        def fwd():
            cache = lm.make_cache()
            lm(mx.array([ids]), cache=cache)
            lm(mx.array([ids[:1]]), cache=cache)
        return fwd
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    return lambda: model(mx.array([[0]]), cache=cache)


def install(model, forward=None) -> dict:
    """Group the same-input Exl3Linears of `model` (see module doc). Returns {'groups': n, 'members': n, 'launches_saved': n, 'fake_quant_memo': bool, 'window_math_shared': bool, 'names': [[...]]}.
    On DS4.1 it also memoises fake_quant_fp8_ue8m0 per input array so that wq_a and wkv see one input (see _memo)."""
    memo = memoize_fake_quant()
    win = share_window_math()
    groups = discover(model, forward or default_forward(model))
    saved = 0
    for g in groups:
        grp = Exl3Group([m for _, m in g])
        for i, (name, m) in enumerate(g):
            _set(model, name, FusedMember(grp, i, m))
        saved += 3 * (len(g) - 1)
    return dict(groups=len(groups), members=sum(len(g) for g in groups), launches_saved=saved, fake_quant_memo=memo, window_math_shared=win, names=[[n for n, _ in g] for g in groups])

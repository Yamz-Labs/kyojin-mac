# SPDX-License-Identifier: Apache-2.0
"""S3: MoE prefill on the stacked (ragged-u32-v1) layout. Many tokens per call (a prompt chunk), not the 1-4 rows of the decode path.

Per layer and chunk of T tokens:
  1. host plan: the T * top-k (token, expert) pairs are sorted by expert; experts with no token are never touched
  2. active experts are cut into groups sized by a byte budget (decoded fp16 weights of gate + up + down of the group). On a GPU with
     neural accelerators (M5) the experts with few rows (<= KYOJIN_PF_FUSED_MAX, default 4096) skip the groups: step 3F
  3. per group: rot_pf (H128 of the token rows with the expert's suh) -> decode_pf (trellis -> fp16 weights, one launch for the whole group)
     -> mx.gather_mm (sorted expert index, fp32 accumulate, fp16 out) for gate and up -> mid_pf (swiglu, routing weight, fp8 fake quant,
     rotation for the down projection) -> decode_pf (down) -> gather_mm; mx.eval at the end of the group frees the decoded weights
  3F. fused (metal_nax, S6): rot_pf with the k permutation -> one launch for gate + up of ALL such experts, each simdgroup decoding the
     trellis in registers straight into the right input of a `matmul2d` (the decoded weights never reach memory) -> mid_pf -> one launch
     for the down projection. No group budget, no mx.eval inside. x2.5-4 on the Air for 8-128 rows per expert (decode is repeated per
     row block, so experts above the limit keep the group path, which decodes once).
  4. final_pf: unsort, H128, svh, routing weight, sum over the top-k in order (fp32)
Only the matmul differs from the decode path (fp16 inputs / outputs, fp32 accumulate inside the matmul); `prec="f32"` runs the same plan
with fp32 activations and weights (slow, 2x memory): it isolates plan / index bugs from fp16 rounding in the tests.

Peak extra memory = group budget + 3 pair buffers [pairs, hidden] fp16 (a 4096-token chunk with top-6 and hidden 4096: 200 MB each).
Two copy levers on the fused path (both default off, bit-identical, tests/test_pf_moe_copies.py): `pf_moe_rowmap` has the gate / up GEMM
read x through the slot -> token map with the transform in the tile load (`metal_nax.coop_rm_source`, no 2P x K rot copy) and
`pf_moe_epi` runs gate + up + SwiGLU + routing weight in one kernel (`metal_nax_gu`, no [2P, inter] product)."""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from . import metal_nax as NX, metal_pf as PF
from . import fired as FD
from .moe import MoeGather, Stack

GROUP_GIB = float(os.environ.get("KYOJIN_PF_GROUP_GIB", "2"))
GLU = os.environ.get("KYOJIN_PF_GLU", "1") != "0"      # default on (DS4.1 8192 chunk 2.99 -> 2.94 s, same logits); device sort (moe_sort) + double-copy + existing rot/NAX/mid/down kernels: no host argsort, no sorted-copy change
FUSED = "0" if os.environ.get("KYOJIN_PF_NAX") == "0" else os.environ.get("KYOJIN_PF_FUSED", "auto")                  # "1" force, "0" off, "auto" = on when the GPU has neural accelerators
FUSED_MAX = int(os.environ.get("KYOJIN_PF_FUSED_MAX", "4096"))      # experts with more rows than this use the decode + gather_mm group path
# fused path: one pass (rot, gate + up, mid, down) per bitrate class of the experts instead of one launch for all of them. A kernel that holds the
# decode bodies of several K runs every expert slower (DS4.1 K2 + K3 experts, 5120 x 2304, 32 rows each: 50.1 ms one launch vs 21.9 + 13.6 split).
# Same tiles (one cfg for all classes), same per-row arithmetic: bit-identical. 0 = one launch for all bitrates.
PF_SPLIT_K = os.environ.get("KYOJIN_PF_SPLIT_K", "1") != "0"
# fused path: gate and up as two nproj=1 launches instead of one nproj=2 launch in one eval. Same tiles, same per-row arithmetic (disjoint outputs):
# bit-identical. Kernel-isolated +20-40 pct at production rows (384 experts: uniform M=128 68.4 -> 57.4 ms, skewed 8K 68.7 -> 55.9 ms, all buckets
# ratio 0.70-0.85, tight paired spreads); but in the full prefill-pass eval the two halves sharing one eval with rot are bimodal (median 1.25x
# slower, 44 ms or 90 ms), so this stays OPT-IN (KYOJIN_PF_SPLIT_GU=1) until it validates on a free machine. 0 = one combined launch (default).
SPLIT_GU = os.environ.get("KYOJIN_PF_SPLIT_GU", "0") != "0"
# pf_moe_rowmap / pf_moe_epi: read per call (the flags module rewrites these on POST /admin/flags), default off.
MOE_ROWMAP = os.environ.get("KYOJIN_PF_MOE_ROWMAP", "0") != "0"    # gate / up GEMM reads x through the slot -> token map, transform in the tile load
MOE_EPI = os.environ.get("KYOJIN_PF_MOE_EPI", "0") != "0"          # SwiGLU + routing weight in the GEMM epilogue (no [2P, inter] product)
# pf_moe_nosync: the device sort's `cum` stays on the device. The per-layer readback that sized the per-class buffers and their row offsets is gone:
# buffers are sized from the pair count (P = T * topk, known on the host), the gate / up and down GEMMs walk each expert's row range with one
# threadgroup (`persist`, grid.y = 1) and write at the GLOBAL slot index, the mid kernel is given its class's slot range as a device value, and the
# final combine picks the class block with a compare against cum. Same tiles, same per-row arithmetic -> bit-identical. Memory: the pair buffers are
# P-sized instead of class-sized, so the path runs only up to NOSYNC_MAX pairs (a 8192-row chunk at topk 6 is 49152).
MOE_NOSYNC = os.environ.get("KYOJIN_PF_MOE_NOSYNC", "0") != "0"
# pf_moe_epi_cap: the epilogue kernel's output (the SwiGLU product the down GEMM reads) is allocated at the bucketed row count `cap_rows(n)` like every other pair buffer
# of the path, instead of n rows. n (the class's pairs) changes at every layer, so an exact-size buffer never matches a cached one: a fresh allocation per class per layer.
# Same bits (rows >= n are never written or read). Read per call.
EPI_CAP = os.environ.get("KYOJIN_PF_EPI_CAP", "0") != "0"
# pf_nosync_rowend: pf_moe_nosync's gate / up GEMM gets a row range closed at the class end for the last gate expert. The glued range
# `concatenate([cl[:-1], cl + P])` ends that expert at (class start + P), so it walks P - (rows of the class before it) slots of garbage on PERSIST_Y
# threadgroups, at every class of every layer; the output rows are the same either way. Needs the kernel built with rowend (metal_nax.coop_source). Read per call.
NOSYNC_ROWEND = os.environ.get("KYOJIN_PF_NOSYNC_ROWEND", "0") != "0"
# pf_moe_one_pass: one pass over the expert banks per call (up to the 8192-row cap) instead of splitting the call at `chunk` rows. Read per call.
ONE_PASS = os.environ.get("KYOJIN_PF_ONE_PASS", "0") != "0"
# pf_moe_hoist: a call of more than one pass (the served 8192-row chunk is two 4096-row passes) plans every pass on the device and reads ALL the `cum` offsets back in ONE sync, then
# builds pass 1, SUBMITS it (async_eval) and builds pass 2 while the GPU runs pass 1. Today each pass is plan -> sync -> build -> sync(y): the GPU idles while the host builds a pass
# and again over the plan round trip of the next one (3 gaps a layer; the second pass's build never overlaps GPU work). Same kernels on the same inputs in the same order, so the
# output is bit-identical; pass 2's pair buffers are allocated while pass 1's are still live (about one extra pass of pair buffers in flight, ~1.1 GB at 4096 rows x top-6). Read per call.
HOIST = os.environ.get("KYOJIN_PF_MOE_HOIST", "0") != "0"
# cpfmoe: with pf_moe_nosync a call of several passes submits each pass with async_eval instead of mx.eval, so the host encodes pass k + 1 (no readback inside a nosync pass) while the GPU
# runs pass k. Same kernels on the same inputs in the same order: bit-identical. Costs about one extra pass of P-sized pair buffers in flight. Needs MOE_NOSYNC; read per call.
CPFMOE = os.environ.get("KYOJIN_CPFMOE", "0") != "0"
PASS_CAP = 8192                                          # rows of one pass: 16384-row passes panicked the Ultra (29/10 00:26)
NOSYNC_MAX = 8192 * 6
# per-bitrate pair buffers (rot copy, gate/up, mid, down) are allocated at a bucketed row count: the pair count of a class changes every layer and
# chunk, and the MLX buffer cache only reuses a free buffer of (almost) the same byte size, so exact sizes miss it and pay a fresh allocation each
# time. Buckets per octave (4: <= 25 % spare rows, never computed or read); 0 = exact sizes.
# pf_moe_hot_fused: with a MoE pass of 8192 rows (pf_moe_chunk) one expert can take more than FUSED_MAX rows (a token set that routes alike, e.g. repeated
# filler); the device-plan path then returns None AFTER its plan + cum readback and the whole layer reruns on the host plan (np.array(ids) sync, host argsort,
# uploads). A 4096-row pass can never exceed FUSED_MAX, which is why only the 8192 pass paid. 1 = the device-plan path keeps such experts in the fused
# launch (limit = max(FUSED_MAX, pass rows)); their row blocks each decode the weights again. Default 0: unchanged. Not bit-identical for those experts.
HOT_FUSED = os.environ.get("KYOJIN_PF_HOT_FUSED", "0") != "0"
# pf_moe_bail_split: a pass that bails (an expert above FUSED_MAX rows) reruns as passes of FUSED_MAX rows on the device plan, the passes pf_moe_chunk=4096 runs,
# instead of the host plan. A pass of <= FUSED_MAX rows cannot bail (one expert takes at most one pair per row). Default 0.
BAIL_SPLIT = os.environ.get("KYOJIN_PF_BAIL_SPLIT", "0") != "0"
# pf_moe_rm_rows: pf_moe_rowmap / pf_moe_epi row map only for a bitrate class whose active experts average at most this many rows (0 = always).
RM_ROWS = int(os.environ.get("KYOJIN_PF_RM_ROWS", "0"))
CAP_STEPS = int(os.environ.get("KYOJIN_PF_CAP", "4"))
_kc: dict = {}


def cap_rows(n: int) -> int:
    """Rows to allocate for `n` pair rows (>= n): n rounded up to one of CAP_STEPS sizes per octave; n itself when CAP_STEPS <= 0 or n < 256."""
    if CAP_STEPS <= 0 or n < 256:
        return n
    step = max(1, (1 << (n.bit_length() - 1)) // CAP_STEPS)
    return -(-n // step) * step


def rm_allowed(pairs: int, active: int, limit: int) -> bool:
    """pf_moe_rm_rows: whether a class of `pairs` rows over `active` experts may use the row map (limit 0 = always)."""
    return limit <= 0 or pairs <= limit * max(1, active)


def fused_default() -> bool:
    """Whether a new `MoePrefill` runs the fused EXL3 + NAX path (KYOJIN_PF_FUSED, auto = GPU with neural accelerators)."""
    return NX.site_on(FUSED)


def _kernel(key, name, **kw):
    if key not in _kc:
        _kc[key] = mx.fast.metal_kernel(header=PF.HEADER_PF, name=f"{name}_{len(_kc)}", **kw)
    return _kc[key]


def decode_group(stacks: tuple[Stack, ...], eids: mx.array, cb: str) -> mx.array:
    """Wr^T fp16 [len(stacks) * G, N, KD] for the experts `eids` (i32 [G]) of every projection in `stacks` (same shape, same codebook)."""
    s0, np_ = stacks[0], len(stacks)
    G = int(eids.shape[0])
    Ks = tuple(sorted(set().union(*[set(s.ks) for s in stacks])))
    key = ("dec", Ks, np_, cb)
    names = ["eids", "tr0"] + (["tr1"] if np_ == 2 else []) + ["off0"] + (["off1"] if np_ == 2 else []) + \
            ["kk0"] + (["kk1"] if np_ == 2 else []) + ["params"]
    kern = _kernel(key, f"kyojin_decpf_{cb}_{'_'.join(map(str, Ks))}_p{np_}", input_names=names, output_names=["w"],
                   source=PF.decode_pf_source(Ks, np_, cb))
    ins = [eids] + [s.words for s in stacks] + [s.offset for s in stacks] + [s.k for s in stacks] + [mx.array([s0.kt, s0.nt, G], dtype=mx.uint32)]
    return kern(inputs=ins, grid=(s0.kt * s0.nt * 8, np_ * G, 1), threadgroup=(256, 1, 1),
                output_shapes=[(np_ * G, s0.nout, s0.kin)], output_dtypes=[mx.float16])[0]


class MoePrefill:
    """Same contract as `MoeGather.__call__` (x [T, hidden] fp16 / fp32, ids [T, topk], wts [T, topk] -> fp32 [T, hidden]) for any T."""

    def __init__(self, moe: MoeGather, group_gib: float | None = None, prec: str = "f16", chunk: int | None = None,
                 fused: bool | None = None, fused_max: int | None = None, nax: bool | None = None):
        assert prec in ("f16", "f32")
        self.m, self.prec = moe, prec
        if fused is None:
            fused = fused_default() if nax is None else nax                 # nax= is the feat/macmoepf name of fused=
        self.prof: dict | None = None                                       # set to {} to time the stages of the fused path (adds a sync per stage)
        self.fused = bool(fused) and prec == "f16" and NX.site_on("1")      # an explicit fused=True on a GPU without matmul2d takes the group path and NX.site_on("1")      # an explicit fused=True on a GPU without matmul2d takes the group path
        self.nax = self.fused
        self.fused_max = FUSED_MAX if fused_max is None else fused_max
        self.group_bytes = int((GROUP_GIB if group_gib is None else group_gib) * 2 ** 30)
        self._chunk_rows = chunk
        self.h = prec == "f16"
        self.last: dict = {}
        self.bailed = 0

    @property
    def chunk(self) -> int:
        """Rows of one MoE pass (KYOJIN_PF_CHUNK, read per call), capped: 16384-row passes panicked the Ultra (29/10 00:26)."""
        return min(self._chunk_rows or int(os.environ.get("KYOJIN_PF_CHUNK", "4096")), PASS_CAP)

    def pass_rows(self, rows: int) -> int:
        """Rows of the next MoE pass for a call of `rows` rows: `chunk` rows, or (pf_moe_one_pass) the whole call up to the cap, so a
        prefill chunk sweeps every expert bank once. The bank is swept once per pass and every row block of an expert decodes its weights
        again, so a second pass re-decodes what the first one decoded (metal_nax.coop_source)."""
        return min(rows, PASS_CAP) if ONE_PASS else self.chunk

    @staticmethod
    def _blk128() -> bool:
        """The 128-row block (pf_blk128) is offered to pick_cfg only for the plain GEMM pair: the row map stages its A block in threadgroup memory and the epilogue
        kernel doubles the accumulators, neither is sized for four 32-row sub-blocks."""
        return not (MOE_ROWMAP or MOE_EPI)

    def glu_limit(self) -> int:
        """Rows above which an expert sends the device-plan path to the host plan."""
        return max(self.fused_max, self.chunk) if HOT_FUSED else self.fused_max

    # ---- host plan
    def _plan(self, ids_np: np.ndarray):
        """Pairs sorted by processing rank: with the fused path the experts above `fused_max` rows come first (group path), then the others
        (fused launch), each class by expert id. Returns order, proc (experts in processing order), cnt / cum (rows per expert / offsets in
        that order), inv, nhot (experts that take the group path)."""
        E = self.m.g.experts
        flat = ids_np.reshape(-1).astype(np.int64)
        counts = np.bincount(flat, minlength=E)
        active = np.nonzero(counts)[0]
        if self.fused:
            hot = counts[active] > self.fused_max
            cold = active[~hot]
            if PF_SPLIT_K:                                                  # fused experts grouped by bitrate class (then by id): one pass per class
                cold = cold[np.lexsort((cold, self._kclass()[cold]))]
            proc = np.concatenate([active[hot], cold])
            nhot = int(hot.sum())
        else:
            proc, nhot = active, len(active)
        rank = np.zeros(E, np.int64)
        rank[proc] = np.arange(len(proc))
        order = np.argsort(rank[flat], kind="stable")
        cnt = counts[proc]
        cum = np.concatenate([[0], np.cumsum(cnt)])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return order, proc, cnt, cum, inv, nhot

    def _kclass(self) -> np.ndarray:
        """Per expert: an id of its (gate, up, down) bitrate triple."""
        if getattr(self, "_kc_np", None) is None:
            m = self.m
            trip = np.stack([m.g.kh, m.u.kh, m.d.kh], 1)
            self._kc_np = np.unique(trip, axis=0, return_inverse=True)[1].reshape(-1)
        return self._kc_np

    def _st(self, name: str, arr, t0: float) -> float:
        """Stage timer of the profiling mode: sync, add the elapsed time to prof[name], return the new start."""
        import time
        mx.eval(arr)
        t1 = time.perf_counter()
        self.prof[name] = self.prof.get(name, 0.0) + t1 - t0
        return t1

    def group_size(self) -> int:
        g, d = self.m.g, self.m.d
        per_expert = (2 * g.kin * g.nout + d.kin * d.nout) * (2 if self.h else 4)
        return max(1, self.group_bytes // per_expert)

    def _mm(self, a: mx.array, w: mx.array, ridx: mx.array) -> mx.array:
        """a [n, 1, K] x w [E, N, K]^T gathered by the sorted expert index -> [n, 1, N]."""
        if not self.h:
            w = w.astype(mx.float32)
        # f32 plan check: the sorted fp32 kernel rounds through the M5 tensor units (rel 8e-4 on random data), the unsorted one is exact
        return mx.gather_mm(a, w.swapaxes(-1, -2), rhs_indices=ridx, sorted_indices=self.h)

    def _rot(self, x, tok_s, ids_s, n, ih, oh, perm, cap=None):
        m = self.m
        kr = _kernel(("rot", ih, oh, perm), "kyojin_rotpf", input_names=["x", "tok", "ids", "suh0", "suh1", "params"], output_names=["xh"],
                     source=PF.rot_pf_source(2, ih, oh, perm))
        grid, tgp = _had(m.g.kin // 128, 2 * n)
        return kr(inputs=[x, tok_s, ids_s, m.g.suh, m.u.suh, mx.array([m.g.kin, n], dtype=mx.uint32)], grid=grid, threadgroup=tgp,
                  output_shapes=[(2 * (cap or n), m.g.kin)], output_dtypes=[mx.float16 if oh else mx.float32])[0]

    def _mid(self, inner, ids_s, wts_s, n, oh, perm, cap=None, bounded=None):
        m = self.m
        km = _kernel(("mid", m.limit, m.wpre, m.fq, m.alpha, m.beta, oh, perm, bounded is not None), "kyojin_midpf",
                     input_names=["part", "ids", "wts", "svh0", "svh1", "suhd", "params"] + (["rows"] if bounded is not None else []),
                     output_names=["xh2"],
                     source=PF.mid_pf_source(m.limit, m.wpre, m.fq, m.alpha, m.beta, oh, oh, perm, bounded=bounded is not None))
        rows = n if bounded is None else bounded[1]
        grid, tgp = _had(m.g.nout // 128, rows)
        return km(inputs=[inner, ids_s, wts_s, m.g.svh, m.u.svh, m.d.suh, mx.array([n, m.g.nout], dtype=mx.uint32)]
                  + ([bounded[0]] if bounded is not None else []), grid=grid, threadgroup=tgp,
                  output_shapes=[(cap or n, m.g.nout)], output_dtypes=[mx.float16 if oh else mx.float32])[0]

    def _chunk(self, x: mx.array, ids: mx.array, wts: mx.array) -> mx.array:
        if GLU and self.fused and self.h:
            y = self._chunk_glu(x, ids, wts)
            if y is None and BAIL_SPLIT and x.shape[0] > self.fused_max:
                FD.hit("pf_moe_bail_split")
                y = self._bail_split(x, ids, wts)
            if y is not None:
                FD.hit("pf_glu")
                return y
        elif GLU:
            FD.hit("pf_glu_miss")
        return self._chunk_host(x, ids, wts)

    def _chunk_host(self, x: mx.array, ids: mx.array, wts: mx.array) -> mx.array:
        """The host-plan path (experts above fused_max rows take the decode + gather_mm groups, the others the fused launch)."""
        m = self.m
        T, topk = ids.shape
        P = T * topk
        ids_np = np.array(ids)
        wts_np = np.array(wts).astype(np.float32)
        order, proc, cnt, cum, inv, nhot = self._plan(ids_np)
        G = self.group_size()
        flat_e = ids_np.reshape(-1)[order].astype(np.int32)
        flat_w = wts_np.reshape(-1)[order]
        tok = (order // topk).astype(np.uint32)
        ih, oh = x.dtype == mx.float16, self.h
        g, u, d = m.g, m.u, m.d
        outs, ngroups = [], 0
        for a in range(0, nhot, G):
            ex = proc[a:min(a + G, nhot)]
            s0, s1 = int(cum[a]), int(cum[a + len(ex)])
            n = s1 - s0
            lidx_np = np.repeat(np.arange(len(ex), dtype=np.uint32), cnt[a:a + len(ex)])
            ids_s, wts_s, tok_s = mx.array(flat_e[s0:s1]), mx.array(flat_w[s0:s1]), mx.array(tok[s0:s1])
            lidx, eids = mx.array(lidx_np), mx.array(ex.astype(np.int32))
            Gl = len(ex)
            # rot (gate and up share one launch), decode both projections of the group, two gathered matmuls in one call
            xh = self._rot(x, tok_s, ids_s, n, ih, oh, False)
            wgu = decode_group((g, u), eids, m.cb)
            if not self.h:
                xh = xh.astype(mx.float32)
            inner = self._mm(xh.reshape(2 * n, 1, g.kin), wgu, mx.concatenate([lidx, lidx + Gl])).reshape(2 * n, g.nout)
            xh2 = self._mid(inner, ids_s, wts_s, n, oh, False)
            wd = decode_group((d,), eids, m.cb)
            out = self._mm(xh2.reshape(n, 1, d.kin), wd, lidx).reshape(n, d.nout)
            mx.eval(out)                                                  # frees the decoded weights of this group
            outs.append(out)
            ngroups += 1
        nf = len(proc) - nhot
        cfg = NX.pick_cfg(cnt[nhot:], self._blk128()) if nf else None                    # one tile config for every bitrate pass: same bits as one launch
        kc = self._kclass()[proc[nhot:]] if nf and PF_SPLIT_K else np.zeros(nf, np.int64)
        for c0 in np.flatnonzero(np.r_[True, kc[1:] != kc[:-1]]) if nf else ():   # fused EXL3 + NAX path for the experts with few rows, per bitrate class
            c1 = c0 + int((kc[c0:] == kc[c0]).sum()) if PF_SPLIT_K else nf
            a0, a1 = nhot + c0, nhot + c1
            s0, s1 = int(cum[a0]), int(cum[a1])
            n = s1 - s0
            ex, c = proc[a0:a1], cnt[a0:a1]
            ids_s, wts_s, tok_s = mx.array(flat_e[s0:s1]), mx.array(flat_w[s0:s1]), mx.array(tok[s0:s1])
            eids = mx.array(ex.astype(np.int32))
            cl = (cum[a0:a1 + 1] - s0).astype(np.uint32)
            mr = int(c.max())
            kgu = tuple(sorted(set(g.kh[ex].tolist()) | set(u.kh[ex].tolist())))
            kd = tuple(sorted(set(d.kh[ex].tolist())))
            pr = self.prof
            if pr is not None:
                import time
                t0 = time.perf_counter()
            nc = n if SPLIT_GU else cap_rows(n)
            xh = self._rot(x, tok_s, ids_s, n, ih, oh, True, nc)
            if pr is not None:
                t0 = self._st("rot", xh, t0)
            if SPLIT_GU:
                FD.hit("pf_split_gu")
                ksg = tuple(sorted(set(g.kh[ex].tolist())))
                ksu = tuple(sorted(set(u.kh[ex].tolist())))
                inner_g = NX.expert_gemm((g,), eids, mx.array(cl), xh[:n], n, mr, cfg, ks=ksg)
                inner_u = NX.expert_gemm((u,), eids, mx.array(cl), xh[n:], n, mr, cfg, ks=ksu)
                inner = mx.concatenate([inner_g, inner_u], axis=0)
            else:
                inner = NX.expert_gemm((g, u), eids, mx.array(np.concatenate([cl[:-1], n + cl]).astype(np.uint32)), xh, 2 * nc, mr, cfg, ks=kgu)
            if pr is not None:
                t0 = self._st("gate_up", inner, t0)
            xh2 = self._mid(inner, ids_s, wts_s, n, oh, True, nc)
            if pr is not None:
                t0 = self._st("act_rot", xh2, t0)
            down = NX.expert_gemm((d,), eids, mx.array(cl), xh2, nc, mr, cfg, ks=kd)
            outs.append(down if nc == n else down[:n])
            if pr is not None:
                self._st("down", outs[-1], t0)
        part = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        kf = _kernel(("fin", not m.wpre, oh), "kyojin_finpf", input_names=["part", "inv", "ids", "wts", "svh", "params"], output_names=["y"],
                     source=PF.final_pf_source(not m.wpre, oh))
        grid, tgp = _had(d.blocks, T)
        y = kf(inputs=[part, mx.array(inv.astype(np.uint32)), mx.array(ids_np.reshape(-1).astype(np.int32)), mx.array(wts_np.reshape(-1)), d.svh,
                       mx.array([1, d.nout, topk], dtype=mx.uint32)], grid=grid, threadgroup=tgp,
               output_shapes=[(T, d.nout)], output_dtypes=[mx.float32])[0]
        self.last = dict(groups=ngroups, group_experts=G, active=int(len(proc)), pairs=P, fused_experts=nf)
        return y

    def _glu_plan(self, ids: mx.array, wts: mx.array):
        """The device plan of one pass: (tok_s, ids_s, wts_s, inv, cum), experts ranked by (bitrate class, id). Lazy: nothing is evaluated."""
        from . import moe_sort as MS
        m = self.m
        E = m.g.experts
        kc = self._kclass()
        if getattr(self, "_glu_rank", None) is None or self._glu_rank[0] != E or self._glu_rank[1] is not kc:
            proc_all = np.lexsort((np.arange(E), kc))
            rank_h = np.zeros(E, np.uint32)
            rank_h[proc_all] = np.arange(E)
            rank_d = mx.array(rank_h)
            mx.eval(rank_d)
            self._glu_rank = (E, kc, proc_all, rank_d)
        rank_d = self._glu_rank[3]
        return MS.plan_device(ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32), E, ids.shape[1], rank_d)

    def _bail_split(self, x: mx.array, ids: mx.array, wts: mx.array) -> mx.array:
        """The rows of a bailed pass as passes of `fused_max` rows: each stays on the device plan (no expert can exceed the limit), one eval per pass like __call__."""
        step = self.fused_max
        ys = []
        for a in range(0, x.shape[0], step):
            y = self._chunk(x[a:a + step], ids[a:a + step], wts[a:a + step])
            mx.eval(y)
            ys.append(y)
        return ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=0)

    def _chunk_glu(self, x: mx.array, ids: mx.array, wts: mx.array, plan=None, counts=None) -> mx.array:
        """Device-plan + double-copy path (TF lever 5, first half, per-bitrate passes): the device plan (moe_sort, no host
        sync) replaces the host argsort; rot writes the gate/up-normalized [2P, K] copy (gate rows with g.suh, up rows with u.suh)
        which ONE nproj=2 NAX GEMM consumes; mid / down / combine stay. Bit-identical to the
        fused path (same arithmetic; slot order up to a consistent permutation). None when an expert holds more than fused_max rows (host plan)."""
        m = self.m
        T, topk = ids.shape
        P = T * topk
        E = m.g.experts
        g, u, d = m.g, m.u, m.d
        ih, oh = x.dtype == mx.float16, self.h
        # rank experts by (kclass, id) so cum follows the per-bitrate passes (same as _plan
        # with PF_SPLIT_K=1); one kernel holds one K body per pass, not the union (50.1 ms
        # mixed vs 21.9+13.6 split at DS4.1 shapes).
        kc = self._kclass()
        tok_s, ids_s, wts_s, inv, cum = self._glu_plan(ids, wts) if plan is None else plan      # plan / counts: given by the hoisted pass loop (pf_moe_hoist), already evaluated
        proc_all = self._glu_rank[2]
        if MOE_NOSYNC and not (ih and oh and not SPLIT_GU and not MOE_ROWMAP and not MOE_EPI and P <= NOSYNC_MAX
                               and g.kin % 128 == 0 and m.g.nout % 128 == 0):
            FD.hit("pf_moe_nosync_miss")
            if not ih:
                FD.hit("pf_moe_nosync_x_not_f16")
            if SPLIT_GU or MOE_ROWMAP or MOE_EPI:
                FD.hit("pf_moe_nosync_other_flag")
            if P > NOSYNC_MAX:
                FD.hit("pf_moe_nosync_P_gt_max")
            if g.kin % 128 or m.g.nout % 128:
                FD.hit("pf_moe_nosync_shape")
        if (MOE_NOSYNC and ih and oh and not SPLIT_GU and not MOE_ROWMAP and not MOE_EPI and P <= NOSYNC_MAX
                and g.kin % 128 == 0 and m.g.nout % 128 == 0):
            kcs = kc[proc_all]
            cuts = np.flatnonzero(np.r_[True, kcs[1:] != kcs[:-1]])
            if len(cuts) > 4:
                FD.hit("pf_moe_nosync_miss")
                FD.hit("pf_moe_nosync_cuts_gt4")
            if len(cuts) <= 4:                                        # the final combine picks among at most 4 class blocks
                FD.hit("pf_moe_nosync")
                return self._chunk_glu_nosync(x, ids, wts, tok_s, ids_s, wts_s, inv, cum, proc_all, kc, cfg_pos=cuts)
        # one small D2H of cum [E+1] for the tile pick + row blocks (no ids sync, no argsort). `pf_moe_nosync` (default off) removes this
        # readback: it keeps the same passes but sizes the buffers from P, walks each expert's rows on the device and picks the class block in the
        # final combine, at the cost of P-sized pair buffers instead of class-sized ones (see _chunk_glu_nosync).
        import numpy as _np
        if counts is None:
            mx.eval(cum)
            counts = _np.diff(_np.array(cum, copy=False))
        if counts.max() > self.glu_limit():                               # hot experts take the group path: host plan (None)
            FD.hit("pf_moe_hot_fused_bail")
            self.bailed += 1                                              # layers that paid the plan + readback for nothing (see HOT_FUSED)
            return None
        if (counts == 0).all():
            return mx.zeros((T, m.d.nout), dtype=mx.float32)
        cfg = NX.pick_cfg(counts, self._blk128())
        if HOT_FUSED and counts.max() > self.fused_max:
            FD.hit("pf_moe_hot_fused")                                   # an expert above fused_max stays in the fused launch
        if NX.BLK128:
            FD.hit("pf_blk128" if cfg == NX._CFG_128 else "pf_blk128_miss")
        pr = self.prof
        if pr is not None:
            import time
            t0 = time.perf_counter()
        kcs = kc[proc_all]
        outs = []
        for c0 in _np.flatnonzero(_np.r_[True, kcs[1:] != kcs[:-1]]):
            c1 = c0 + int((kcs[c0:] == kcs[c0]).sum())
            s0, s1 = int(counts[:c0].sum()), int(counts[:c1].sum())
            n = s1 - s0
            if n == 0:
                continue
            ex = proc_all[c0:c1]
            c = counts[c0:c1]
            mr = int(c.max())
            tok_c, ids_c, wts_c = tok_s[s0:s1], ids_s[s0:s1], wts_s[s0:s1]
            nc = cap_rows(n)
            eids = mx.array(ex.astype(np.int32))
            cl = (cum[c0:c1 + 1] - s0).astype(mx.uint32)
            rowoff2 = mx.concatenate([cl[:-1], cl + n])
            kgu = tuple(sorted(set(g.kh[ex].tolist()) | set(u.kh[ex].tolist())))
            # pf_moe_rowmap: transform the token rows in the gate / up tile loads, no [2P, K] rot copy.
            rmap = MOE_ROWMAP and ih and oh and m.g.kin % 128 == 0 and rm_allowed(n, int((c > 0).sum()), RM_ROWS)
            # pf_moe_epi: SwiGLU + routing weight in the GEMM epilogue, no [2P, inter] product written then read.
            epi = MOE_EPI and ih and oh and not SPLIT_GU
            if rmap:
                FD.hit("pf_moe_rowmap")
            elif MOE_ROWMAP:
                FD.hit("pf_moe_rowmap_miss")
            if epi:
                FD.hit("pf_moe_epi")
            elif MOE_EPI:
                FD.hit("pf_moe_epi_miss")
            xh = None if rmap else self._rot(x, tok_c, ids_c, n, ih, oh, True, nc)
            if pr is not None and c0 == 0:
                t0 = self._st("rot", xh, t0) if xh is not None else t0
            elif pr is not None and xh is not None:
                mx.eval(xh)
            if epi:
                from . import metal_nax_gu as GU
                xh2 = GU.fused_gu_swiglu(g, u, d, eids, ex, cl, xh, n, mr, cfg, wts_c, m,
                                         rowmap=(x, tok_c, g.suh, u.suh) if rmap else None, out_rows=nc if EPI_CAP else None)
                inner = None
            else:
                tok2 = mx.concatenate([tok_c, tok_c]) if rmap else None
                inner = NX.expert_gemm((g, u), eids, rowoff2, xh, 2 * nc, mr, cfg, ks=kgu,
                                       rowmap=(x, tok2, g.suh, u.suh) if rmap else None)
            if pr is not None and c0 == 0:
                t0 = self._st("gate_up", inner if inner is not None else xh2, t0)
            elif pr is not None:
                mx.eval(inner if inner is not None else xh2)
            if not epi:
                xh2 = self._mid(inner, ids_c, wts_c, n, oh, True, nc)
            if pr is not None and c0 == 0:
                t0 = self._st("act_rot", xh2, t0)
            elif pr is not None:
                mx.eval(xh2)
            kd = tuple(sorted(set(d.kh[ex].tolist())))
            down = NX.expert_gemm((d,), eids, cl, xh2, nc, mr, cfg, ks=kd)
            outs.append(down if nc == n else down[:n])
            if pr is not None and c0 == 0:
                t0 = self._st("down", outs[-1], t0)
            elif pr is not None:
                mx.eval(outs[-1])
        down = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        kf = _kernel(("fin", not m.wpre, oh), "kyojin_finpf", input_names=["part", "inv", "ids", "wts", "svh", "params"],
                     output_names=["y"], source=PF.final_pf_source(not m.wpre, oh))
        grid, tgp = _had(d.blocks, T)
        y = kf(inputs=[down, inv, ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32), d.svh,
                       mx.array([1, d.nout, topk], dtype=mx.uint32)], grid=grid, threadgroup=tgp,
               output_shapes=[(T, d.nout)], output_dtypes=[mx.float32])[0]
        if pr is not None:
            self._st("final", y, t0)
        self.last = dict(groups=0, group_experts=0, active=int((counts > 0).sum()), pairs=P, fused_experts=E)
        return y

    def _chunk_glu_nosync(self, x: mx.array, ids: mx.array, wts: mx.array, tok_s, ids_s, wts_s, inv, cum,
                          proc_all: np.ndarray, kc: np.ndarray, cfg_pos: np.ndarray) -> mx.array:
        """pf_moe_nosync: the same per-bitrate passes as `_chunk_glu` with every row bound read on the device (`cum` is never evaluated on the host).

        Layout: the rows of all passes are global sorted slots [0, P). `rot` is one launch over all P pairs; each class's gate / up pass and down
        pass is a `persist` GEMM (one threadgroup per expert walks that expert's own row blocks, `rowoff` = the class's slice of `cum`) that writes
        at the global slot, so `mid` can be given the class's slot range as a device value and the buffers can be sized from P alone. `part` is the
        concat of the class blocks (block c = rows [c * cap, (c+1) * cap)), and the final combine turns a slot into `c * cap + slot`. Same tiles and
        the same per-row arithmetic as the sync path, so the output is bit-identical."""
        m = self.m
        T, topk = ids.shape
        P = T * topk
        E = m.g.experts
        g, u, d = m.g, m.u, m.d
        ih, oh = x.dtype == mx.float16, self.h
        cap = cap_rows(P)
        # tile config from the mean rows per expert: a perf-only choice (the tiles change no per-row arithmetic, so the bits do not move)
        cfg = NX.pick_cfg([max(1, P // max(1, E))] * E, self._blk128())
        bounds = [(int(cfg_pos[i]), int(cfg_pos[i + 1]) if i + 1 < len(cfg_pos) else int(len(proc_all))) for i in range(len(cfg_pos))]
        zero2 = mx.zeros((2,), dtype=mx.uint32)
        pr, t0 = self.prof, None
        if pr is not None:
            import time
            t0 = time.perf_counter()
        xh = self._rot(x, tok_s, ids_s, P, ih, oh, True, cap)
        if pr is not None:
            t0 = self._st("rot", xh, t0)
        outs = []
        for c0, c1 in bounds:
            ex = proc_all[c0:c1]
            eids = mx.array(ex.astype(np.int32))
            cl = cum[c0:c1 + 1]                                        # device slice of the sort's offsets: no host value
            kgu = tuple(sorted(set(g.kh[ex].tolist()) | set(u.kh[ex].tolist())))
            kd = tuple(sorted(set(d.kh[ex].tolist())))
            if NOSYNC_ROWEND:
                FD.hit("pf_nosync_rowend")
                rowoff2 = mx.concatenate([cl, cl + P])                # gate rows [cl[0], cl[GL]) and up rows [P + cl[0], P + cl[GL]), each range closed by its own end
            else:
                rowoff2 = mx.concatenate([cl[:-1], cl + P])           # gate rows [0, P), up rows [P, 2P); the last gate expert ends at cl[0] + P (see NOSYNC_ROWEND)
            inner = NX.expert_gemm((g, u), eids, rowoff2, xh, 2 * cap, 0, cfg, ks=kgu, persist=True, rowend=NOSYNC_ROWEND)
            if pr is not None:
                t0 = self._st("gate_up", inner, t0)
            rows = mx.concatenate([cl[:1], cl[-1:], zero2])            # (slot lo, slot hi) of this class, padded to 16 bytes for the device pointer
            xh2 = self._mid(inner, ids_s, wts_s, P, oh, True, cap, bounded=(rows, cap))
            if pr is not None:
                t0 = self._st("act_rot", xh2, t0)
            outs.append(NX.expert_gemm((d,), eids, cl, xh2, cap, 0, cfg, ks=kd, persist=True))
            if pr is not None and len(outs) == len(bounds):
                t0 = self._st("down", outs[-1], t0)
        part = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        kf = _kernel(("fin", not m.wpre, oh, len(bounds)), "kyojin_finpf",
                     input_names=["part", "inv", "ids", "wts", "svh", "params", "cum", "zb"], output_names=["y"],
                     source=PF.final_pf_source(not m.wpre, oh, classes=len(bounds)))
        grid, tgp = _had(d.blocks, T)
        zb = np.zeros(4, np.uint32)                                    # class -> first RANK position (cum is indexed by rank, not by expert id)
        zb[:len(bounds)] = [c0 for c0, _ in bounds]
        y = kf(inputs=[part, inv, ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32), d.svh,
                       mx.array([1, d.nout, topk, cap], dtype=mx.uint32), cum, mx.array(zb)],
               grid=grid, threadgroup=tgp, output_shapes=[(T, d.nout)], output_dtypes=[mx.float32])[0]
        if pr is not None:
            self._st("final", y, t0)
        self.last = dict(groups=0, group_experts=0, active=-1, pairs=P, fused_experts=E, nosync=True)
        return y

    def __call__(self, x: mx.array, ids: mx.array, wts: mx.array) -> mx.array:
        T, chunk = x.shape[0], self.pass_rows(x.shape[0])
        if chunk != 4096:
            FD.hit("pf_moe_chunk")                                    # a non-default pass size is in effect
        if T <= chunk:
            return self._chunk(x, ids.astype(mx.int32), wts)
        if HOIST and GLU and self.fused and self.h and not MOE_NOSYNC:
            return self._call_hoisted(x, ids.astype(mx.int32), wts, chunk)
        ys = []
        for a in range(0, T, chunk):
            y = self._chunk(x[a:a + chunk], ids[a:a + chunk].astype(mx.int32), wts[a:a + chunk])
            if CPFMOE and MOE_NOSYNC:
                FD.hit("cpfmoe")
                mx.async_eval(y)
            else:
                mx.eval(y)
            ys.append(y)
        return mx.concatenate(ys, axis=0)


    def _call_hoisted(self, x: mx.array, ids: mx.array, wts: mx.array, chunk: int) -> mx.array:
        """pf_moe_hoist: every pass planned on the device, ONE sync for the offsets of all of them, then each pass is built and submitted without waiting for the one before
        (the host builds pass k + 1 while the GPU runs pass k). A pass with a hot expert (None from `_chunk_glu`) takes the host plan, as in the pass loop."""
        T = x.shape[0]
        spans = [(a, min(a + chunk, T)) for a in range(0, T, chunk)]
        plans = [self._glu_plan(ids[a:b], wts[a:b]) for a, b in spans]
        mx.eval(*[p[4] for p in plans])
        counts = [np.diff(np.array(p[4], copy=False)) for p in plans]
        FD.hit("pf_moe_hoist")
        ys = []
        for (a, b), plan, c in zip(spans, plans, counts):
            y = self._chunk_glu(x[a:b], ids[a:b], wts[a:b], plan=plan, counts=c)
            if y is None and BAIL_SPLIT and b - a > self.fused_max:
                FD.hit("pf_moe_bail_split")
                y = self._bail_split(x[a:b], ids[a:b], wts[a:b])
            if y is None:
                y = self._chunk_host(x[a:b], ids[a:b], wts[a:b])
            else:
                FD.hit("pf_glu")
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys, axis=0)


def _had(blocks: int, rows: int):
    from . import metal_src as S
    return S.had_grid(blocks, rows)

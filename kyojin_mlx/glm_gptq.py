# SPDX-License-Identifier: Apache-2.0
"""G4c: GPTQ onto the NVFP4 grid (the layout of kyojin_mlx.glm_dense4: E2M1 codes, one UE4M3 scale per 16 inputs, one fp32 per-tensor scale).

The runtime reads the same bytes as for plain RTN, so a GPTQ pack is a drop-in for `glm_dense4_convert.py` output; only the choice of the 4-bit codes and block scales
changes: columns are quantised left to right and the rounding error of each column is pushed onto the not-yet-quantised columns through the inverse Hessian
H = X^T X of the layer's calibration inputs (Frantar et al., GPTQ), so the layer's OUTPUT error X (W - W^) shrinks instead of the weight error.

Choices that follow from the format:
* the block scale (UE4M3 code, 7 candidates around the amax scale, the same set as the RTN search) is chosen per 16 columns AT THAT POINT of the sweep, i.e. on the
  weights already compensated by the columns to the left, and by the quantity GPTQ minimises: the 7 candidates are run through the 16 sequential column steps together
  (vectorised on a leading axis, one compiled kernel graph per group) and the one with the smallest sum of e^2 (e = scaled error of the column step) wins;
* groups of 16 are contiguous in the stored layout, so act-order is done at GROUP granularity (whole 16-column groups sorted by descending diag(H) mass; the columns
  inside a group keep their order) and the result is un-permuted before packing;
* lazy batch updates: the error of a block of 128 columns reaches the columns beyond the block with ONE matmul (Err @ U[block, rest]);
* rows are independent given H, so a fused group (qkv, gate|up, q_a|kv_a) is one concatenated matrix with its shared per-tensor scale, and tall matrices (lm_head)
  are swept in row chunks.
"""
from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

from . import glm_dense4 as G4

GROUP = 16
_STEP = {}


# ---------------------------------------------------------------------------------------------------------------------------------- Hessian
def hess_factor(H: np.ndarray, damp: float = 0.01, damp_mode: str = "mean", perm: np.ndarray | None = None) -> mx.array:
    """H [k, k] (sum or mean of x x^T, any scale) -> U float32 [k, k] upper-triangular with U^T U = (H + damping)^-1 (GPTQ's `Hinv` Cholesky factor).
    damp_mode 'mean': + damp * mean(diag H) * I (the reference GPTQ); 'diag': + damp * diag(H) (relative to each channel; matters when a few massive channels dominate the mean).
    A zero diagonal (a channel never active in the calibration set) gets the mean diagonal and no cross terms. perm: columns reordered before the factorisation.
    The inverse is taken after a Jacobi scaling D^-1/2 H D^-1/2 (unit diagonal) so fp32 is enough on channels that span many decades; CPU stream (MLX linalg)."""
    H = np.array(H, np.float32, copy=True)
    k = H.shape[0]
    if perm is not None:
        H = H[np.ix_(perm, perm)]
    d = np.diag(H).copy()
    dead = ~(d > 0)
    mean_d = float(d[~dead].mean()) if (~dead).any() else 1.0
    if dead.any():
        H[dead, :] = 0
        H[:, dead] = 0
        H[dead, dead] = mean_d
        d[dead] = mean_d
    s = (1.0 / np.sqrt(d)).astype(np.float32)
    Hs = H * s[:, None] * s[None, :]                                           # unit diagonal
    base = (damp * mean_d * s * s) if damp_mode == "mean" else np.full(k, damp, np.float32)          # the damping in the scaled space
    cpu = mx.cpu                                                                # every op of the factorisation on the CPU stream: a 16384^2 elementwise kernel on a busy GPU hit the Metal command-buffer timeout
    for attempt in range(6):                                                    # on a failed factorisation: ten times the damping each retry
        Hd = Hs.copy()
        Hd[np.diag_indices(k)] += base * (10 ** attempt)
        with mx.stream(cpu):
            L = mx.linalg.cholesky(mx.array(Hd), stream=cpu)
            ok = bool(mx.all(mx.isfinite(L)).item())
        if ok:
            break
    else:
        raise RuntimeError("hess_factor: Hessian not positive definite after 6 damping retries")
    Hinv = mx.linalg.cholesky_inv(L, stream=cpu)                               # (Hs + damp)^-1
    with mx.stream(cpu):
        sm = mx.array(s)
        Hinv = Hinv * sm[:, None] * sm[None, :]                                # H^-1 = D^-1/2 Hs^-1 D^-1/2
        Hinv = (Hinv + Hinv.T) * 0.5
        mx.eval(Hinv)
    U = mx.linalg.cholesky(Hinv, upper=True, stream=cpu)
    mx.eval(U)
    return U


# ---------------------------------------------------------------------------------------------------------------------------------- one group of 16 columns
def _group_step(cands):
    """Compiled sweep of the 16 columns of one group for len(cands) scale candidates at once.
    (w16 [R, 16] compensated weights, Ug [16, 16] strictly upper, dinv [16], g scalar array: the per-tensor scale) -> (code uint8 [R, 16], scale byte uint8 [R], E [R, 16] scaled errors, Q [R, 16] dequantised)."""
    key = tuple(cands)
    if key in _STEP:
        return _STEP[key]
    grid, e4 = G4._tabs()
    off = mx.array(list(cands), mx.int32)
    C = len(cands)
    bnd = G4.E2M1_BOUNDS

    def step(w16, Ug, dinv, g):
        R = w16.shape[0]
        amax = mx.max(mx.abs(w16), axis=-1)                                      # [R]
        t = amax / (6.0 * g)
        b0 = mx.argmin(mx.abs(t[:, None] - e4[None, :]), axis=-1).astype(mx.int32)
        byte = mx.clip(b0[None, :] + off[:, None], 1, 126)                       # [C, R]
        s = mx.take(e4, byte) * g                                                # [C, R]
        w = mx.broadcast_to(w16[None], (C, R, GROUP))
        err = mx.zeros((C, R))
        qs, es, cs = [], [], []
        for j in range(GROUP):
            col = w[:, :, j]
            v = col / s
            a = mx.abs(v)
            idx = (a > bnd[0]).astype(mx.int32)
            for bd in bnd[1:]:
                idx = idx + (a > bd).astype(mx.int32)
            dq = mx.take(grid, idx)
            neg = v < 0
            q = mx.where(neg, -dq, dq) * s
            code = idx | mx.where(neg & (idx > 0), 8, 0)
            e = (col - q) * dinv[j]
            err = err + e * e
            if j < GROUP - 1:
                w = w - e[..., None] * Ug[j][None, None, :]
            qs.append(q); es.append(e); cs.append(code)
        best = mx.argmin(err, axis=0)                                            # [R]
        sel = lambda lst: mx.take_along_axis(mx.stack(lst, axis=-1), best[None, :, None], axis=0)[0]       # [C, R, 16] -> [R, 16]
        bbyte = mx.take_along_axis(byte, best[None, :], axis=0)[0]
        return sel(cs).astype(mx.uint8), bbyte.astype(mx.uint8), sel(es), sel(qs)

    fn = mx.compile(step)
    _STEP[key] = fn
    return fn


def gptq_rows(Wc: mx.array, U: mx.array, g: float, block: int = 128, cands=G4.NV_SEARCH):
    """Wc float32 [R, k] (columns already in the processing order), U [k, k] -> (codes uint8 [R, k], scale bytes uint8 [R, k/16], Q float32 [R, k] dequantised incl. the scale)."""
    R, k = Wc.shape
    assert k % GROUP == 0 and block % GROUP == 0
    step = _group_step(cands)
    garr = mx.array(g, mx.float32)
    dinv = 1.0 / mx.diag(U)
    Ustr = mx.triu(U, 1)
    codes, bytes_, Qs = [], [], []
    W = Wc
    for b0 in range(0, k, block):
        b1 = min(b0 + block, k)
        Wb = W[:, b0:b1]
        errs = []
        for gi in range(0, b1 - b0, GROUP):
            a, z = b0 + gi, b0 + gi + GROUP
            code, byte, E, Q = step(Wb[:, gi:gi + GROUP], Ustr[a:z, a:z], dinv[a:z], garr)
            if z < b1:
                Wb = mx.concatenate([Wb[:, :gi + GROUP], Wb[:, gi + GROUP:] - E @ U[a:z, z:b1]], axis=1)
            errs.append(E); codes.append(code); bytes_.append(byte); Qs.append(Q)
        if b1 < k:
            W = mx.concatenate([W[:, :b1], W[:, b1:] - mx.concatenate(errs, axis=1) @ U[b0:b1, b1:]], axis=1)
        mx.eval(W, *codes[-(b1 - b0) // GROUP:], *bytes_[-(b1 - b0) // GROUP:], *Qs[-(b1 - b0) // GROUP:])
    return mx.concatenate(codes, axis=1), mx.stack(bytes_, axis=1), mx.concatenate(Qs, axis=1)


# ---------------------------------------------------------------------------------------------------------------------------------- whole matrix
def group_perm(H: np.ndarray, k: int) -> np.ndarray:
    """Column order for act-order at group granularity: the 16-column groups by descending sum of diag(H); inside a group the order is kept."""
    gd = np.diag(H).reshape(k // GROUP, GROUP).sum(1)
    og = np.argsort(-gd, kind="stable")
    return (og[:, None] * GROUP + np.arange(GROUP)[None, :]).reshape(-1)


def rtn_deq(W: mx.array, g: float, chunk_elems: int = 1 << 21) -> mx.array:
    """The RTN (scale-searched) dequantisation of W in row chunks: the scale search holds 8x the chunk in temporaries (a [rows, k/16, 127] table), so keep the chunk small."""
    n, k = W.shape
    rows = max(1, chunk_elems // k)
    out = []
    for a in range(0, n, rows):
        d = G4.quant_nvfp4_rows(W[a:a + rows], g)[2]
        mx.eval(d)
        out.append(d)
    return mx.concatenate(out, axis=0)


def _out_err(D: mx.array, W: mx.array, H: mx.array):
    """sum over rows of (D H D^T) and (W H W^T), chunk-safe scalars."""
    return float(mx.sum((D @ H) * D)), float(mx.sum((W @ H) * W))


def gptq_matrix(w, H, g: float | None = None, damp: float = 0.01, damp_mode: str = "mean", act_order: bool = False, block: int = 128,
                row_chunk: int = 32768, cands=G4.NV_SEARCH, with_rtn: bool = True, return_deq: bool = False, U: mx.array | None = None, log=None):
    """w float32 [n, k] (numpy or mx), H [k, k] -> dict(qs uint8 [n, k/2], scale uint8 [n, k/16], scale2 float, rel_rmse, out_err (Hessian-weighted, relative; GPTQ),
    out_err_rtn (the same for plain RTN with the same g, when with_rtn), deq (float32 [n, k], when return_deq)). g: per-tensor scale (default amax / (6 * 448); the converter
    passes its g-boost-ed one). U: a precomputed hess_factor (must match the act_order permutation)."""
    W = mx.array(w) if not isinstance(w, mx.array) else w
    W = W.astype(mx.float32)
    n, k = W.shape
    assert k % 32 == 0
    Hn = np.asarray(H, np.float32)
    g = G4.global_scale(float(mx.max(mx.abs(W)))) if g is None else g
    perm = group_perm(Hn, k) if act_order else None
    t0 = time.time()
    if U is None:
        U = hess_factor(Hn, damp, damp_mode, perm)
    inv = None if perm is None else np.argsort(perm)
    Wp = W if perm is None else mx.take(W, mx.array(perm.astype(np.int32)), axis=1)
    qs_l, sc_l, num, den, e_g, e_r, d_h = [], [], 0.0, 0.0, 0.0, 0.0, 0.0
    Hm = mx.array(Hn)
    deq_l = []
    for a in range(0, n, row_chunk):
        wc = Wp[a:a + row_chunk]
        codes, byte, Q = gptq_rows(wc, U, g, block, cands)
        if perm is not None:                                                    # back to the stored column order; the bytes of a group travel with its columns
            gperm = perm.reshape(-1, GROUP)[:, 0] // GROUP
            ginv = np.argsort(gperm)
            codes = mx.take(codes, mx.array(inv.astype(np.int32)), axis=1)
            Q = mx.take(Q, mx.array(inv.astype(np.int32)), axis=1)
            byte = mx.take(byte, mx.array(ginv.astype(np.int32)), axis=1)
        wo = W[a:a + row_chunk]
        D = wo - Q
        eg, dh = _out_err(D, wo, Hm)
        e_g += eg; d_h += dh
        if with_rtn:
            e_r += _out_err(wo - rtn_deq(wo, g), wo, Hm)[0]
        num += float(mx.sum(D * D)); den += float(mx.sum(wo * wo))
        qs_l.append(G4.pack_nibbles(codes))
        sc_l.append(byte)
        if return_deq:
            deq_l.append(Q)
        mx.eval(qs_l[-1], sc_l[-1])
        if log:
            log(f"    rows {a}-{min(a + row_chunk, n)} of {n} [{time.time() - t0:.0f} s]")
    out = dict(qs=mx.concatenate(qs_l, axis=0), scale=mx.concatenate(sc_l, axis=0), scale2=float(g), rel_rmse=float(np.sqrt(num / max(den, 1e-30))),
               out_err=float(np.sqrt(e_g / max(d_h, 1e-30))), out_err_rtn=float(np.sqrt(e_r / max(d_h, 1e-30))) if with_rtn else None)
    if return_deq:
        out["deq"] = mx.concatenate(deq_l, axis=0)
    return out

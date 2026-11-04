#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXL3 conversion pipeline on MLX / Metal for dense Qwen3.5-style models (port of exllamav3 convert_model.py, mul1 codebook).

usage: python tools/conv_mlx.py SRC_DIR OUT_DIR --cal CAL.safetensors [--rows R] [--cols C] [--k-attn K] [--k-mlp K] [--head-bits 6]
                   [--strategy strategy.json] [--layers N] [--seed0 0]

Stages per decoder block (timed separately, see stage_times in the final log):
  capture   FP block forward over the calibration rows, H = sum x^T x for each linear input         (MLX, Metal)
  finalize  H/count + damping, sign flips + 128-Hadamards, Cholesky, block LDL (16)              (numpy CPU, float32)
  regular   output/input scales, Hadamards, global scale search (sampled tiles through the Viterbi)  (MLX + Metal Viterbi)
  ldlq      LDLQ over 16-row blocks, every block row = one Viterbi call over n/16 tiles             (MLX matmul + Metal Viterbi)
  pack      trellis bit packing                                                                      (numpy)
  advance   quantised block forward to produce the next block's calibration state                    (MLX, Metal)
The Viterbi (tail-biting trellis, 2^16 states, mul1 codebook) is conv_viterbi.py, bit-exact against the CUDA kernel.
Writes model.safetensors (+ config.json with quantization_config, tokenizer files) readable by kyojin_mlx.exl3_linear.load_exl3."""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, sys, time
from collections import defaultdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import conv_viterbi as V
from kyojin_mlx.vendor_src import load_source, src_items

HAD = 128
CODEBOOK_SCALE = 1.24371088
MUL1 = 0x83DCD12D
SIGMA_REG = 0.025
BUF_K = 128
STAGE = defaultdict(float)
QUANT_PREFIXES = ("linear_attn.in_proj_qkv.", "linear_attn.in_proj_z.", "linear_attn.out_proj.", "self_attn.q_proj.", "self_attn.k_proj.", "self_attn.v_proj.", "self_attn.o_proj.", "mlp.gate_proj.", "mlp.up_proj.", "mlp.down_proj.",
                  "mlp.shared_expert.gate_proj.", "mlp.shared_expert.up_proj.", "mlp.shared_expert.down_proj.", "mlp.experts.")
NORM_SFX = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


PEAK = defaultdict(float)


class T:
    """with T('stage'): ...  accumulates wall seconds and the MLX peak (GiB) per stage (callers mx.eval inside the block before leaving)."""
    def __init__(self, name): self.name = name
    def __enter__(self):
        mx.reset_peak_memory(); self.t0 = time.perf_counter()
    def __exit__(self, *a):
        STAGE[self.name] += time.perf_counter() - self.t0
        PEAK[self.name] = max(PEAK[self.name], mx.get_peak_memory() / 2**30)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem_gb():
    return mx.get_peak_memory() / 2**30


def mem_log(tag):
    """Process RSS (ps, GiB) next to the MLX active / peak memory: what the guard and `top` see against what MLX allocated."""
    import subprocess
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip() or 0) / 2**20
    log(f"mem {tag}: rss {rss:.2f} GiB, mlx active {mx.get_active_memory() / 2**30:.2f} peak {mx.get_peak_memory() / 2**30:.2f} cache {mx.get_cache_memory() / 2**30:.2f}")


# ---------------------------------------------------------------------------------------------- Hadamard / tile layout
def _sylvester(n):
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


HADM = mx.array(_sylvester(HAD) / math.sqrt(HAD))


def had_l(x):          # blockwise over rows: each 128-row block <- Had @ block
    k, n = x.shape
    return mx.matmul(HADM, x.reshape(k // HAD, HAD, n)).reshape(k, n)


def had_r(x):          # blockwise over columns: each 128-col block <- block @ Had
    k, n = x.shape
    return mx.matmul(x.reshape(k, n // HAD, HAD), HADM).reshape(k, n)


def _tc_perm():
    p = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9; c0 = t // 4; c1 = c0 + 8
        p[t * 8 + 0] = r0 * 16 + c0; p[t * 8 + 1] = r1 * 16 + c0; p[t * 8 + 2] = r2 * 16 + c0; p[t * 8 + 3] = r3 * 16 + c0
        p[t * 8 + 4] = r0 * 16 + c1; p[t * 8 + 5] = r1 * 16 + c1; p[t * 8 + 6] = r2 * 16 + c1; p[t * 8 + 7] = r3 * 16 + c1
    return np.array(p, np.int32)


PERM_NP = _tc_perm()
PERM = mx.array(PERM_NP)
PERM_I = mx.array(np.argsort(PERM_NP).astype(np.int32))


# ---------------------------------------------------------------------------------------------- packing
def pack_trellis(enc: np.ndarray, K: int, chunk: int = 512) -> np.ndarray:
    """uint16 [tk, tn, 256] (16-bit trellis states, tensor-core order) -> int16 [tk, tn, 16K]; same bytes as ext.pack_trellis."""
    tk, tn, _ = enc.shape
    if tn > chunk:                                       # the bit expansion below is K x 4 bytes per weight: 12 GiB for a 2048 x 248320 head at K6; tiles are independent
        return np.concatenate([pack_trellis(enc[:, c:c + chunk], K, chunk) for c in range(0, tn, chunk)], axis=1)
    v = (enc.reshape(-1, 256).astype(np.uint32)) & ((1 << K) - 1)
    bits = ((v[:, :, None] >> np.arange(K - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8).reshape(-1, 256 * K)
    by = np.packbits(bits, axis=1).reshape(-1, 8 * K, 4).astype(np.uint32)
    words = (by[..., 0] << 24) | (by[..., 1] << 16) | (by[..., 2] << 8) | by[..., 3]
    return np.ascontiguousarray(words.astype("<u4")).view("<i2").reshape(tk, tn, 16 * K)


# ---------------------------------------------------------------------------------------------- Hessian
def finalize_H(H_sum: mx.array, count: int, seed: int):
    """Returns dict(su [k], L mx [k,k], H mx (transformed, damped), diag [k])."""
    with T("finalize"):
        k = H_sum.shape[0]
        H = np.array(H_sum, copy=True) / np.float32(count)
        diag_mean = float(np.mean(np.diag(H)))
        assert math.isfinite(diag_mean) and diag_mean > 1e-20, "degenerate Hessian"
        H[np.diag_indices(k)] += np.float32(SIGMA_REG * diag_mean)
        diag = np.diag(H).copy()
        rng = np.random.default_rng(1000003 * seed + k)
        su = (np.sign(rng.standard_normal(k)) + 1e-5)
        su = np.sign(su).astype(np.float32)
        suc = mx.array(su)
        Hm = mx.array(H)
        Hm = had_r(Hm * suc[None, :])
        Hm = had_l(Hm * suc[:, None])
        mx.eval(Hm)
        Hn = np.array(Hm)
        # block LDL, b = 16
        tries = 0
        while True:
            try:
                Lc = np.linalg.cholesky(Hn)
                break
            except np.linalg.LinAlgError:
                tries += 1
                assert tries <= 10, "Cholesky failed after 10 damping retries"
                log(f" !! Cholesky failed, extra damping {tries}/10")
                Hn[np.diag_indices(k)] += np.float32(2.0 * SIGMA_REG * Hn.diagonal().mean())
        b, m = 16, k // 16
        L = Lc.reshape(m, b, m, b)
        DL = np.stack([L[i, :, i, :] for i in range(m)])                      # [m, b, b]
        DLi = np.linalg.inv(DL.astype(np.float64)).astype(np.float32)
        Lv = Lc.reshape(k, m, b)
        L2 = np.einsum("nmb,mbc->nmc", Lv, DLi).reshape(k, k)                # column block i <- col block i @ inv(DL_i)
        Lb = L2.reshape(m, b, m, b)
        for i in range(m):
            Lb[i, :, i, :] = 0.0                                              # identity blocks, then the scalar diagonal is zeroed
        L2 = np.ascontiguousarray(Lb.reshape(k, k))
        return {"su": su, "L": mx.array(L2), "H": mx.array(Hn), "diag": diag}


# ---------------------------------------------------------------------------------------------- regularize + global scale
def tile_batch_mse(tiles_s: mx.array, K, scales_list, n_per):
    """quantise tile sets scaled by each scale; returns list of mse."""
    out = []
    batch = mx.concatenate([tiles_s * s for s in scales_list]) if len(scales_list) > 1 else tiles_s * scales_list[0]
    qw, _ = V.quantize_tiles(batch, K)
    qw = qw.reshape(len(scales_list), n_per, 256)
    for i, s in enumerate(scales_list):
        out.append(mx.mean(mx.square(qw[i] / s - tiles_s)))
    return out


def sample_indices(tile_ms: np.ndarray, tk: int, tn: int, width=3):
    """flat tile indices (row-major over [tk, tn]) of the global-scale sample: wrapped diagonal + highest / lowest RMS tiles."""
    diag_len = max(tk, tn)
    ii = np.repeat(np.arange(diag_len), width)
    ww = np.tile(np.arange(width), diag_len)
    kk, nn_ = ii % tk, (ii + ww) % tn
    tile_ms = tile_ms.reshape(-1)
    num_x = min(max(8, (diag_len * width) // 16), (tile_ms.shape[0] + 1) // 2)
    hi = np.argsort(-tile_ms, kind="stable")[:num_x]
    lo = np.argsort(tile_ms, kind="stable")[:num_x]
    return np.concatenate([kk * tn + nn_, hi, lo])


def scale_search_tiles(tiles: mx.array, K: int):
    """tiles [S, 256] row-major sample (not yet permuted). Coarse grid on every 3rd tile, fine grid + parabola on all."""
    tiles = tiles[:, PERM]
    coarse = [0.1 + 0.2 * i for i in range(10)]
    subs = tiles[::3]
    mse1 = np.array(mx.stack(tile_batch_mse(subs, K, coarse, subs.shape[0])))
    center = coarse[int(mse1.argmin())]
    step = 0.075
    fine = [center + step * (i - 2) for i in range(5)]
    mse2 = [float(x) for x in tile_batch_mse(tiles, K, fine, tiles.shape[0])]
    best = min(range(5), key=lambda i: mse2[i])
    off = 0.0
    if 0 < best < 4:
        y0, y1, y2 = mse2[best - 1], mse2[best], mse2[best + 1]
        d = y0 - 2.0 * y1 + y2
        off = max(-0.5, min(0.5, 0.5 * (y0 - y2) / d)) if d > 0 else 0.0
    return max(fine[best] + off * step, 0.01), mse2[best]


def scale_search(weight_r: mx.array, K: int, width=3):
    k, n = weight_r.shape
    tk, tn = k // 16, n // 16
    w4 = weight_r.reshape(tk, 16, tn, 16).transpose(0, 2, 1, 3)               # [tk, tn, 16, 16]
    tile_ms = np.array(mx.mean(mx.square(w4), axis=(2, 3)))
    flat = sample_indices(tile_ms, tk, tn, width)
    return scale_search_tiles(w4.reshape(tk * tn, 256)[mx.array(flat.astype(np.int32))], K)


def regularize(w: mx.array, su_in: np.ndarray, sv_in: np.ndarray, K: int):
    """w [k, n] float32 -> (regularised weight, su [k] f32, sv [n] f32, g_scale). apply_out_scales = always."""
    k, n = w.shape
    out_ch = mx.sqrt(mx.mean(mx.square(w), axis=0, keepdims=True))            # [1, n]
    mean = float(mx.mean(out_ch))
    if mean > 1e-30:
        out_ch = out_ch / mean
    zero = mx.abs(out_ch) < 1e-30
    out_ch = mx.where(zero, 0.1, out_ch)
    sv = mx.array(sv_in)[None, :] * out_ch + 1e-10
    w = w / sv
    sv = mx.where(zero, 0.0, sv)
    w = had_r(w)
    in_ch = mx.sqrt(mx.mean(mx.square(w), axis=1, keepdims=True))             # [k, 1]
    in_ch = mx.where(mx.abs(in_ch) < 1e-30, 0.1, in_ch)
    su = mx.array(su_in)[:, None] * in_ch / (-CODEBOOK_SCALE) + 1e-10
    w = w / su
    w = had_l(w)
    mx.eval(w, su, sv)
    g, mse = scale_search(w, K)
    w = w * g
    su = su / g
    mx.eval(w, su)
    return w, su, sv, g


# ---------------------------------------------------------------------------------------------- LDLQ
def ldlq(weight: mx.array, L: mx.array, K: int, label=""):
    """weight [k, n] (regularised, f32), L from finalize_H. Returns (weight_q [k, n], enc uint16 np [tk, tn, 256])."""
    k, n = weight.shape
    tk, tn = k // 16, n // 16
    assert k % 16 == 0 and n % 128 == 0 and k % BUF_K == 0
    prod_cache = mx.zeros((k, n), mx.float32)
    wq_parts = [None] * (k // BUF_K)
    enc_parts = [None] * (k // BUF_K)
    for bi_idx, j in enumerate(range(k, 0, -BUF_K)):
        i = j - BUF_K
        bw = weight[i:j]
        b_L = L[i:j]
        bprod = prod_cache[i:j]
        bwq_rows = [None] * (BUF_K // 16)
        benc_rows = [None] * (BUF_K // 16)
        for bj in range(BUF_K, 0, -16):
            bi = bj - 16
            if bj < BUF_K:
                err = bw[bj:] - mx.concatenate(bwq_rows[bj // 16:], axis=0)
                comp = bprod[bi:bj] + mx.matmul(b_L[bj:, i + bi:i + bj].T, err)
            else:
                comp = bprod[bi:bj]
            rows = bw[bi:bj] + comp
            tiles = rows.reshape(16, tn, 16).transpose(1, 0, 2).reshape(tn, 256)[:, PERM]
            qw, qi = V.quantize_tiles(tiles, K)
            qw = qw[:, PERM_I].reshape(tn, 16, 16).transpose(1, 0, 2).reshape(16, n)
            bwq_rows[bi // 16] = qw
            benc_rows[bi // 16] = qi
        bwq = mx.concatenate(bwq_rows, axis=0)
        wq_parts[bi_idx] = bwq
        enc_parts[bi_idx] = mx.stack(benc_rows, axis=0)                      # [BUF/16, tn, 256]
        prod_cache = prod_cache + mx.matmul(b_L.T, bw - bwq)
        mx.eval(prod_cache, bwq, enc_parts[bi_idx])
    # parts were produced from the last rows to the first
    weight_q = mx.concatenate(wq_parts[::-1], axis=0)
    enc = np.concatenate([np.array(p) for p in enc_parts[::-1]], axis=0)
    return weight_q, enc


def block_trace(a: mx.array, H: mx.array):
    return float(mx.sum(a * mx.matmul(H, a)))


def quantize_group(names, weights, hd, K_of, seed):
    """names/weights [n_i, k] HF orientation (np or mx fp32); hd = finalize_H() result. Returns dict name -> tensors + dequantised W [n, k] f16."""
    k = weights[0].shape[1]
    su0 = hd["su"]
    rng = np.random.default_rng(777 + seed)
    regs, metas = [], []
    for nm, w in zip(names, weights):
        n = w.shape[0]
        sv_in = (np.sign(np.sign(rng.standard_normal(n)) + 1e-5)).astype(np.float32)
        with T("regular"):
            wr, su, sv, g = regularize(mx.array(w).T.astype(mx.float32), su0, sv_in, K_of[nm])
        regs.append(wr); metas.append((su, sv, g))
    out = {}
    # LDLQ is column-independent given L, so same-H, same-K linears run as one concatenated matrix (identical result, bigger Viterbi batches)
    by_k = defaultdict(list)
    for i, nm in enumerate(names):
        by_k[K_of[nm]].append(i)
    for K, idxs in by_k.items():
        with T("ldlq"):
            cat = mx.concatenate([regs[i] for i in idxs], axis=1) if len(idxs) > 1 else regs[idxs[0]]
            wq_cat, enc_cat = ldlq(cat, hd["L"], K)
        col = 0
        for i in idxs:
            nm = names[i]
            n = regs[i].shape[1]
            wq = wq_cat[:, col:col + n]
            enc = enc_cat[:, col // 16:(col + n) // 16]
            col += n
            su, sv, g = metas[i]
            with T("metrics"):
                err = regs[i] - wq
                proxy = block_trace(err, hd["H"]) / max(block_trace(regs[i], hd["H"]), 1e-8)
            with T("pack"):
                tr = pack_trellis(enc, K)
            wd = had_r(had_l(wq) * su) * sv                                   # dequantised W [k, n]
            mx.eval(wd)
            out[nm] = {"trellis": tr, "suh": np.array(su.reshape(-1).astype(mx.float16)),
                       "svh": np.array(sv.reshape(-1).astype(mx.float16)), "proxy": proxy, "g": g, "K": K,
                       "deq": wd.T.astype(mx.float16)}
    return out


def quantize_big(w_hf, hd, K, seed, scratch, chunk_cols=16384, sample_width=3):
    """Same maths as regularize + ldlq + metrics for a tensor too large to hold several fp32 copies (lm_head, 248320 x 1024):
    columns are independent given L, so every pass walks column chunks; the regularised weight is parked in an fp32 memmap.
    sample_width: tiles of the global-scale sample per diagonal position (3 = the exllamav3 rule; the head's sample is then ~50 K tiles = 40 % of the head's Viterbi time, one scalar does not need it)."""
    n, k = w_hf.shape
    tk, tn = k // 16, n // 16
    cuts = list(range(0, n, chunk_cols)) + [n]
    chunks = list(zip(cuts[:-1], cuts[1:]))
    rng = np.random.default_rng(777 + seed)
    sv_in = np.sign(np.sign(rng.standard_normal(n)) + 1e-5).astype(np.float32)
    su0 = hd["su"]
    # A0: output channel scales
    _t = time.perf_counter()
    with T("regular"):
        oc = []
        for c0, c1 in chunks:
            w = w_hf[c0:c1].astype(mx.float32).T
            oc.append(mx.sqrt(mx.mean(mx.square(w), axis=0))); mx.eval(oc[-1])
        out_ch = mx.concatenate(oc)
        mean = float(mx.mean(out_ch))
        if mean > 1e-30:
            out_ch = out_ch / mean
        zero = mx.abs(out_ch) < 1e-30
        out_ch = mx.where(zero, 0.1, out_ch)
        sv_full = mx.array(sv_in) * out_ch + 1e-10
        sv_final = mx.where(zero, 0.0, sv_full)
        STAGE["regular.A0"] += time.perf_counter() - _t; _t = time.perf_counter()
        mem_log("head A0")
        # A1: input channel scales (rms over all columns of each row, after sv and the right Hadamard)
        sumsq = mx.zeros((k,), mx.float32)
        for c0, c1 in chunks:
            w = had_r(w_hf[c0:c1].astype(mx.float32).T / sv_full[c0:c1][None, :])
            sumsq = sumsq + mx.sum(mx.square(w), axis=1); mx.eval(sumsq)
        in_ch = mx.sqrt(sumsq / n)
        in_ch = mx.where(mx.abs(in_ch) < 1e-30, 0.1, in_ch)
        su = mx.array(su0) * in_ch / (-CODEBOOK_SCALE) + 1e-10
        STAGE["regular.A1"] += time.perf_counter() - _t; _t = time.perf_counter()
        # B: regularised weight to the memmap, per-tile mean squares for the sample
        mm = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=(k, n))
        tile_ms = np.zeros((tk, tn), np.float32)
        for c0, c1 in chunks:
            w = had_l(had_r(w_hf[c0:c1].astype(mx.float32).T / sv_full[c0:c1][None, :]) / su[:, None])
            ms = mx.mean(mx.square(w.reshape(tk, 16, (c1 - c0) // 16, 16)), axis=(1, 3))
            mx.eval(w, ms)
            mm[:, c0:c1] = np.array(w); tile_ms[:, c0 // 16:c1 // 16] = np.array(ms)
        STAGE["regular.B"] += time.perf_counter() - _t; _t = time.perf_counter()
        mem_log("head B (fp32 memmap written)")
        flat = sample_indices(tile_ms, tk, tn, sample_width)
        kt_, nt_ = flat // tn, flat % tn
        samp = np.stack([mm[a_ * 16:(a_ + 1) * 16, b_ * 16:(b_ + 1) * 16].reshape(256) for a_, b_ in zip(kt_, nt_)])
        STAGE["regular.sample"] += time.perf_counter() - _t; _t = time.perf_counter()
        g, mse = scale_search_tiles(mx.array(samp), K)
        STAGE["regular.search"] += time.perf_counter() - _t
    # C: LDLQ per column chunk
    mem_log("head C start")
    encs, num, den = [], 0.0, 0.0
    for c0, c1 in chunks:
        wr = mx.array(np.ascontiguousarray(mm[:, c0:c1])) * g
        with T("ldlq"):
            wq, enc = ldlq(wr, hd["L"], K)
        with T("metrics"):
            num += block_trace(wr - wq, hd["H"]); den += block_trace(wr, hd["H"])
        encs.append(enc)
        del wr, wq
        mx.clear_cache()
        mem_log(f"head chunk {c0}")
    del mm
    os.remove(scratch)
    with T("pack"):
        tr = pack_trellis(np.concatenate(encs, axis=1), K)
    return {"trellis": tr, "suh": np.array((su / g).reshape(-1).astype(mx.float16)),
            "svh": np.array(sv_final.reshape(-1).astype(mx.float16)), "proxy": num / max(den, 1e-8), "g": g, "K": K}


# ---------------------------------------------------------------------------------------------- model / calibration
class Cap(nn.Module):
    """Wraps an nn.Linear; the owner of a Hessian accumulates x^T x (float32)."""
    def __init__(self, lin, sink=None):
        super().__init__()
        self.lin, self.sink = lin, sink

    def __call__(self, x):
        if self.sink is not None:
            x2 = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            self.sink["H"] = self.sink["H"] + mx.matmul(x2.T, x2)
            mx.eval(self.sink["H"])                                          # keep the graph from retaining every batch's activations
            self.sink["n"] += x2.shape[0]
        return self.lin(x)


def layer_linear_groups(layer):
    """H groups (same input) -> list of (group_key, [(name, module-path)])."""
    if layer.is_linear:
        a = layer.linear_attn
        grp = [("attn_in", [("linear_attn.in_proj_qkv", a, "in_proj_qkv"), ("linear_attn.in_proj_z", a, "in_proj_z")]),
               ("attn_out", [("linear_attn.out_proj", a, "out_proj")])]
    else:
        a = layer.self_attn
        grp = [("attn_in", [("self_attn.q_proj", a, "q_proj"), ("self_attn.k_proj", a, "k_proj"), ("self_attn.v_proj", a, "v_proj")]),
               ("attn_out", [("self_attn.o_proj", a, "o_proj")])]
    m = layer.mlp
    pfx = "mlp."
    if hasattr(m, "switch_mlp"):                                              # MoE block: the shared expert is a dense gated MLP, the routed experts go through quantize_experts
        m, pfx = m.shared_expert, "mlp.shared_expert."
    grp += [("mlp_in", [(pfx + "gate_proj", m, "gate_proj"), (pfx + "up_proj", m, "up_proj")]),
            ("mlp_down", [(pfx + "down_proj", m, "down_proj")])]
    return grp


class MoeTap(nn.Module):
    """Wraps a SparseMoeBlock and keeps its input batches (the post-attention-norm hidden states) for the expert-wise Hessians."""
    def __init__(self, inner):
        super().__init__()
        self.inner, self.xs = inner, []

    def __call__(self, x):
        self.xs.append(x)
        mx.eval(x)
        return self.inner(x)


def moe_layer_weights(W, lp):
    """Source tensors of one MoE block -> mlx_lm names. Handles the fused transformers-5 layout (mlp.experts.gate_up_proj [E, 2I, H], down_proj [E, H, I])
    and the per-expert layout (mlp.experts.{e}.{gate,up,down}_proj.weight). Returns (key suffix, array) pairs for mlp.switch_mlp.*."""
    fused = lp + "mlp.experts.gate_up_proj"
    if fused in W:
        gu = W[fused]
        mid = gu.shape[-2] // 2
        return [("mlp.switch_mlp.gate_proj.weight", gu[:, :mid, :]), ("mlp.switch_mlp.up_proj.weight", gu[:, mid:, :]),
                ("mlp.switch_mlp.down_proj.weight", W[lp + "mlp.experts.down_proj"])]
    E = 1 + max(int(k_[len(lp + "mlp.experts."):].split(".")[0]) for k_ in W if k_.startswith(lp + "mlp.experts."))
    return [(f"mlp.switch_mlp.{n}.weight", mx.stack([W[f"{lp}mlp.experts.{e}.{n}.weight"] for e in range(E)])) for n in ("gate_proj", "up_proj", "down_proj")]


def moe_route(mlp, x):
    """Same routing as Qwen3NextSparseMoeBlock: top-k expert ids per token. x [N, H] -> int array [N, k]."""
    gates = mx.softmax(mlp.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-mlp.top_k, axis=-1)[..., -mlp.top_k:]
    mx.eval(inds)
    return np.array(inds)


def expert_hessians(x, Wg, Wu, rows, need_in=True):
    """H_in = sum x x^T and H_down = sum a a^T (a = silu(x Wg^T) * (x Wu^T), the down_proj input of this expert) over the tokens `rows` (None = all).
    need_in=False: H_in is not computed (None): in 'all' mode it is the same matrix for every expert, built once."""
    xe = x if rows is None else x[mx.array(rows.astype(np.int32))]
    a = (nn.silu(xe @ Wg.T) * (xe @ Wu.T)).astype(mx.float32)
    H2 = mx.matmul(a.T, a)
    H1 = None
    if need_in:
        xe32 = xe.astype(mx.float32)
        H1 = mx.matmul(xe32.T, xe32)
    mx.eval(H2) if H1 is None else mx.eval(H1, H2)
    return H1, H2, xe.shape[0]


def quantize_experts(layer, xs, li, lp, a, out_t, strategy):
    """Expert-wise capture + quantisation of a MoE block. Returns (deq stacks (gate, up, down) as mx [E, ., .], report rows, stats).
    mode 'all'   : every expert sees all calibration tokens (H_in is then one matrix for all experts: one Cholesky, and the gate / up matrices of
                   `a.expert_batch` experts run as ONE LDLQ whose Viterbi calls hold thousands of tiles).
    mode 'routed': an expert sees the tokens routed to it (H_down from its own activations); experts below a.min_expert_tokens tokens fall back to all tokens."""
    mlp = layer.mlp
    E = mlp.num_experts
    x = mx.concatenate(xs, axis=0).reshape(-1, xs[0].shape[-1])
    xs.clear()                                                                   # the block's input batches are not needed twice
    N = x.shape[0]
    Wg, Wu, Wd = mlp.switch_mlp.gate_proj.weight, mlp.switch_mlp.up_proj.weight, mlp.switch_mlp.down_proj.weight
    inds = moe_route(mlp, x)
    counts = np.bincount(inds.reshape(-1), minlength=E)
    Kexp = strategy.get(f"L{li}.exp", a.k_expert)                      # the ladder's expert cell of this block
    Kof = lambda nm: strategy.get(lp + nm, Kexp)
    dq = {"gate_proj": [None] * E, "up_proj": [None] * E, "down_proj": [None] * E}
    rep, fallback = [], 0

    def store(e, nm, r):
        full = f"mlp.experts.{e}.{nm}"
        for t in ("trellis", "suh", "svh"):
            out_t[f"{lp}{full}.{t}"] = r[t]
        dq[nm][e] = r["deq"]
        rep.append({"layer": li, "name": full, "K": r["K"], "proxy": r["proxy"], "g": r["g"], "tokens": int(counts[e])})

    if a.expert_mode == "all":
        with T("capture"):
            x32 = x.astype(mx.float32)
            H1 = mx.matmul(x32.T, x32); mx.eval(H1)
            del x32
        hd_in = finalize_H(H1, N, li * 100003)
        for e0 in range(0, E, a.expert_batch):
            es = list(range(e0, min(E, e0 + a.expert_batch)))
            names = [f"mlp.experts.{e}.{n}" for e in es for n in ("gate_proj", "up_proj")]
            ws = [np.array((Wg if n == "gate_proj" else Wu)[e].astype(mx.float32)) for e in es for n in ("gate_proj", "up_proj")]
            res = quantize_group(names, ws, hd_in, {nm: Kof(nm) for nm in names}, li * 100003 + e0 + 1)
            for e in es:
                for n in ("gate_proj", "up_proj"):
                    store(e, n, res[f"mlp.experts.{e}.{n}"])
            for e in es:
                with T("capture"):
                    _, H2, n2 = expert_hessians(x, Wg[e], Wu[e], None, need_in=False)
                nm = f"mlp.experts.{e}.down_proj"
                r = quantize_group([nm], [np.array(Wd[e].astype(mx.float32))], finalize_H(H2, n2, li * 100003 + 2 * e + 1), {nm: Kof(nm)}, li * 100003 + 50000 + e)
                store(e, "down_proj", r[nm])
    else:
        for e in range(E):
            rows = np.nonzero((inds == e).any(axis=1))[0]
            if len(rows) < a.min_expert_tokens:
                rows, fallback = None, fallback + 1
            with T("capture"):
                H1, H2, n_e = expert_hessians(x, Wg[e], Wu[e], rows)
            nms = [f"mlp.experts.{e}.gate_proj", f"mlp.experts.{e}.up_proj"]
            r = quantize_group(nms, [np.array(Wg[e].astype(mx.float32)), np.array(Wu[e].astype(mx.float32))], finalize_H(H1, n_e, li * 100003 + 2 * e),
                               {nm: Kof(nm) for nm in nms}, li * 100003 + e)
            store(e, "gate_proj", r[nms[0]]); store(e, "up_proj", r[nms[1]])
            nm = f"mlp.experts.{e}.down_proj"
            r = quantize_group([nm], [np.array(Wd[e].astype(mx.float32))], finalize_H(H2, n_e, li * 100003 + 2 * e + 1), {nm: Kof(nm)}, li * 100003 + 50000 + e)
            store(e, "down_proj", r[nm])
    stacks = []
    for n in ("gate_proj", "up_proj", "down_proj"):                              # one projection at a time: the per-expert list and its stack are never both whole for all three
        stacks.append(mx.stack(dq[n])); mx.eval(stacks[-1]); dq[n] = None
    stacks = tuple(stacks)
    return stacks, rep, {"tokens": N, "min_count": int(counts.min()), "max_count": int(counts.max()), "fallback": fallback}


def k_of(strategy, lp, li, nm, a):
    """K of one linear: per-linear key (HF name), else per-cell key "L{block}.attn" / "L{block}.mlp" (what the bpw ladder writes), else --k-attn / --k-mlp."""
    attn = nm.startswith(("self_attn", "linear_attn"))
    return strategy.get(lp + nm, strategy.get(f"L{li}.{'attn' if attn else 'mlp'}", a.k_attn if attn else a.k_mlp))


def run_state(layer, state, mask_fn, bs):
    outs = []
    for s in range(0, state.shape[0], bs):
        h = state[s:s + bs]
        outs.append(layer(h, mask=mask_fn(layer, h), cache=None))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--cal", required=True)
    ap.add_argument("--rows", type=int, default=0); ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--k-attn", type=int, default=3); ap.add_argument("--k-mlp", type=int, default=2)
    ap.add_argument("--head-bits", type=int, default=6)
    ap.add_argument("--k-expert", type=int, default=2, help="MoE routed experts (gate / up / down of every expert); the shared expert uses --k-mlp")
    ap.add_argument("--expert-mode", choices=("all", "routed"), default="all", help="expert Hessians from all calibration tokens (default) or from the tokens routed to each expert")
    ap.add_argument("--min-expert-tokens", type=int, default=128, help="routed mode: experts that received fewer tokens use all tokens")
    ap.add_argument("--expert-batch", type=int, default=32, help="all mode: experts per batched gate / up LDLQ")
    ap.add_argument("--strategy", default=None, help="json {linear name (HF .weight stripped): K}")
    ap.add_argument("--layers", type=int, default=0, help="only the first N decoder blocks (debug)")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--bs", type=int, default=8, help="calibration rows per forward batch")
    ap.add_argument("--fp-prefix", action="store_true", help="calibration state advances through the FP block instead of the quantised one (blocks become independent of each other: A/B for pipelining blocks across devices)")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/ckpt (per-block tensors + calibration state)")
    ap.add_argument("--cache", default=None, help="per-block / head result cache dir (implies --fp-prefix): a block whose (layer, K per linear, calibration) key was converted before is loaded, "
                                                  "not re-quantised; also measures the local error per block (attention-only / MLP-only / both quantised, FP input) = the sensitivity the bpw ladder reads")
    ap.add_argument("--head-chunk", type=int, default=16384, help="head columns per LDLQ pass (quantize_big)")
    ap.add_argument("--head-sample-width", type=int, default=1, help="global-scale sample of the head: tiles per diagonal position (3 = exllamav3 rule, 3x the search time for a 0.3 %% different scale, same proxy)")
    a = ap.parse_args()
    t_start = time.perf_counter()
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    cfg = json.load(open(f"{a.src}/config.json"))
    tc = cfg.get("text_config", cfg)
    targs = TextModelArgs.from_dict(tc)
    index = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"] if os.path.exists(f"{a.src}/model.safetensors.index.json") else None
    files = sorted({f for f in (index.values() if index else [])}) or [f for f in os.listdir(a.src) if f.endswith(".safetensors")]
    def load_src():                                                            # lazy; reloaded after every block: an array evaluated through W stays resident as long as W holds it
        return load_source(a.src, files, log)                                  # plain dict for BF16; lazy per-linear dequant for FP8 / MXFP4 / NVFP4 sources
    W = load_src()
    mx.set_cache_limit(512 << 20)                                              # the allocator cache counts in the process footprint (a MoE block's stacks are GiB)
    pre = "model.language_model."
    ids = np.array(mx.load(a.cal)["input_ids"])
    if a.rows: ids = ids[:a.rows]
    if a.cols: ids = ids[:, :a.cols]
    R, C = ids.shape
    nl = a.layers or targs.num_hidden_layers
    strategy = json.load(open(a.strategy)) if a.strategy else {}
    log(f"src={a.src} rows={R} cols={C} layers={nl} k_attn={a.k_attn} k_mlp={a.k_mlp} head_bits={a.head_bits}")
    if a.cache:
        a.fp_prefix = True
        os.makedirs(a.cache, exist_ok=True)
    cal_key = hashlib.sha1(ids.tobytes()).hexdigest()[:10]
    sens = []                                                                    # per block: local errors + parameter counts (cache mode)

    emb = W[pre + "embed_tokens.weight"]
    state = emb[mx.array(ids.astype(np.int32))].astype(mx.float16)
    mx.eval(state)
    del emb
    W = load_src()                                                               # the 1 GiB embedding was evaluated through the old dict
    out_t = {}                                                                   # name -> np array
    report = []

    def masks(layer, h):
        return create_ssm_mask(h, None) if layer.is_linear else create_attention_mask(h, None)

    ck = f"{a.out}/ckpt"
    start = 0
    if a.resume and not a.cache and os.path.exists(f"{ck}/last.json"):
        last = json.load(open(f"{ck}/last.json"))
        start = last["li"] + 1
        state = mx.array(np.load(f"{ck}/state.npy")); mx.eval(state)
        for b_ in range(start):
            out_t.update(mx.load(f"{ck}/blk{b_}.safetensors"))
            lpb = f"{pre}layers.{b_}."
            for k_, v in src_items(W, lpb):
                if k_.startswith(lpb) and not any(k_[len(lpb):].startswith(q) for q in QUANT_PREFIXES):
                    out_t[k_] = v
        report = json.load(open(f"{ck}/report.json"))
        STAGE.update(last.get("stage", {}))
        log(f"resumed after block {last['li']}")

    for li in range(start, nl):
        t_layer = time.perf_counter()
        mem_log(f"block {li} start")
        layer = DecoderLayer(targs, li)
        lp = f"{pre}layers.{li}."
        wts = []
        if targs.num_experts > 0:
            wts += [(rk, v.astype(mx.float16) if v.dtype == mx.bfloat16 else v) for rk, v in moe_layer_weights(W, lp)]
        for k_, v in src_items(W, lp):
            if not k_.startswith(lp):
                continue
            rk = k_[len(lp):]
            if rk.startswith("mlp.experts."):
                continue
            if "conv1d.weight" in rk and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            if rk.endswith(NORM_SFX) and v.ndim == 1:
                v = v.astype(mx.float32) + 1.0           # HF checkpoints store (w - 1) for the (1 + w) RMSNorms; mlx_lm.sanitize does the same
            wts.append((rk, v.astype(mx.float16) if v.dtype == mx.bfloat16 else v))
        layer.load_weights(wts, strict=False)
        layer.eval()
        mx.eval(layer.parameters())
        mem_log(f"block {li} layer loaded")
        groups = layer_linear_groups(layer)
        qnames = {nm for _, ms in groups for nm, _, _ in ms}
        ckey = None
        if a.cache:
            kk = {nm: k_of(strategy, lp, li, nm, a) for nm in sorted(qnames)}
            if targs.num_experts > 0:
                kk["experts"] = [strategy.get(f"L{li}.exp", a.k_expert), a.expert_mode]
            cf = f"{a.cache}/blk{li}_" + hashlib.sha1(json.dumps([li, cal_key, kk, "v1"]).encode()).hexdigest()[:12]
            if os.path.exists(cf + ".json"):                                     # same block, same bits, same calibration: no capture / quantisation, only the FP forward for the next state
                with T("cached"):
                    state = run_state(layer, state, masks, a.bs)
                out_t.update(mx.load(cf + ".safetensors"))
                meta = json.load(open(cf + ".json"))
                report += meta["rows"]; sens.append(meta["sens"])
                for k_, v in src_items(W, lp):
                    if k_.startswith(lp) and k_[len(lp):].rsplit(".weight", 1)[0] not in qnames and not k_[len(lp):].startswith("mlp.experts."):
                        out_t[k_] = v
                log(f"block {li:2d} cached {time.perf_counter()-t_layer:5.1f}s " + " ".join(f"{r_['name'].split('.')[-1]}:K{r_['K']}/{r_['proxy']:.4f}" for r_ in meta["rows"]))
                mx.clear_cache()
                del layer
                wts = groups = None
                W = load_src()
                continue
        # ---- capture
        sinks = {}
        orig = {}
        for gk, members in groups:
            lin0 = getattr(members[0][1], members[0][2])
            sinks[gk] = {"H": mx.zeros((lin0.weight.shape[1], lin0.weight.shape[1]), mx.float32), "n": 0}
            for mi, (nm, parent, attr) in enumerate(members):
                orig[nm] = getattr(parent, attr)
                setattr(parent, attr, Cap(orig[nm], sinks[gk] if mi == 0 else None))
        tap = None
        if hasattr(layer.mlp, "switch_mlp"):
            tap = MoeTap(layer.mlp)
            layer.mlp = tap
        with T("capture"):
            fp_out = run_state(layer, state, masks, a.bs)
            mx.eval([s["H"] for s in sinks.values()])
        if tap is not None:
            layer.mlp = tap.inner
        mem_log(f"block {li} captured")
        for gk, members in groups:
            for nm, parent, attr in members:
                setattr(parent, attr, orig[nm])
        # ---- quantise
        deq = {}
        lrep = []
        for gk, members in groups:
            names = [nm for nm, _, _ in members]
            hd = finalize_H(sinks[gk]["H"], sinks[gk]["n"], li)
            Kof = {nm: k_of(strategy, lp, li, nm, a) for nm in names}
            ws = [np.array(orig[nm].weight.astype(mx.float32)) for nm in names]
            res = quantize_group(names, ws, hd, Kof, li)
            for nm in names:
                r = res[nm]
                for t in ("trellis", "suh", "svh"):
                    out_t[f"{lp}{nm}.{t}"] = r[t]
                deq[nm] = r["deq"]
                lrep.append((nm, r["K"], r["proxy"], r["g"]))
            del hd
        if tap is not None:
            stacks, erep, estat = quantize_experts(layer, tap.xs, li, lp, a, out_t, strategy)
            report += erep
            mem_log(f"block {li} experts quantised")
            log(f"block {li:2d} experts: tokens {estat['tokens']} per-expert min {estat['min_count']} max {estat['max_count']} fallback {estat['fallback']} mean proxy {np.mean([r_['proxy'] for r_ in erep]):.4f}")
            tap.xs = []
        # ---- advance with the quantised block
        fpw = {nm: parent_attr.weight for gk, members in groups for nm, parent, attr in members for parent_attr in [getattr(parent, attr)]}
        for gk, members in groups:
            for nm, parent, attr in members:
                getattr(parent, attr).weight = deq[nm]
        if tap is not None:
            sm = layer.mlp.switch_mlp
            fp_stacks = (sm.gate_proj.weight, sm.up_proj.weight, sm.down_proj.weight)
            sm.gate_proj.weight, sm.up_proj.weight, sm.down_proj.weight = stacks
        mx.eval(layer.parameters())
        if a.fp_prefix:
            if a.cache:
                with T("advance"):
                    errs = {}
                    cells = [("attn", lambda n_: n_.startswith(("self_attn", "linear_attn")), False), ("mlp", lambda n_: n_.startswith("mlp."), False)]
                    if tap is not None:
                        cells.append(("exp", lambda n_: False, True))                     # routed experts only
                    for tag, sel, exp_on in cells + [("all", lambda n_: True, True)]:
                        for gk, members in groups:
                            for nm, parent, attr in members:
                                getattr(parent, attr).weight = deq[nm] if sel(nm) else fpw[nm]
                        if tap is not None:
                            sm = layer.mlp.switch_mlp
                            sm.gate_proj.weight, sm.up_proj.weight, sm.down_proj.weight = stacks if exp_on else fp_stacks
                        mx.eval(layer.parameters())
                        d = run_state(layer, state, masks, a.bs).astype(mx.float32) - fp_out.astype(mx.float32)
                        errs[tag] = float(mx.sqrt(mx.sum(mx.square(d)) / mx.sum(mx.square(fp_out.astype(mx.float32)))))
            rel, state = (errs["all"] if a.cache else 0.0), fp_out
        else:
            with T("advance"):
                new_state = run_state(layer, state, masks, a.bs)
            d = new_state.astype(mx.float32) - fp_out.astype(mx.float32)
            rel = float(mx.sqrt(mx.sum(mx.square(d)) / mx.sum(mx.square(fp_out.astype(mx.float32)))))
            state = new_state
        mem_log(f"block {li} advanced / sensitivity")
        for k_, v in src_items(W, lp):                                           # non-quantised tensors of this block, as in the source
            if k_.startswith(lp) and k_[len(lp):].rsplit(".weight", 1)[0] not in qnames and not k_[len(lp):].startswith("mlp.experts."):
                out_t[k_] = v
        for nm, K, pe, g in lrep:
            report.append({"layer": li, "name": nm, "K": K, "proxy": pe, "g": g})
        log(f"block {li:2d} done {time.perf_counter()-t_layer:6.1f}s state_err {rel:.4f} peakGB {mem_gb():.2f} " +
            " ".join(f"{nm.split('.')[-1]}:K{K}/{pe:.4f}" for nm, K, pe, g in lrep))
        if a.cache:
            par = lambda sel: int(sum(int(np.prod(fpw[nm].shape)) for nm in fpw if sel(nm)))      # noqa: E731
            moe_cell = {}
            if tap is not None:
                moe_cell = {"err_exp": errs["exp"], "params_exp": int(sum(int(np.prod(s_.shape)) for s_ in fp_stacks)), "K_exp": strategy.get(f"L{li}.exp", a.k_expert)}
            sens.append({"layer": li, "err_attn": errs["attn"], "err_mlp": errs["mlp"], "err_all": errs["all"], **moe_cell,
                         "params_attn": par(lambda n_: n_.startswith(("self_attn", "linear_attn"))), "params_mlp": par(lambda n_: n_.startswith("mlp.")),
                         "K_attn": max(K for nm, K, _, _ in lrep if nm.startswith(("self_attn", "linear_attn"))), "K_mlp": max(K for nm, K, _, _ in lrep if nm.startswith("mlp."))})
            mx.save_safetensors(cf + ".safetensors", {k_: mx.array(v) for k_, v in out_t.items() if k_.startswith(lp) and k_.rsplit(".", 1)[1] in ("trellis", "suh", "svh")})
            json.dump({"rows": [r_ for r_ in report if r_["layer"] == li], "sens": sens[-1]}, open(cf + ".json", "w"))
            mx.clear_cache()
            del layer, deq, sinks, orig, fp_out
            stacks = fp_stacks = tap = wts = groups = res = hd = ws = fpw = deq = None
            W = load_src()
            continue
        os.makedirs(ck, exist_ok=True)
        mx.save_safetensors(f"{ck}/blk{li}.safetensors", {k_: mx.array(v) for k_, v in out_t.items() if k_.startswith(lp) and k_.rsplit(".", 1)[1] in ("trellis", "suh", "svh")})
        np.save(f"{ck}/state.npy", np.array(state))
        json.dump(report, open(f"{ck}/report.json", "w"))
        json.dump({"li": li, "stage": dict(STAGE)}, open(f"{ck}/last.json", "w"))
        mx.clear_cache()
        del layer, deq, sinks, orig, fp_out
        stacks = fp_stacks = tap = wts = groups = res = hd = ws = fpw = deq = None                  # per-block temporaries (the layer's fp16 weights are in `wts` and `groups`)
        W = load_src()
    # ---- head
    if not a.no_head and nl == targs.num_hidden_layers:
        t_head = time.perf_counter()
        nw = (W[pre + "norm.weight"].astype(mx.float32) + 1.0).astype(mx.float16)      # (1 + w) RMSNorm, as mlx_lm.sanitize
        norm = nn.RMSNorm(targs.hidden_size, eps=targs.rms_norm_eps); norm.weight = nw
        Kh = a.head_bits
        hf = f"{a.cache}/head_K{Kh}_{cal_key}_{nl}" if a.cache else None
        if hf and os.path.exists(hf + ".json"):                                  # fp-prefix: the head input does not depend on the block bits
            out_t.update(mx.load(hf + ".safetensors"))
            r = json.load(open(hf + ".json"))
        else:
            hs = []
            for s in range(0, R, a.bs):
                hs.append(norm(state[s:s + a.bs])); mx.eval(hs[-1])
            hstate = mx.concatenate(hs, axis=0).reshape(-1, targs.hidden_size).astype(mx.float32)
            with T("capture"):
                Hh = mx.matmul(hstate.T, hstate); mx.eval(Hh)
            hd = finalize_H(Hh, hstate.shape[0], nl)
            del hs, hstate, Hh
            state = None                                                         # the head quantiser needs room: 2 GiB fp32 scratch + chunks
            mem_log("head input Hessian")
            head_w = W["lm_head.weight"] if "lm_head.weight" in W else W[pre + "embed_tokens.weight"]
            os.makedirs(a.out, exist_ok=True)
            with T("head"):
                r = quantize_big(head_w, hd, Kh, nl, f"{a.out}/head_scratch.npy", a.head_chunk, a.head_sample_width)
            for t in ("trellis", "suh", "svh"):
                out_t[f"lm_head.{t}"] = r[t]
            if hf:
                mx.save_safetensors(hf + ".safetensors", {f"lm_head.{t}": mx.array(r[t]) for t in ("trellis", "suh", "svh")})
                json.dump({"proxy": r["proxy"], "g": r["g"]}, open(hf + ".json", "w"))
        report.append({"layer": "head", "name": "lm_head", "K": Kh, "proxy": r["proxy"], "g": r["g"]})
        log(f"head done {time.perf_counter()-t_head:.1f}s K{Kh} proxy {r['proxy']:.5f} peakGB {mem_gb():.2f}")
    for k_ in (pre + "embed_tokens.weight", pre + "norm.weight"):
        out_t[k_] = W[k_]
    # ---- write
    os.makedirs(a.out, exist_ok=True)
    tens = {k_: (v if isinstance(v, mx.array) else mx.array(v)) for k_, v in out_t.items()}
    for k_ in list(tens):
        if k_.endswith(("in_proj_a.weight", "in_proj_b.weight")):
            tens[k_] = tens[k_].astype(mx.float16)              # the converter keeps these two 16-row linears as fp16
    names_mul1 = sorted({k_[:-len(".trellis")] for k_ in tens if k_.endswith(".trellis")})
    for p in names_mul1:
        tens[p + ".mul1"] = mx.array(np.array([MUL1], np.uint32).view(np.int32)[0], dtype=mx.int32)
    mx.save_safetensors(f"{a.out}/model.safetensors", tens, metadata={"format": "pt"})
    for f in os.listdir(a.src):
        if f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json") or f.endswith((".txt", ".jinja")):
            shutil.copy(f"{a.src}/{f}", f"{a.out}/{f}")
    nb = np_ = 0                                                                 # real mean bits per weight of the quantised blocks (mixed K per cell)
    for k_, v in tens.items():
        if k_.endswith(".trellis") and not k_.startswith("lm_head"):
            nb += int(v.shape[0]) * int(v.shape[1]) * 256 * (int(v.shape[2]) // 16); np_ += int(v.shape[0]) * int(v.shape[1]) * 256
    cfg["quantization_config"] = {"quant_method": "exl3", "version": "mlx-conv-0", "bits": round(nb / np_, 3) if np_ else a.k_mlp, "head_bits": a.head_bits,
                                 "calibration": {"rows": R, "cols": C}, "out_scales": "always", "codebook": "mul1"}
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)
    # quantization_config.json: the per-module tensor map the exllamav3 loaders read (same schema as quant_config.py)
    tn = {mx.float16: "torch.float16", mx.bfloat16: "torch.bfloat16", mx.float32: "torch.float32", mx.int16: "torch.int16", mx.int32: "torch.int32"}
    storage = {}
    for k_, v in sorted(tens.items()):
        mod = k_.rsplit(".", 1)[0]
        if mod.endswith((".linear_attn", ".conv1d")):
            continue                                  # GatedDeltaNet state parameters are not listed by the converter
        e = storage.setdefault(mod, {"stored_tensors": {}})
        e["stored_tensors"][k_] = {"shape": list(v.shape), "n_bytes": int(v.nbytes), "dtype": tn[v.dtype]}
        if k_.endswith(".trellis"):
            e["quant_format"] = "exl3"; e["bits_per_weight"] = v.shape[-1] // 16; e["mul1_multiplier"] = MUL1
    qc = dict(cfg["quantization_config"]); qc["tensor_storage"] = storage
    json.dump(qc, open(f"{a.out}/quantization_config.json", "w"), indent=4)
    json.dump(report, open(f"{a.out}/conv_report.json", "w"), indent=1)
    if a.cache:
        json.dump(sens, open(f"{a.out}/conv_sens.json", "w"), indent=1)
    total = time.perf_counter() - t_start
    log("STAGES " + json.dumps({k_: round(v, 1) for k_, v in STAGE.items()}) + f" total {total:.1f}s peakGB {mem_gb():.2f} stage_peakGB " + json.dumps({k_: round(v, 2) for k_, v in PEAK.items()}))


if __name__ == "__main__":
    main()

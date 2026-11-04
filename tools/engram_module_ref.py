#!/usr/bin/env python3
"""BOX side of the Engram MODULE check (wkv + gate path): CPU-only reference computed from the official FP8 shards with the box engine's
own `engram_inject` (exllamav3.dsv41.engram) and `QuantLin` (fp8 block weight decode, ue8m0 activation fake quant).

usage (inside the engine env, CPU only; tools/../macair3/engram/pyrun.sh does that):
  python tools/engram_module_ref.py CUT.npz ENGRAM_DIR OUT.npz [--per-seq 16] [--seed 7] [--dim 5120] [--hc 4]

CUT.npz = tools/engram_cut.py output for the ref.json sequences (hash ids, raw rows). ENGRAM_DIR = the official Engram shards (wkv, q_weight and
k_weight are read from there; the pack copy is byte-identical, checked 24/10). Picks --per-seq positions per sequence (first, last and seeded
random ones: real token ids from the ref.json prompts and their greedy continuations), builds a seeded residual stream h [T, hc, dim] (fp16
exact values, a third of the rows aligned with the key so that the gate leaves 0.5, a third anti-aligned) and ships:
  ids, seq, pos, hash<p> [T, 24] [T, hc, dim] f32 (fp16 values)
  per layer p: h_in<p> [T, hc, dim] f32 (fp16 values), emb<p> [T, 24, 256] f32 (engine dequant), kv32<p> [T, dim*(hc+1)], gate32<p> [T, hc], out32<p> [T, hc, dim]   fp32 truth
               kv16<p>, gate16<p>, out16<p>                                                         the engine in fp16 (noise floor)
  meta (json)
The Mac side (tools/engram_module_check.py) compares the mlx-vlm Engram module, built the way the Mac loads it, against these."""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DS41_ROOT", os.path.expanduser("~/kyojin-work/ds41golive"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))

import torch
from exllamav3.dsv41 import quant
from exllamav3.dsv41.engram import EngramRowStore, dequant_rows, engram_inject
from exllamav3.dsv41.linear import QuantLin

ap = argparse.ArgumentParser()
ap.add_argument("cut"); ap.add_argument("engram_dir"); ap.add_argument("out")
ap.add_argument("--per-seq", type=int, default=16)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--dim", type=int, default=5120)
ap.add_argument("--hc", type=int, default=4)
ap.add_argument("--eps", type=float, default=1e-6)
a = ap.parse_args()

z = np.load(a.cut, allow_pickle=False)
meta = json.loads(str(z["meta"]))
layer_ids, shards, n_seq = meta["layer_ids"], meta["shards"], meta["n_seq"]
DIM, HC = a.dim, a.hc


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return n, hdr


def tensor(path, name):
    """One tensor of a safetensors shard as a numpy view (memmap slice; nothing else of the 95 GiB file is touched)."""
    n, hdr = read_header(path)
    e = hdr[name]
    lo, hi = e["data_offsets"]
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + lo, shape=(hi - lo,))
    return e, mm


class CutStore(EngramRowStore):
    """Row store backed by the cut npz (raw rows at their table index); asserts every requested row is present."""
    fmt = "fp8"

    def __init__(self, z, n_layers):
        self.rows = [z[f"rows{p}"] for p in range(n_layers)]
        self.raw = [z[f"raw{p}"] for p in range(n_layers)]

    def num_rows(self, layer_pos):
        return int(self.rows[layer_pos].max()) + 1

    def fetch(self, layer_pos, rows):
        r = rows.numpy()
        i = np.searchsorted(self.rows[layer_pos], r)
        assert (self.rows[layer_pos][np.minimum(i, len(self.rows[layer_pos]) - 1)] == r).all(), "row missing from the cut"
        return torch.from_numpy(self.raw[layer_pos][i])


# ---- positions: real ids from the ref.json sequences
rng = np.random.default_rng(a.seed)
sel_seq, sel_pos = [], []
for k in range(n_seq):
    L = len(z[f"seq{k}_ids"])
    pick = {0, 1, L - 1} | set(rng.choice(L, size=a.per_seq - 3, replace=False).tolist())
    for p in sorted(pick):
        sel_seq.append(k); sel_pos.append(p)
sel_seq, sel_pos = np.array(sel_seq), np.array(sel_pos)
T = len(sel_seq)
ids = np.array([z[f"seq{k}_ids"][p] for k, p in zip(sel_seq, sel_pos)], dtype=np.int64)
out = dict(ids=ids, seq=sel_seq, pos=sel_pos)
store = CutStore(z, len(layer_ids))

hrng = np.random.default_rng(a.seed + 1)
h_noise = hrng.standard_normal((T, HC, DIM)).astype(np.float32)
h_scale = np.exp(hrng.uniform(-1.0, 1.5, size=(T, 1, 1))).astype(np.float32)             # token-dependent magnitude, 0.37 .. 4.5
mode = np.arange(T) % 3                                                                   # 0 random, 1 aligned with the key, 2 anti-aligned

for p, lid in enumerate(layer_ids):
    shard = os.path.join(a.engram_dir, shards[p])
    hash_ids = np.stack([z[f"seq{k}_hash"][pos, p, :] for k, pos in zip(sel_seq, sel_pos)])      # [T, 24]
    out[f"hash{p}"] = hash_ids
    # weights from the official shard
    ew, wbytes = tensor(shard, f"layers.{lid}.engram.wkv.weight")
    es, sbytes = tensor(shard, f"layers.{lid}.engram.wkv.scale")
    assert ew["shape"] == [DIM * (HC + 1), 24 * 256] and es["shape"] == [DIM * (HC + 1) // 32, 24 * 256 // 32], (ew["shape"], es["shape"])
    w8 = torch.from_numpy(np.array(wbytes)).view(torch.float8_e4m3fn).view(*ew["shape"])
    sc = torch.from_numpy(np.array(sbytes)).view(torch.float8_e8m0fnu).view(*es["shape"])
    w32 = quant.dequant_fp8_block(w8, sc)                                                         # [25600, 6144] f32
    eq, qb = tensor(shard, f"layers.{lid}.engram.q_weight")
    ek, kb = tensor(shard, f"layers.{lid}.engram.k_weight")
    qk = (torch.from_numpy(np.array(qb)).view(torch.bfloat16).view(HC, DIM).float()
          * torch.from_numpy(np.array(kb)).view(torch.bfloat16).view(HC, DIM).float())              # [hc, dim] f32

    # embeddings and key/value in fp32 (truth)
    raw = store.fetch(p, torch.from_numpy(hash_ids.reshape(-1))).view(T, 24, 264)
    emb = dequant_rows(raw, torch.float32, "fp8")                                                  # [T, 24, 256]
    out[f"emb{p}"] = emb.numpy()
    lin32 = QuantLin(w32, act_quant=True)
    kv32 = lin32(emb.flatten(-2).clone())                                                          # [T, 25600]
    key32 = kv32[:, :HC * DIM].view(T, HC, DIM)
    # residual stream. mode 0: random (saturated gate); modes 1 / 2: h = a * (qk*key direction) + noise with `a` solved per (token, copy) so that the
    # gate dot hits a target spread over +-[0.05, 9] (the interesting part of sigmoid(copysign(sqrt|dot|))); a few exact-zero / tiny targets hit the clamp
    h = torch.from_numpy(h_noise.copy())
    al = qk * key32
    al = al / al.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    num_al, num_n = (al * qk * key32).sum(-1), (h * qk * key32).sum(-1)                           # [T, hc]
    tgt = torch.from_numpy(hrng.uniform(0.05, 9.0, size=(T, HC)).astype(np.float32))
    tgt[:, :1] *= 0.01 if p == 0 else 1.0
    tgt[0, :] = torch.tensor([0.0, 1e-8, 1e-3, -1e-3]); tgt[mode == 2] *= -1
    tgt[mode == 2] = torch.where(tgt[mode == 2] == 0, tgt[mode == 2], tgt[mode == 2])
    kr = torch.rsqrt(key32.square().mean(-1) + a.eps)
    acoef = torch.zeros(T, HC)
    for _ in range(8):
        hh_ = acoef.unsqueeze(-1) * al + h
        r = torch.rsqrt(hh_.square().mean(-1) + a.eps) * kr * DIM ** -0.5
        acoef = (tgt / r.clamp_min(1e-12) - num_n) / num_al
    for t in range(T):
        if mode[t] != 0 or t == 0: h[t] = acoef[t].unsqueeze(-1) * al[t] + h[t]
    h = (h * torch.from_numpy(h_scale)).half().float()
    out[f"h_in{p}"] = h.numpy()
    hh = h.view(1, T, HC, DIM)

    def gate_of(hf, key, dtype_note):
        rstd = torch.rsqrt(hf.square().mean(-1) + a.eps) * torch.rsqrt(key.float().square().mean(-1) + a.eps)
        dot = (hf * qk * key.float()).sum(-1) * rstd * DIM ** -0.5
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)), dot

    gate32, dot32 = gate_of(hh.float()[0], key32, "f32")
    new32 = engram_inject(hh.float(), torch.from_numpy(hash_ids).view(1, T, 24), store, p, lin32, qk, HC, DIM, a.eps)
    chk = hh.float()[0] + gate32.unsqueeze(-1) * kv32[:, HC * DIM:].unsqueeze(1)
    assert torch.allclose(new32[0], chk, atol=1e-5, rtol=1e-5), "engram_inject != manual formula"
    out[f"kv32{p}"], out[f"gate32{p}"], out[f"dot32{p}"], out[f"out32{p}"] = kv32.numpy(), gate32.numpy(), dot32.numpy(), new32[0].numpy()

    # engine in fp16 (what runs in production): weights decoded to half, activations half
    lin16 = QuantLin(w32.half(), act_quant=True)
    emb16 = dequant_rows(raw, torch.float16, "fp8")
    kv16 = lin16(emb16.flatten(-2).clone())
    gate16, _ = gate_of(hh.half().float()[0], kv16[:, :HC * DIM].view(T, HC, DIM), "f16")
    new16 = engram_inject(hh.half(), torch.from_numpy(hash_ids).view(1, T, 24), store, p, lin16, qk, HC, DIM, a.eps)
    out[f"kv16{p}"], out[f"gate16{p}"], out[f"out16{p}"] = kv16.float().numpy(), gate16.numpy(), new16[0].float().numpy()
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
    print(f"layer {lid}: T={T} |emb| max {float(emb.abs().max()):.3g}, kv16 vs kv32 rel {rel(out[f'kv16{p}'], out[f'kv32{p}']):.2e}, "
          f"gate16-gate32 max {float(np.abs(out[f'gate16{p}'] - out[f'gate32{p}']).max()):.2e}, "
          f"out16 vs out32 rel {rel(out[f'out16{p}'], out[f'out32{p}']):.2e}, gate range {float(gate32.min()):.3f}..{float(gate32.max()):.3f}", flush=True)
    del w32, w8, sc, lin32, lin16

out["meta"] = np.array(json.dumps(dict(layer_ids=layer_ids, T=int(T), dim=DIM, hc=HC, eps=a.eps, seed=a.seed, per_seq=a.per_seq, cut=os.path.basename(a.cut))))
np.savez_compressed(a.out, **out)
print("saved", a.out, round(os.path.getsize(a.out) / 2 ** 20, 1), "MiB")

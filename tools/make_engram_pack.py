#!/usr/bin/env python3
"""Turn a shrunk DS4.1 pack (tools/shrink_pack.py, no Engram layer) into a small pack WITH an Engram layer, for loader / hookup tests on the Air.
usage: make_engram_pack.py SRC_PACK OUT_DIR [--layer 1 | --layers 0,1] [--vocab 499] [--seed 1] [--real-shard SHARD]
--real-shard: an official Engram shard (or the sparse copy tools/engram_check.py builds): the config takes the real table size and the real
Engram numbers (vocab 16000000, 99092 compressed ids), the shard is symlinked in place of the generated embeds. Tests that a pack with the
real 95 GiB table loads without evaluating it.

OUT_DIR: symlinks to the shards of SRC_PACK, a config.json with engram_* set (layer `--layer`, bucket primes from `--vocab`, so ~12k table rows
instead of 384M), engram_token_map.json (4096 tokens -> 500 ids), engram.safetensors (q_weight, k_weight, wkv.weight / wkv.scale with the real
dtypes and 32x32 block scales, as in the official pack) and engram_embed.safetensors (embed.weight F8_E4M3 [rows, 256] + embed.scale F8_E8M0
[rows, 8], as in the official shards 47 / 48), and a merged model.safetensors.index.json. Random valid fp8 values, no NaN codes."""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import engram_rows as ER


def write_st(path, tensors):
    """tensors: list of (name, dtype string, shape, bytes)."""
    hdr, off = {}, 0
    for name, dt, shape, b in tensors:
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for _, _, _, b in tensors:
            fh.write(b)


def fp8_bytes(rng, shape, lo=3, hi=9):
    """Random e4m3 bytes with exponent field in [lo, hi] (moderate magnitudes, never the NaN code)."""
    e = rng.integers(lo, hi + 1, shape)
    m = rng.integers(0, 8, shape)
    s = rng.integers(0, 2, shape)
    return ((s << 7) | (e << 3) | m).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--layers", help="comma list of Engram layers (several tables); overrides --layer")
    ap.add_argument("--vocab", type=int, default=499)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real-shard")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    t = cfg.get("text_config", cfg)
    hidden, hc, vocab = t["hidden_size"], t["hc_mult"], t["vocab_size"]
    K = 500
    if a.real_shard:
        a.vocab, K = 16000000, 99092
    t.update(engram_layer_ids=layers, engram_max_ngram_size=4, engram_vocab_size=a.vocab, engram_n_heads=8, engram_head_dim=256,
             engram_pad_token_id=2, engram_compressed_vocab_size=K, engram_num_embeddings=[0])
    spec = ER.EngramSpec.from_config(cfg)
    rows_l = [sum(p for ng in pl for p in ng) for pl in spec.primes()]                      # the primes differ per layer
    rows = rows_l[0]
    if a.real_shard:
        hdr, _ = ER.read_header(a.real_shard)
        assert hdr[f"layers.{layers[0]}.engram.embed.weight"]["shape"][0] == rows, "shard rows differ from the primes of the config"
    t["engram_num_embeddings"] = rows_l
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            link = os.path.join(a.out, f)
            if f.endswith(".safetensors") and not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(a.src, f)), link)
        elif f not in ("config.json",) and os.path.isfile(os.path.join(a.src, f)):
            os.system(f"cp '{os.path.join(a.src, f)}' '{a.out}/'")
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    tm = np.concatenate([np.arange(K), rng.integers(0, K, vocab - K)]) if vocab >= K else np.append(rng.integers(0, K, vocab - 1), K - 1)
    tm[2] = 2
    json.dump([int(x) for x in tm], open(os.path.join(a.out, "engram_token_map.json"), "w"))
    n_cols, hd = spec.n_cols, spec.head_dim
    bf16_ones = (np.ones((hc, hidden), np.float32).view(np.uint32) >> 16).astype("<u2")
    out_f = hidden * (hc + 1)
    tens = []
    for lyr in layers:
        p = f"layers.{lyr}.engram"
        tens += [
            (f"{p}.q_weight", "BF16", (hc, hidden), bf16_ones.tobytes()),
            (f"{p}.k_weight", "BF16", (hc, hidden), bf16_ones.tobytes()),
            (f"{p}.wkv.weight", "F8_E4M3", (out_f, n_cols * hd), fp8_bytes(rng, (out_f, n_cols * hd), 2, 6).tobytes()),
            (f"{p}.wkv.scale", "F8_E8M0", (out_f // 32, n_cols * hd // 32), rng.integers(118, 124, (out_f // 32, n_cols * hd // 32)).astype(np.uint8).tobytes()),
        ]
    write_st(os.path.join(a.out, "engram.safetensors"), tens)
    if a.real_shard:
        link = os.path.join(a.out, os.path.basename(a.real_shard))
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(a.real_shard), link)
    else:
        write_embeds(a, layers, rows_l, hd, rng)
    wmap = ER.merged_index(a.out, spec)
    ER.write_merged_index(a.out, wmap)
    print(f"{a.out}: engram layers {layers}, {rows} table rows, {len(wmap)} tensors indexed")


def write_embeds(a, layers, rows_l, hd, rng):
    tens = []
    for lyr, rows in zip(layers, rows_l):
        p = f"layers.{lyr}.engram"
        tens += [
            (f"{p}.embed.weight", "F8_E4M3", (rows, hd), fp8_bytes(rng, (rows, hd)).tobytes()),
            (f"{p}.embed.scale", "F8_E8M0", (rows, hd // 32), rng.integers(120, 126, (rows, hd // 32)).astype(np.uint8).tobytes()),
        ]
    write_st(os.path.join(a.out, "engram_embed.safetensors"), tens)


if __name__ == "__main__":
    main()

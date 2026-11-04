#!/usr/bin/env python3
"""MAC side of the Engram cross-check. Three implementations of the table-row mapping and two readers of the rows, against the box engine's truth.

  python tools/engram_check.py SLICE.npz CONFIG.json TOKENIZER.json [--work DIR]            # Air: sparse shards rebuilt from the slice
  python tools/engram_check.py SLICE.npz CONFIG.json TOKENIZER.json --model-dir DIR           # day 1: the real tables (DIR holds the shards)

SLICE.npz comes from tools/engram_cut.py on the box. Checks (each prints PASS / FAIL):
  1. token map: mlx-vlm build_compressed_token_map on the HF tokenizer gives n ids == config.engram_compressed_vocab_size
  2. row mapping: kyojin_mlx.engram_rows.RowMap, mlx-vlm NgramHashState (what the model runs) and the box NgramHasher agree exactly
     on every token of every sequence of the slice (also when the sequence is fed in two chunks, as a prefill/decode split)
  3. shards: headers match the config (dtype F8_E4M3 / F8_E8M0, shape [rows, 256] / [rows, 8], sizes), merged index lists the embeds
  4. rows: EngramTables.read_rows (pread) and mlx-vlm OffloadedEngramEmbedding (np.memmap + mx.dequantize mxfp8) return the box rows / values
  5. prefetch: hash ids -> background preads, rows touched == unique ids
Without --model-dir the shards are SPARSE files with the real headers (apparent size 190 GiB, a few MB allocated): the rows of the slice sit at
their real offsets, any other row reads as zero."""
import argparse
import json
import os
import shutil
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from kyojin_mlx import engram_rows as ER

ap = argparse.ArgumentParser()
ap.add_argument("slice"); ap.add_argument("config"); ap.add_argument("tokenizer")
ap.add_argument("--work", default=None, help="directory for the sparse shards (default: <slice dir>/sparse)")
ap.add_argument("--model-dir", default=None, help="real tables: a directory with the Engram shards (skips the sparse rebuild)")
a = ap.parse_args()

fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra else ""), flush=True)
    if not ok:
        fails.append(name)


z = np.load(a.slice, allow_pickle=False)
meta = json.loads(str(z["meta"]))
cfg = json.load(open(a.config))
spec = ER.EngramSpec.from_config(cfg)
check("slice meta matches the config", list(spec.layer_ids) == meta["layer_ids"] and list(spec.num_embeddings) == meta["num_embeddings"]
      and spec.compressed_vocab_size == meta["compressed_vocab"], f"layers {spec.layer_ids} rows {spec.num_embeddings}")

# ---- sparse shards (Air) or real directory (day 1)
if a.model_dir:
    mdir = a.model_dir
else:
    mdir = a.work or os.path.join(os.path.dirname(os.path.abspath(a.slice)), "sparse")
    shutil.rmtree(mdir, ignore_errors=True)
    os.makedirs(mdir)
    for shard in meta["shards"]:
        hb = z["hdr_" + shard].tobytes()
        n = struct.unpack("<Q", hb[:8])[0]
        hdr = json.loads(hb[8:8 + n]); hdr.pop("__metadata__", None)
        end = max(v["data_offsets"][1] for v in hdr.values())
        path = os.path.join(mdir, shard)
        with open(path, "wb") as fh:
            fh.write(hb)
            fh.truncate(8 + n + end)
        for p, lid in enumerate(spec.layer_ids):
            ent = {t: hdr.get(f"layers.{lid}.engram.embed.{t}") for t in ("weight", "scale")}
            if ent["weight"] is None:
                continue
            rows, raw = z[f"rows{p}"], z[f"raw{p}"]
            with open(path, "r+b") as fh:
                for r, row in zip(rows.tolist(), raw):
                    fh.seek(8 + n + ent["weight"]["data_offsets"][0] + r * spec.head_dim)
                    fh.write(row[:spec.head_dim].tobytes())
                    fh.seek(8 + n + ent["scale"]["data_offsets"][0] + r * (spec.head_dim // 32))
                    fh.write(row[spec.head_dim:].tobytes())
    st = os.stat(os.path.join(mdir, meta["shards"][0]))
    print(f"sparse shards in {mdir}: apparent {st.st_size / 2 ** 30:.0f} GiB, allocated {st.st_blocks * 512 / 2 ** 20:.1f} MiB", flush=True)

# ---- 1. token map
from transformers import PreTrainedTokenizerFast
from mlx_vlm.models.deepseek_v41.engram import build_compressed_token_map, EngramLayout as VLayout, NgramHashState, OffloadedEngramEmbedding, QuantizedEngramEmbedding
import mlx.core as mx

tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer)
t0 = time.time()
tmap, nv = build_compressed_token_map(tok)
check("token map: compressed vocab == config", nv == spec.compressed_vocab_size and len(tmap) == meta["token_map_len"], f"{len(tmap)} tokens -> {nv} ids, {time.time() - t0:.1f}s")

# ---- 2. row mapping
text_cfg = cfg.get("text_config", cfg)
vcfg = SimpleNamespace(engram_layer_ids=list(spec.layer_ids), engram_max_ngram_size=spec.max_ngram_size, engram_n_heads=spec.n_heads,
                       engram_vocab_size=spec.vocab_size, engram_num_embeddings=list(spec.num_embeddings), engram_head_dim=spec.head_dim,
                       engram_pad_token_id=spec.pad_token_id, engram_compressed_vocab_size=spec.compressed_vocab_size)
layout = VLayout.from_config(vcfg)
hasher = NgramHashState(vcfg, layout, token_map=tmap)
for k in range(meta["n_seq"]):
    ids, box = z[f"seq{k}_ids"], z[f"seq{k}_hash"]
    rm = ER.RowMap(spec, tmap)
    mine = rm(ids)
    state = SimpleNamespace(engram=None)
    theirs = np.array(hasher(mx.array(ids[None]), 0, state))[0]
    state2 = SimpleNamespace(engram=None)
    cut = max(1, len(ids) // 2)
    two = np.concatenate([np.array(hasher(mx.array(ids[None, :cut]), 0, state2))[0], np.array(hasher(mx.array(ids[None, cut:]), cut, state2))[0]])
    rm2 = ER.RowMap(spec, tmap)
    mine2 = np.concatenate([rm2(ids[:cut], 0), rm2(ids[cut:], cut)])
    check(f"row mapping seq {k} ({len(ids)} tokens): RowMap == box == mlx-vlm", np.array_equal(mine, box) and np.array_equal(theirs, box),
          f"{box.shape}, max row {int(box.max())}")
    check(f"row mapping seq {k}: chunked prefill / decode split == one pass", np.array_equal(two, box) and np.array_equal(mine2, box))

# ---- 3. shards
tables = ER.EngramTables(mdir, spec)
bad = tables.validate()
check("shard headers match the config", not bad, "; ".join(bad[:3]))
wmap = ER.merged_index(mdir, spec)
check("merged index lists the embed tables", all(n in wmap for lid in spec.layer_ids for n in (f"layers.{lid}.engram.embed.weight", f"layers.{lid}.engram.embed.scale")),
      f"{len(wmap)} names")
if not a.model_dir:
    ER.write_merged_index(mdir, wmap)
else:
    idx = os.path.join(mdir, "model.safetensors.index.json")
    check("model dir has an index file with the embed tables", os.path.exists(idx) and all(
        f"layers.{lid}.engram.embed.weight" in json.load(open(idx))["weight_map"] for lid in spec.layer_ids))

# ---- 4. rows: pread reader and mlx-vlm OffloadedEngramEmbedding
for p, lid in enumerate(spec.layer_ids):
    rows, raw, deq = z[f"rows{p}"], z[f"raw{p}"], z[f"deq{p}"]
    pick = np.random.default_rng(p).permutation(len(rows))[:2000]                    # arbitrary order, with the box dequantisation as truth
    w, s = tables.read_rows(p, rows[pick])
    check(f"layer {lid}: pread rows == box rows ({len(pick)})", np.array_equal(w, raw[pick, :spec.head_dim]) and np.array_equal(s, raw[pick, spec.head_dim:]))
    d = ER.dequant_rows(w, s)
    check(f"layer {lid}: numpy dequant == box dequant", np.array_equal(d, deq[pick], equal_nan=True), f"max |x| {np.nanmax(np.abs(d)):.3g}")
    src = QuantizedEngramEmbedding(spec.num_embeddings[p], spec.head_dim, 32, 8, scale_dtype=mx.uint8, mode="mxfp8")
    emb = OffloadedEngramEmbedding(mdir, f"language_model.layers.{lid}.engram.embed", src)
    got = np.array(emb(mx.array(rows[pick])).astype(mx.float32))
    check(f"layer {lid}: mlx-vlm OffloadedEngramEmbedding rows == box dequant", np.array_equal(got, deq[pick], equal_nan=True),
          f"shape {got.shape}, weight {emb._arrays['weight'].shape} {emb._arrays['weight'].dtype}, scales {emb._arrays['scales'].shape} {emb._arrays['scales'].dtype}")

# ---- 5. prefetch
h = z["seq0_hash"]
t0 = time.time()
touched = tables.prefetch(h).result()
want = sum(np.unique(h[:, p, :]).size for p in range(len(spec.layer_ids)))
check("prefetch touches every unique row of a sequence", touched == want, f"{touched} rows, {time.time() - t0:.2f}s")
tables.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)

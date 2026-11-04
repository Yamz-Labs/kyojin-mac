#!/usr/bin/env python3
"""Engram gather benchmark on a REAL-SIZE, fully allocated table (bench/make_engram_table.py): random rows of a 94 GiB layer table, fresh row sets per run
(no sudo purge here, so rows come from the first 80 percent of the table: the page cache holds the tail written last, plus whatever earlier runs touched).

  python bench/bench_engram_gather.py TABLE_DIR CONFIG.json [--decode-calls 2000] [--prefill-rows 100000] [--threads 8,16,32,64] [--json OUT.json]

Patterns (per Engram layer call; a decode token = 2 layer calls, DS4.1 has Engram layers 1 and 14):
  decode   24 distinct random rows (3 n-gram sizes x 8 heads), one call, latency per call (mean / median / p95)
  prefill  N distinct random rows (a 4096-token chunk touches about 100k per layer), one call
Readers: `EngramTables.read_rows` (parallel pread, Kyojin; thread sweep) and mlx-vlm `OffloadedEngramEmbedding` (np.memmap + one fancy-index gather).
Reported: rows/s, useful MB/s (264 B per row: 256 values + 8 scales) and page MB/s (two 16 KB pages per row on Apple Silicon: value page + scale page)."""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mlx.core as mx

from kyojin_mlx import engram_rows as ER
from kyojin_mlx import engram_mlx as EM

ap = argparse.ArgumentParser()
ap.add_argument("table_dir"); ap.add_argument("config")
ap.add_argument("--decode-calls", type=int, default=2000)
ap.add_argument("--prefill-rows", type=int, default=100000)
ap.add_argument("--prefill-reps", type=int, default=3)
ap.add_argument("--memmap-prefill-rows", type=int, default=0, help="rows for the memmap prefill run (default: same as --prefill-rows)")
ap.add_argument("--threads", default="8,16,32,64,128")
ap.add_argument("--rounds", type=int, default=2, help="thread sweep rounds (odd rounds run the list backwards: cancels drift)")
ap.add_argument("--json", default=None)
a = ap.parse_args()

cfg = json.load(open(a.config))
spec = ER.EngramSpec.from_config(cfg)
lid = spec.layer_ids[0]
tabs = ER.EngramTables(a.table_dir, spec, threads=16)                       # the bench dir holds the first Engram layer only
assert "w" in tabs.loc[lid], "first Engram layer table not found in TABLE_DIR"
rows_total = spec.num_embeddings[0]
path, off, shape, _, nb = tabs.loc[lid]["w"]
st = os.stat(path)
print(f"table {path}: rows {rows_total}, file {st.st_size / 2 ** 30:.1f} GiB, allocated {st.st_blocks * 512 / 2 ** 30:.1f} GiB", flush=True)
assert st.st_blocks * 512 > 0.9 * (nb + rows_total * 8), "table is sparse: refusing to benchmark holes"
limit = int(rows_total * 0.8)
rng = np.random.default_rng(int(time.time()))
out = dict(rows_total=rows_total, file_GiB=st.st_size / 2 ** 30)
ROW_B, PAGE_B = 264, 2 * 16384


def draw(n):
    r = rng.choice(limit, size=n, replace=False) if n < limit // 20 else np.unique(rng.integers(0, limit, size=int(n * 1.01)))[:n]
    return np.sort(r.astype(np.int64))


def stats(tag, times, n_rows):
    t = np.array(times)
    d = dict(calls=len(t), rows=n_rows, mean_ms=float(t.mean() * 1e3), median_ms=float(np.median(t) * 1e3), p95_ms=float(np.quantile(t, 0.95) * 1e3),
             rows_per_s=float(n_rows / t.mean()), useful_MBps=float(n_rows * ROW_B / t.mean() / 1e6), page_MBps=float(n_rows * PAGE_B / t.mean() / 1e6))
    out[tag] = d
    print(f"{tag:34s} {d['mean_ms']:9.2f} ms/call (median {d['median_ms']:.2f}, p95 {d['p95_ms']:.2f})  {d['rows_per_s']:11.0f} rows/s  {d['useful_MBps']:8.1f} MB/s useful  {d['page_MBps']:8.1f} MB/s pages", flush=True)


# ---- readers
def parallel_decode(threads):
    t = ER.EngramTables(a.table_dir, spec, threads=threads)
    emb = EM.ParallelRowEmbedding(t, 0)
    ts = []
    for _ in range(a.decode_calls):
        idx = mx.array(rng.integers(0, limit, size=24).astype(np.int64))
        t0 = time.perf_counter(); y = emb(idx); mx.eval(y); ts.append(time.perf_counter() - t0)
    t.close()
    return ts


def parallel_prefill(threads, n):
    t = ER.EngramTables(a.table_dir, spec, threads=threads)
    ts = []
    for _ in range(a.prefill_reps):
        rows = draw(n)
        t0 = time.perf_counter(); w, s = t.read_rows(0, rows); ts.append(time.perf_counter() - t0)
    t.close()
    return ts


from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding, QuantizedEngramEmbedding
src = QuantizedEngramEmbedding(rows_total, spec.head_dim, 32, 8, scale_dtype=mx.uint8, mode="mxfp8")
memmap = OffloadedEngramEmbedding(a.table_dir, f"language_model.layers.{lid}.engram.embed", src)

ths = [int(x) for x in a.threads.split(",")]
print("--- decode: 24 random rows per call", flush=True)
acc = {}
for rd in range(a.rounds):
    for th in (ths if rd % 2 == 0 else ths[::-1]):
        acc.setdefault(th, []).extend(parallel_decode(th))
for th in ths:
    stats(f"decode parallel pread x{th}", acc[th], 24)
ts = []
for _ in range(a.decode_calls):
    idx = mx.array(rng.integers(0, limit, size=24).astype(np.int64))
    t0 = time.perf_counter(); y = memmap(idx); mx.eval(y); ts.append(time.perf_counter() - t0)
stats("decode mlx-vlm memmap gather", ts, 24)

print(f"--- prefill: {a.prefill_rows} distinct random rows per call (x{a.prefill_reps}, {a.rounds} rounds)", flush=True)
acc = {}
for rd in range(a.rounds):
    for th in (ths if rd % 2 == 0 else ths[::-1]):
        acc.setdefault(th, []).extend(parallel_prefill(th, a.prefill_rows))
for th in ths:
    stats(f"prefill parallel pread x{th}", acc[th], a.prefill_rows)
n_mm = a.memmap_prefill_rows or a.prefill_rows
ts = []
for _ in range(max(1, a.prefill_reps - 1)):
    rows = draw(n_mm)
    t0 = time.perf_counter(); y = memmap(mx.array(rows)); mx.eval(y); ts.append(time.perf_counter() - t0)
stats("prefill mlx-vlm memmap gather", ts, n_mm)

best_dec = min((v["mean_ms"], k) for k, v in out.items() if k.startswith("decode parallel"))
best_pre = max((v["rows_per_s"], k) for k, v in out.items() if k.startswith("prefill parallel"))
out["decode_token_ms"] = 2 * best_dec[0]
out["prefill_chunk_4096_s"] = 2 * 100000 / best_pre[0]
print(f"best decode ({best_dec[1]}): 1 token = 2 Engram layer calls = {out['decode_token_ms']:.2f} ms; best prefill ({best_pre[1]}): "
      f"a 4096-token chunk (2 layers x ~100k rows) = {out['prefill_chunk_4096_s']:.2f} s", flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)

#!/usr/bin/env python3
"""MAC side of the Engram MODULE check: the mlx-vlm `Engram` module (wkv + gate path) built and loaded the way the Mac port loads DS4.1
(`patch.build_model` + `patch.load_dense`: pack tensors -> bf16->fp16 cast -> mlx-vlm sanitize with the real FP8 block-scale expansion) is run on
real token ids and compared with the box CPU reference of tools/engram_module_ref.py (fp32 truth from the official FP8 shards, engine fp16 floor).

  python tools/engram_module_check.py REF.npz MODEL_DIR [--json OUT.json] [--bf16]
MODEL_DIR: config.json + model.safetensors.index.json (merged: Engram wkv/q/k from the pack, embed tables from the shards) + the shards. Day 1 it is
the converted model dir (real tables); on the Air: results_mac/engram_module/model (mini pack with layers.{1,14}.engram.* + sparse shards).
Compared per Engram layer, for both gathers (mlx-vlm memmap `OffloadedEngramEmbedding` and Kyojin `ParallelRowEmbedding`):
  rows   fp16 rows fed to wkv == box dequantised rows (exact)
  kv     key|value projection (after the ue8m0 fp8 fake quant of the rows), relative L2 vs the fp32 truth
  gate   sigmoid(copysign(sqrt|dot|)) of every (token, hc copy), max abs diff vs the fp32 truth
  out    updated residual stream, relative L2 vs the fp32 truth, and of the injected delta (out - h)
  call   Engram.__call__ == the step-by-step reproduction used for kv / gate (bit equal)
Thresholds: the box engine in fp16 sets the noise floor (kv16 / gate16 / out16 columns); PASS = Mac error <= 4x that floor (plus a small absolute term)."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import mlx.core as mx

from kyojin_mlx import engram_rows as ER
from kyojin_mlx import engram_mlx as EM
from kyojin_mlx import patch as PM

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("model_dir")
ap.add_argument("--json", default=None)
ap.add_argument("--bf16", action="store_true", help="also run the module with a bf16 residual stream (information only)")
a = ap.parse_args()

z = np.load(a.ref, allow_pickle=False)
meta = json.loads(str(z["meta"]))
fails, res = [], {}


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra else ""), flush=True)
    if not ok:
        fails.append(name)


def rel(x, y):
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


t0 = time.time()
model, spec, cfg = PM.build_model(a.model_dir)
plan = PM.load_dense(model, a.model_dir, spec, cfg)
mx.eval([model.language_model.layers[l].engram.wkv.weight for l in meta["layer_ids"]])
print(f"model skeleton + dense load {time.time() - t0:.1f}s, mlx peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB", flush=True)
from mlx_vlm.models.deepseek_v41.engram import OffloadedEngramEmbedding, _engram_gate_score, _engram_apply_gate
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0

especs = ER.EngramSpec.from_config(cfg)
tables = ER.EngramTables(a.model_dir, especs)
T, HC, DIM, EPS = meta["T"], meta["hc"], meta["dim"], meta["eps"]


def run(eng, h_in, hash_ids, dtype):
    """mlx-vlm Engram.__call__ reproduced step by step (same ops, so kv and gate can be read out); returns kv, gate, out (+ the module's own out)."""
    x = mx.array(h_in[None]).astype(dtype)                                               # [1, T, hc, dim]
    ids = mx.array(hash_ids[None].astype(np.int32))                                      # [1, T, 24]
    n_cols = ids.shape[-1]
    emb = eng.embed(ids).astype(dtype)
    kv = eng.wkv(fake_quant_fp8_ue8m0(emb.reshape(*ids.shape[:-1], n_cols * eng.head_dim)))
    key, value = mx.split(kv, [eng.hc_mult * eng.dim], axis=-1)
    key = key.reshape(*key.shape[:-1], eng.hc_mult, eng.dim)
    score = _engram_gate_score(x, key, eng.q_weight, eng.k_weight, mx.array(eng.dim ** -0.5, dtype=mx.float32), eng.eps, eng.clamp_value)
    gate = mx.sigmoid(score)
    own = eng(x, ids)
    manual = _engram_apply_gate(x, value, gate, None)
    mx.eval(emb, kv, gate, own, manual)
    return emb, kv, gate, own, bool(mx.array_equal(own, manual).item())


names = ["mlx-vlm memmap gather", "Kyojin parallel pread gather"]
for p, lid in enumerate(meta["layer_ids"]):
    eng = model.language_model.layers[lid].engram
    assert isinstance(eng.embed, OffloadedEngramEmbedding), type(eng.embed)
    memmap_embed = eng.embed
    h_in, hash_ids = z[f"h_in{p}"], z[f"hash{p}"]
    kv32, gate32, out32, emb32 = z[f"kv32{p}"], z[f"gate32{p}"], z[f"out32{p}"], z[f"emb{p}"]
    floor_kv, floor_g = rel(z[f"kv16{p}"], kv32), float(np.abs(z[f"gate16{p}"] - gate32).max())
    floor_out = rel(z[f"out16{p}"], out32)
    dbox16 = z[f"out16{p}"] - h_in
    print(f"layer {lid}: wkv {tuple(eng.wkv.weight.shape)} {eng.wkv.weight.dtype}, q_weight {eng.q_weight.dtype}, gate {gate32.min():.3f}..{gate32.max():.3f}; "
          f"box fp16 floor: kv {floor_kv:.2e} gate {floor_g:.2e} out {floor_out:.2e}", flush=True)
    for gi, gname in enumerate(names):
        eng.embed = memmap_embed if gi == 0 else EM.ParallelRowEmbedding(tables, p)
        for dt, dname in ([(mx.float16, "fp16")] + ([(mx.bfloat16, "bf16")] if a.bf16 else [])):
            emb, kv, gate, own, same = run(eng, h_in, hash_ids, dt)
            e = np.array(emb.astype(mx.float32))[0]
            kvn, gn, on = np.array(kv.astype(mx.float32))[0], np.array(gate.astype(mx.float32))[0], np.array(own.astype(mx.float32))[0]
            r = dict(rows_exact=bool(np.array_equal(e, emb32.astype(np.float16).astype(np.float32), equal_nan=True)) if dt == mx.float16 else None, call_equals_steps=same,
                     rows_max_abs=float(np.abs(e - emb32).max()), kv_rel=rel(kvn, kv32), gate_max=float(np.abs(gn - gate32).max()), gate_mean=float(np.abs(gn - gate32).mean()),
                     out_rel=rel(on, out32), delta_rel=rel(on - h_in, out32 - h_in), box16_delta_rel=rel(dbox16, out32 - h_in),
                     kv_dtype=str(kv.dtype), out_dtype=str(own.dtype))
            tag = f"layer {lid} {gname} {dname}"
            res[tag] = r
            print(f"  {tag}: rows max|d| {r['rows_max_abs']:.2e}, kv rel {r['kv_rel']:.2e} ({r['kv_dtype']}), gate max|d| {r['gate_max']:.2e} mean {r['gate_mean']:.2e}, "
                  f"out rel {r['out_rel']:.2e} ({r['out_dtype']}), delta rel {r['delta_rel']:.2e} (box fp16 {r['box16_delta_rel']:.2e})", flush=True)
            if dname == "fp16":
                check(f"{tag}: fp16 rows == box dequantised rows", r["rows_exact"] and r["rows_max_abs"] < 1e-3 * max(1.0, float(np.abs(emb32).max())) + 1e-6, f"max|d| {r['rows_max_abs']:.2e}")
                check(f"{tag}: kv rel <= 4x box fp16 floor", r["kv_rel"] <= 4 * floor_kv + 1e-5, f"{r['kv_rel']:.2e} vs floor {floor_kv:.2e}")
                check(f"{tag}: gate max|d| <= 4x box fp16 floor", r["gate_max"] <= 4 * floor_g + 1e-4, f"{r['gate_max']:.2e} vs floor {floor_g:.2e}")
                check(f"{tag}: out rel <= 4x box fp16 floor", r["out_rel"] <= 4 * floor_out + 1e-5, f"{r['out_rel']:.2e} vs floor {floor_out:.2e}")
                check(f"{tag}: Engram.__call__ == step-by-step reproduction (bit equal)", r["call_equals_steps"])
    eng.embed = memmap_embed
# both gathers must return the same values (bit equal logits are already tested in tests/test_engram_rows.py)
for lid in meta["layer_ids"]:
    r0, r1 = res[f"layer {lid} {names[0]} fp16"], res[f"layer {lid} {names[1]} fp16"]
    check(f"layer {lid}: parallel pread gather == memmap gather (kv, gate, out identical)", r0["kv_rel"] == r1["kv_rel"] and r0["gate_max"] == r1["gate_max"] and r0["out_rel"] == r1["out_rel"])
tables.close()
if a.json:
    json.dump(dict(meta=meta, results=res, fails=fails), open(a.json, "w"), indent=1)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)

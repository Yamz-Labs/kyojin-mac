#!/usr/bin/env python3
"""opbw: per-layer-kind weight bytes of one plain decode step and one prefill chunk, read from the pack headers only.

No model load, no GPU: every figure is a sum of `data_offsets` deltas of the safetensors files, so the tool runs beside the lane.
Kinds: attn (wq_a/wq_b/wkv/wo_a slices/wo_b + their suh/svh), shared (shared expert w1/w3/w2), routed (w1/w3/w2 of the 384
experts), glue (mHC fn/base/scale, gate, norms), head (lm head trellis + suh/svh), embed row, engram rows.

Decode: one token reads the attn/shared/glue/head weights once, and topk of the 384 routed experts. The routed figure is the
mean expert x topk, so it is the expected value over the routing (expert bits differ, 2 or 3).
Prefill: one chunk of `chunk` rows touches every expert of every layer, once per `cap`-row MoE call, i.e. ceil(chunk/cap)
weight sweeps per layer (the docstring of moe_prefill: each extra call re-streams the expert bank).

usage: opbw_bytes.py [--pack DIR] [--chunk 8192] [--cap 4096] [--json F]
"""
import os
import argparse
import glob
import json
import struct
import sys
from collections import defaultdict

KINDS = ("attn", "shared", "routed", "glue", "head", "embed", "engram")


def _hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def load_index(pack):
    """{name: (dtype, shape, nbytes)} over every .safetensors of the pack. Vision/aligner are not served on a text step."""
    out = {}
    for path in sorted(glob.glob(f"{pack}/*.safetensors")):
        h = _hdr(path)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            out[k] = (v["dtype"], tuple(v["shape"]), e - s)
    return out


def kind_of(name):
    """Layer kind of a pack tensor name. `layers.N.ffn.experts.E.<proj>` is routed, `layers.N.ffn.shared_experts` shared.
    Only the trunk (layers.*), the head and the embedding are read by a text step: the vision tower and the aligner never are."""
    if not (name.startswith("layers.") or name.startswith("head.") or name == "embed.weight"):
        return None
    if ".engram." in name:
        return "engram"
    if name.startswith("head."):
        return "head"
    if name == "embed.weight":
        return "embed"
    if ".ffn.experts." in name:
        return "routed"
    if ".ffn.shared_experts." in name:
        return "shared"
    if ".attn." in name:
        return "attn"
    if any(t in name for t in (".hc_", ".ffn.gate.", "attn_norm.", "ffn_norm.", "norm.weight")):
        return "glue"
    return None


def account(idx):
    """{kind: bytes} for the whole trunk (all 40 layers), plus per-layer routed detail."""
    tot = defaultdict(int)
    routed_layers = defaultdict(int)          # layer -> bytes of its 384 experts x 3 proj
    per_layer = defaultdict(lambda: defaultdict(int))
    skipped = 0
    for name, (dt, shape, nb) in idx.items():
        k = kind_of(name)
        if k is None:
            skipped += nb                          # vision tower, aligner, image tokens: not read by a text step
            continue
        tot[k] += nb
        if k == "routed":
            lay = name.split(".")[1]
            routed_layers[lay] += nb
        if name.startswith("layers.") and k != "engram":
            per_layer[name.split(".")[1]][k] += nb
    return tot, routed_layers, per_layer, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.expanduser("~/ds41/model"))
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--cap", type=int, default=4096, help="rows per MoE prefill call (moe_prefill.chunk)")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--ced", action="store_true", help="also report the CED split (layers below mid on all rows, above mid on `tail` rows)")
    ap.add_argument("--tail", type=int, default=128)
    ap.add_argument("--json")
    a = ap.parse_args()
    idx = load_index(a.pack)
    tot, routed_layers, per_layer, skipped = account(idx)
    res = {"pack": a.pack, "n_tensors": len(idx), "trunk": dict(tot), "routed_layers": dict(routed_layers)}
    n = len(routed_layers)
    if n == 0:
        raise SystemExit("opbw_bytes: no routed expert tensors in the pack")

    # ---- plain decode step: one token
    routed_mean_expert = tot["routed"] / n / 384.0
    step = {k: tot[k] for k in ("attn", "shared", "glue", "head")}
    step["routed"] = routed_mean_expert * a.topk * n
    step["embed"] = 2 * 5120 * idx["embed.weight"][2] / idx["embed.weight"][2]      # one row of the table (bf16, 5120 dims)
    res["step_bytes"] = step
    res["step_total"] = sum(step.values())
    res["routed_mean_expert"] = routed_mean_expert

    # ---- prefill chunk: every expert swept once per MoE call
    sweeps = (a.chunk + a.cap - 1) // a.cap
    ch = {k: tot[k] for k in ("attn", "shared", "glue")}
    ch["routed"] = tot["routed"] * sweeps
    ch["head"] = 0                                       # head runs on the last chunk of a request only
    ch["embed"] = 0
    res["chunk"] = {"rows": a.chunk, "cap": a.cap, "sweeps": sweeps, "bytes": ch, "total": sum(ch.values())}
    # expert bytes per token at that chunk (weights are amortised over the chunk rows)
    res["chunk_bytes_per_token"] = sum(ch.values()) / a.chunk
    res["skipped_vision_bytes"] = skipped

    # ---- prefill chunk under CED (kyojin_mlx/ced.py): layers below mid run all rows, layers above mid only the last `tail` rows and
    #      read every expert their tail tokens route to; the decoder half therefore streams the banks for `tail` rows.
    if a.ced:
        mid = n // 2
        enc = [l for l in sorted(routed_layers) if int(l) < mid]
        tail = [l for l in sorted(routed_layers) if int(l) > mid]
        draws = a.tail * a.topk
        act = 1.0 - (1.0 - 1.0 / 384.0) ** draws          # expected share of the 384 experts a tail of `tail` rows touches
        enc_gb = sum(routed_layers[l] for l in enc) * sweeps
        tail_gb = sum(routed_layers[l] for l in tail) * act * ((a.tail + a.cap - 1) // a.cap)
        dense_layers = per_layer
        enc_dense = sum(sum(v.values()) - v.get("routed", 0) for k, v in dense_layers.items() if int(k) < mid)
        tail_dense = sum(sum(v.values()) - v.get("routed", 0) for k, v in dense_layers.items() if int(k) > mid)
        ced = {"mid": mid, "enc_layers": len(enc), "tail_layers": len(tail), "tail": a.tail, "tail_active_experts": act * 384,
               "enc_routed_gb": enc_gb / 1e9, "tail_routed_gb": tail_gb / 1e9,
               "enc_dense_gb": enc_dense / 1e9, "tail_dense_gb": tail_dense / 1e9,
               "total_gb": (enc_gb + tail_gb + enc_dense + tail_dense) / 1e9}
        res["ced"] = ced

    # ---- per-layer table (4-layer sample + mean)
    sample = sorted(per_layer)[:1] + ["20"]
    res["per_layer"] = {l: dict(per_layer[l]) for l in sorted(per_layer)}

    print(f"pack {a.pack}: {len(idx)} tensors, {sum(v[2] for v in idx.values()) / 1e9:.2f} GB on disk "
          f"({skipped / 1e9:.2f} GB vision/aligner/other, not read by a text step)")
    print(f"\n== one plain decode step (1 token, topk {a.topk}) ==")
    for k in ("attn", "shared", "routed", "glue", "head", "embed"):
        print(f"  {k:8s} {step[k] / 1e6:9.1f} MB  {100 * step[k] / res['step_total']:5.1f} %")
    print(f"  {'total':8s} {res['step_total'] / 1e9:9.3f} GB   (gap2 measured 7.5 GB)")
    print(f"  routed: mean expert {routed_mean_expert / 1e6:.2f} MB x {a.topk} x {n} layers")
    print(f"\n== one prefill chunk: {a.chunk} rows, MoE calls of {a.cap} rows -> {sweeps} sweeps/layer ==")
    for k in ("attn", "shared", "routed", "glue"):
        print(f"  {k:8s} {ch[k] / 1e9:9.3f} GB  {100 * ch[k] / res['chunk']['total']:5.1f} %")
    print(f"  {'total':8s} {res['chunk']['total'] / 1e9:9.3f} GB = {res['chunk_bytes_per_token'] / 1e6:.1f} MB/token")
    if a.ced:
        c = res["ced"]
        print(f"\n== the same chunk under CED (mid {c['mid']}, {c['enc_layers']} encoder layers x {a.chunk} rows, "
              f"{c['tail_layers']} decoder layers x {c['tail']} rows) ==")
        print(f"  encoder: routed {c['enc_routed_gb']:.1f} GB ({sweeps} sweeps) + dense {c['enc_dense_gb']:.2f} GB")
        print(f"  tail   : routed {c['tail_routed_gb']:.1f} GB ({c['tail_active_experts']:.0f} of 384 active) + dense {c['tail_dense_gb']:.2f} GB")
        print(f"  total  : {c['total_gb']:.1f} GB per chunk = {c['total_gb'] * 1e9 / a.chunk / 1e6:.1f} MB/token, "
              f"tail layers alone {c['tail_routed_gb'] / a.chunk * 1e9 / 1e6:.1f} MB/token")
    for l in sorted(per_layer):
        d = per_layer[l]
        print(f"  layer {l:>2s}: attn {d.get('attn', 0) / 1e6:7.1f}  shared {d.get('shared', 0) / 1e6:6.1f}  "
              f"routed {d.get('routed', 0) / 1e9:5.2f} GB  glue {d.get('glue', 0) / 1e6:6.1f} MB")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()

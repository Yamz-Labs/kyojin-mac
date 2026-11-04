#!/usr/bin/env python3
"""Cut a few layers out of a DS4.1 / GLM-5.3 EXL3 pack into a small stacked pack (CPU, numpy, streaming, source read-only).
usage: python3 -I shrink_pack.py ds41|glm SRC_PACK OUT_DIR --layers 2,3 [--experts 32] [--vocab 4096] [--mtp 0,1,2 --taps 0,1,1]

  --layers   source layer ids, renumbered 0..n-1 in the given order
  --experts  keep the first N routed experts of every MoE layer (router rows / biases cut to match; the ragged stack is re-based)
  --vocab    keep the first V token rows of the embedding and the first V output columns of the quantized head
             (a 128-block cut of the output side keeps the Hadamard exact: trellis tile columns and svh are cut together)
Everything in the kept layers is copied byte for byte (same tensor names with the layer id renumbered) except the sliced tensors.
Routed experts and wo_a slices go through convert_stack's planners, so the output is the ragged-u32-v1 stacked layout.
Writes: shrunk.safetensors (non-stacked tensors), stacked.layers.<i>.safetensors, model.safetensors.index.json, config.json."""
import argparse, copy, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_stack as CS

ARCH = {
    "ds41": dict(layer=re.compile(r"^layers\.(\d+)\."), lname="layers.{}.",
                 keep=("embed.weight", "norm.weight", "head.suh", "head.svh", "head.mul1", "head.trellis", "head.mcg"),
                 embed="embed.weight", head="head", router=("ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl"),
                 mtp=re.compile(r"^mtp\.(\d+)\."), mtp_vocab=("markov_head.embed.weight", "markov_head.head.weight")),
    "glm": dict(layer=re.compile(r"^model\.language_model\.layers\.(\d+)\."), lname="model.language_model.layers.{}.",
                keep=("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.suh", "lm_head.svh",
                      "lm_head.mul1", "lm_head.trellis", "lm_head.mcg"),
                embed="model.language_model.embed_tokens.weight", head="lm_head",
                router=("mlp.gate.weight", "mlp.gate.e_score_correction_bias")),
}


def sliced_entry(src, name, new_name, rows=None, vocab_cols=None):
    """Entry for a tensor cut on its first axis (`rows`, contiguous prefix) or on the tile-column axis of a trellis."""
    _, _, nb, dt, shp = src.t[name]
    if vocab_cols is not None:                                           # trellis [kt, nt, 16K] -> [kt, nt', 16K]
        kt, nt, w = shp
        arr = np.frombuffer(src.read(name), "<i2").reshape(shp)[:, :vocab_cols // 16].copy()
        return CS.bytes_entry(new_name, dt, list(arr.shape), arr)
    per_row = nb // shp[0]
    new_shape = [rows] + list(shp[1:])
    n = per_row * rows
    return CS.Entry(new_name, dt, new_shape, n, lambda: iter([src.read(name, 0, n)]))


def edit_config(arch, cfg, layers, n_experts, vocab, mtp=(), taps=None):
    cfg = copy.deepcopy(cfg)
    tc = cfg["text_config"]
    new = {old: i for i, old in enumerate(layers)}
    tc["num_hidden_layers"] = len(layers)
    if n_experts:
        tc["n_routed_experts"] = n_experts
    if vocab:
        tc["vocab_size"] = vocab
    if arch == "glm":
        for key in ("layer_types", "mlp_layer_types", "indexer_types"):
            tc[key] = [tc[key][l] for l in layers]
        la = tc["linear_attn_config"]
        la["kda_layers"] = [new[l] for l in la["kda_layers"] if l in new]
        la["full_attn_layers"] = [new[l] for l in la["full_attn_layers"] if l in new]
        tc["first_k_dense_replace"] = sum(1 for t in tc["mlp_layer_types"] if t == "dense")
        tc["num_nextn_predict_layers"] = 0
    else:
        tc["compress_ratios"] = [tc["compress_ratios"][l] for l in layers]
        for key in ("kv_source_layer_ids", "index_source_layer_ids"):
            tc[key] = [new[l] for l in tc[key] if l in new]
        cand = tc["candidate_source_layer_id"]
        tc["candidate_source_layer_id"] = new.get(cand, 0)
        tc["engram_layer_ids"], tc["engram_num_embeddings"] = [], []
        if mtp:                                                          # DSpark stages kept: the drafter taps the slice's own layers
            assert taps and len(taps) == len(tc["dspark_target_layer_ids"]), "--taps needs one slice layer id per dspark target layer"
            tc["num_nextn_predict_layers"] = len(mtp)
            tc["dspark_target_layer_ids"] = list(taps)
            if n_experts:
                tc["dspark_n_routed_experts"] = n_experts
            if vocab:
                tc["dspark_noise_token_id"] = vocab - 96
        else:
            tc["num_nextn_predict_layers"] = 0
            tc["dspark_target_layer_ids"] = []
            tc.pop("dspark_n_routed_experts", None)
            tc.pop("dspark_num_experts_per_tok", None)
    qc = cfg.setdefault("quantization_config", {})
    qc["expert_layout"] = "ragged-u32-v1"
    qc.pop("expert_bits", None)
    return cfg


def shrink(arch, src_dir, out_dir, layers, n_experts=None, vocab=None, mtp=(), taps=None):
    A = ARCH[arch]
    src = CS.Src(src_dir)
    new_id = {old: i for i, old in enumerate(layers)}
    t, plain = {}, {}                                                    # renamed view of the source tensors
    for k, v in src.t.items():
        m = A["layer"].match(k)
        if m:
            if int(m[1]) not in new_id:
                continue
            nk = A["lname"].format(new_id[int(m[1])]) + k[m.end():]
            em = CS.EXP.match(k)
            if n_experts and em and int(em[2]) >= n_experts:
                continue
            t[nk] = v
            plain[nk] = k
        elif mtp and A.get("mtp") and A["mtp"].match(k):
            if int(A["mtp"].match(k)[1]) not in mtp:
                continue
            em = CS.EXP.match(k)
            if n_experts and em and int(em[2]) >= n_experts:
                continue
            t[k] = v
            plain[k] = k
        elif k in A["keep"]:
            t[k] = v
            plain[k] = k
    view = copy.copy(src)
    view.t = t
    # ---- group routed experts and wo_a slices (convert_stack planners)
    egroups, sgroups, consumed = {}, {}, set()
    for k in t:
        m = CS.EXP.match(k)
        if m:
            egroups.setdefault((m[1], m[3]), {}).setdefault(int(m[2]), {})[m[4]] = k; consumed.add(k); continue
        m = CS.SLC.match(k)
        if m:
            sgroups.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = k; consumed.add(k)
    stacked = {}
    for (prefix, proj), parts in sorted(egroups.items()):
        if n_experts:
            assert len(parts) == n_experts, f"{prefix}.{proj}: {len(parts)} experts kept, wanted {n_experts}"
        stacked.setdefault((CS.LAYER.match(prefix) or [None, prefix])[1], []).extend(CS.plan_expert_group(view, prefix, proj, parts))
    for prefix, parts in sorted(sgroups.items()):
        stacked.setdefault((CS.LAYER.match(prefix) or [None, prefix])[1], []).extend(CS.plan_slices(view, prefix, parts))
    # ---- plain tensors (some sliced)
    ents = []
    full_experts = None
    for k, v in sorted(t.items()):
        if k in consumed:
            continue
        shp = v[4]
        if n_experts and any(k.endswith(r) for r in A["router"]) and shp[0] > n_experts:
            full_experts = shp[0]
            ents.append(sliced_entry(view, k, k, rows=n_experts))
        elif vocab and k == A["embed"]:
            ents.append(sliced_entry(view, k, k, rows=vocab))
        elif vocab and A.get("mtp") and A["mtp"].match(k) and any(k.endswith(r) for r in A["mtp_vocab"]):
            ents.append(sliced_entry(view, k, k, rows=vocab))
        elif vocab and k == A["head"] + ".trellis":
            ents.append(sliced_entry(view, k, k, vocab_cols=vocab))
        elif vocab and k == A["head"] + ".svh":
            ents.append(sliced_entry(view, k, k, rows=vocab))
        else:
            ents.append(CS.Entry(k, v[3], v[4], v[2], CS.copy_chunks(view, k)))
    os.makedirs(out_dir, exist_ok=True)
    meta = src.meta or {"format": "pt"}
    wmap, total = {}, 0
    CS.write_st(os.path.join(out_dir, "shrunk.safetensors"), ents, meta)
    for e in ents:
        wmap[e.name] = "shrunk.safetensors"; total += e.nbytes
    for key, es in sorted(stacked.items()):
        fn = f"stacked.{key}.safetensors"
        CS.write_st(os.path.join(out_dir, fn), es, meta)
        for e in es:
            wmap[e.name] = fn; total += e.nbytes
    cfg = edit_config(arch, json.load(open(os.path.join(src_dir, "config.json"))), layers, n_experts, vocab, mtp, taps)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    print(f"{arch}: layers {layers} -> {len(layers)}, experts {n_experts or 'all'}, vocab {vocab or 'all'}: {len(wmap)} tensors, "
          f"{round(total / 2 ** 30, 2)} GiB, {len(stacked)} stacked files")
    src.close()
    return wmap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=list(ARCH)); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--layers", required=True); ap.add_argument("--experts", type=int); ap.add_argument("--vocab", type=int)
    ap.add_argument("--mtp", help="DSpark / MTP stages to keep, e.g. 0,1,2 (ds41; experts and router cut like the layers, markov head cut to --vocab)")
    ap.add_argument("--taps", help="with --mtp: one SLICE layer id per dspark target layer (the drafter reads the input of these layers), e.g. 0,1,1")
    a = ap.parse_args()
    ints = lambda x: [int(v) for v in x.split(",")] if x else []
    shrink(a.arch, a.src, a.out, ints(a.layers), a.experts, a.vocab, tuple(ints(a.mtp)), ints(a.taps))

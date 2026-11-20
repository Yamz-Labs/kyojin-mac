# SPDX-License-Identifier: Apache-2.0
"""Safetensors side of Qwen3.8-Flash-Next (model_type qwen4_exp): the NVIDIA FP4-Local pack, mapped to mlx-vlm's `qwen4_exp` class from the headers alone. Unproven: no Metal run yet.

How the pack reaches the class (nothing here loads a weight):
  * HF names of the pack = the names mlx-vlm's `Model.sanitize` expects (`model.language_model.*`, `lm_head.weight`), so there are NO name rows: foreign_linear._read_safetensors keeps the
    GLM-5.3 path (quantised linears -> Packs, routed experts stacked per projection into `mlp.switch_mlp`, the rest dense). The class lookup is the `qwen4_exp` entry of
    foreign_vlm_classes.FAMILIES (gate: KYOJIN_KIT_UNPROVEN_ROWS), which is the row of this family.
  * routed experts: one U8 nibble tensor per expert and projection, block scales CSF-compressed (csf_scale decodes them to plain E4M3), one float32 `weight_scale_2` per tensor.
  * attention / shared expert / linear attention: F8_E4M3 weight + U8 (UE8M0) scale per 32 (fp8_block of vendor_input).
  * the per-layer n-gram table: 128 shards `...ple.ple_embedding.ngram_embedding.shard_N.weight` U8 [2500012, 80] + `.weight_scale` F8_E4M3 [2500012, 10] + ONE float32
    `ngram_embedding.weight_scale_2` per table. With KYOJIN_MAPPED_ROW_NVFP4 on it is not loaded: foreign_mapped_rows.MappedNvfp4Rows reads the rows of the step from the mapped shards.
`map_header` is the dry run: every header key -> a class parameter or a named auxiliary, and every class parameter -> its source, both ways, shapes compared."""
from __future__ import annotations

import os
import re

PLE_SHARD = re.compile(r"^(.*\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.(weight|weight_scale)$")
PLE_G2 = "weight_scale_2"
LIN_SFX = frozenset(("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale", "input_global_scale", "scale"))   # foreign_linear._LIN_SFX
EXPERT = re.compile(r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
FLAG = "KYOJIN_MAPPED_ROW_NVFP4"
_TAGS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "I64": "int64", "U8": "uint8", "F8_E4M3": "uint8", "U32": "uint32", "I32": "int32", "F64": "float32"}


def mapped_row_nvfp4() -> bool:
    """Flag KYOJIN_MAPPED_ROW_NVFP4 (default 0, read at load): the NVFP4 n-gram table of a qwen4_exp safetensors pack stays in the mapped shards (foreign_mapped_rows.MappedNvfp4Rows)."""
    return os.environ.get(FLAG, "0") not in ("0", "")


def ple_tables(info) -> dict:
    """{table prefix: dict(shards=[(weight key, scale key)], rows=[rows per shard], k, g2=key)} of the n-gram tables in a header (`info`: name -> (tag, shape)). Raises on a half-described table."""
    parts: dict = {}
    for name in info:
        m = PLE_SHARD.match(name)
        if m:
            parts.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = name
    out = {}
    for pre, sh in parts.items():
        n = len(sh)
        if sorted(sh) != list(range(n)) or any(set(v) != {"weight", "weight_scale"} for v in sh.values()):
            raise ValueError(f"{pre}: n-gram table shards are not 0..{n - 1} with a weight and a weight_scale each")
        g2 = f"{pre}.{PLE_G2}"
        if g2 not in info:
            raise ValueError(f"{pre}: n-gram table without its global scale {g2!r}")
        rows, k = [], None
        for i in range(n):
            (wt, ws), (st, ss) = info[sh[i]["weight"]], info[sh[i]["weight_scale"]]
            if len(ws) != 2 or len(ss) != 2 or ws[0] != ss[0] or ws[1] * 2 != ss[1] * 16 or wt != "U8":
                raise ValueError(f"{sh[i]['weight']}: shard {wt} {ws} / scale {ss} is not nibbles [rows, k/2] + one scale per 16 elements [rows, k/16]")
            if k not in (None, ws[1] * 2):
                raise ValueError(f"{pre}: shards differ in width")
            k = ws[1] * 2
            rows.append(int(ws[0]))
        out[pre] = dict(shards=[(sh[i]["weight"], sh[i]["weight_scale"]) for i in range(n)], rows=rows, k=k, g2=g2)
    return out


def table_keys(tables: dict) -> set:
    return {x for t in tables.values() for pair in t["shards"] for x in pair} | {t["g2"] for t in tables.values()}


def build_tables(ck, tables: dict) -> dict:
    """{table prefix: MappedNvfp4Rows} over the checkpoint's mapped shards (views, nothing is read yet). ck: vendor_input.Checkpoint / CsfCheckpoint."""
    import numpy as np
    from .foreign_mapped_rows import MappedNvfp4Rows
    out = {}
    for pre, t in tables.items():
        sh = [(ck.raw(w)[2], ck.raw(s)[2]) for w, s in t["shards"]]
        g = np.asarray(ck.f32(t["g2"]), np.float32).reshape(-1)
        if g.size != 1:
            raise ValueError(f"{t['g2']}: n-gram table global scale has {g.size} values, one expected")
        out[pre] = MappedNvfp4Rows([(a, b.view(np.uint8)) for a, b in sh], float(g[0]), t["k"])
    return out


def class_of(config: dict):
    """(model, {param name: shape}) of the mlx-vlm class for `config`, built lazily on the host (no array is evaluated)."""
    from mlx.utils import tree_flatten
    from . import foreign_vlm_classes as VC
    M, A, _mt, src = VC.resolve([config.get("model_type")], config=config)
    model = M(A.from_dict(config))
    return model, {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}


def map_header(info, config: dict, model=None, class_shapes: dict | None = None, csf=()) -> dict:
    """Dry run of load_foreign's name mapping on a header. info: name -> (tag, shape) (CsfCheckpoint.info(): CSF scales already shown as decoded `weight_scale`), csf: names of the
    raw CSF streams. model / class_shapes default to class_of(config). Returns counts and the lists of what did not match, both ways."""
    import mlx.core as mx
    from . import vendor_input as VI
    if model is None or class_shapes is None:
        model, class_shapes = class_of(config)
    csf = set(csf)
    tables = ple_tables(info)
    own = set(csf) | table_keys(tables)
    cat: dict = {}                                                  # header key -> (category, target)
    for k in csf:
        cat[k] = ("aux:csf_stream", None)
    dropped = {k for k in info if k.startswith("mtp.")}
    vision = {k for k in info if k.startswith("model.visual.")}
    for k in dropped:
        cat[k] = ("dropped:mtp", None)                              # sanitize drops `mtp.*`
    for k in vision:
        cat[k] = ("dropped:vision", None)                           # load_foreign never reads the vision tower
    packs, experts = {}, {}
    for p in VI.linear_prefixes(info):
        if p.startswith(("mtp.", "model.visual.")) or PLE_SHARD.match(p + ".weight"):
            continue
        c = VI.classify_linear(info, p)
        if c is None or c["fmt"].startswith("dense_"):
            continue
        me = EXPERT.match(p) if c["fmt"] == "nvfp4" else None
        (experts.setdefault((me[1], me[3]), []) if me else packs.setdefault(p, [])).append(c)
    owned = {}
    for k in info:
        if k in own or k in dropped or k in vision:
            continue
        pre, _, sfx = k.rpartition(".")
        for p in (pre, pre.rpartition(".")[0]):                     # `x.weight_scale` and `x.weight_scale.nvfp4_csf_fixed`
            if p in packs or (EXPERT.match(p) and (EXPERT.match(p)[1], EXPERT.match(p)[3]) in experts):
                if k.rpartition(".")[2] in LIN_SFX or k[len(p) + 1:].split(".")[0] in LIN_SFX:
                    owned[k] = p
                    break
    dense = {}
    for k, (tag, shape) in info.items():
        if k in cat or k in owned:
            continue
        dense[k] = mx.zeros(tuple(shape), dtype=getattr(mx, _TAGS.get(tag, "float32")))
    marks = {p + ".weight": mx.array([float(i + 1)]) for i, p in enumerate(packs)}
    weights = model.sanitize({**dense, **marks})
    by_id = {id(v): k for k, v in weights.items()}
    where = {p: by_id[id(marks[p + ".weight"])][: -len(".weight")] for p in packs if id(marks[p + ".weight"]) in by_id}
    sdense = {id(v): k for k, v in weights.items() if id(v) in {id(x) for x in dense.values()}}
    src: dict = {}                                                  # class parameter -> (source header key / group, shape from the header)
    bad, unmatched_header = [], []
    for k, v in dense.items():
        tgt = sdense.get(id(v))
        if tgt is None:                                             # sanitize built a new array (conv1d moveaxis): follow it alone
            one = model.sanitize({k: v})
            tgt = next(iter(one)) if len(one) == 1 else None
            if tgt is not None:
                weights[tgt] = one[tgt]
        if tgt is None:
            unmatched_header.append((k, "sanitize dropped or merged it"))
            continue
        cat[k] = ("dense", tgt)
        src[tgt] = (k, tuple(weights[tgt].shape))
    for p, cs in packs.items():
        c = cs[0]
        tgt = where.get(p)
        _t, shp, *_ = _shape(info, c["weight"])
        n, k_ = shp[0], (shp[1] * 2 if c["fmt"] == "nvfp4" else shp[1])
        if tgt is None:
            unmatched_header.append((p + ".weight", "linear not followed through sanitize"))
            continue
        cat[c["weight"]] = (f"pack:{c['fmt']}", tgt + ".weight")
        src[tgt + ".weight"] = (c["weight"], (n, k_))
    for (mp, proj), cs in experts.items():
        probe = f"{mp}.switch_mlp.{proj}.weight"
        tgt = [t for t in (model.sanitize({probe: mx.zeros((1,), dtype=mx.uint8)}) or {})]
        tgt = tgt[0] if tgt else None
        ws = _shape(info, cs[0]["weight"])[1]
        full = (len(cs), ws[0], ws[1] * 2)
        for c in cs:
            cat[c["weight"]] = ("expert:nvfp4", tgt)
        if tgt:
            src[tgt] = (f"{mp}.experts.*.{proj}.weight", full)
        else:
            unmatched_header.append((f"{mp}.experts.*.{proj}", "no switch_mlp target"))
    for k, p in owned.items():                                      # scales, global scales, input scales, CSF streams of a served linear
        cat.setdefault(k, ("aux:" + ("global_scale" if k.endswith("_2") else "csf" if "csf" in k else "scale"), None))
    for pre, t in tables.items():
        probe = model.sanitize({f"{pre}.shard_0.weight": mx.zeros((1, 1), dtype=mx.uint8)})
        base = next(iter(probe)).rsplit(".shards.", 1)[0] if probe else None
        for i, (w, s) in enumerate(t["shards"]):
            cat[w] = ("ple:table", f"{base}.shards.{i}.weight")
            cat[s] = ("aux:ple_scale", None)
            src[f"{base}.shards.{i}.weight"] = (w, (t["rows"][i], t["k"]))
        cat[t["g2"]] = ("aux:ple_global_scale", None)
    for k in info:
        if k not in cat or (cat[k][1] is not None and cat[k][1] not in class_shapes):     # no rule for it, or a rule whose target the class does not have
            cat[k] = ("UNMATCHED", cat.get(k, (0, None))[1])
    class_side = {k: v for k, v in class_shapes.items() if not k.startswith("vision_tower.")}
    missing = sorted(k for k in class_side if k not in src)
    mism = sorted((k, class_side[k], src[k][1]) for k in class_side if k in src and tuple(class_side[k]) != tuple(src[k][1]))
    cnt: dict = {}
    for k, (c, _t) in cat.items():
        cnt[c] = cnt.get(c, 0) + 1
    return dict(header_keys=len(info), by_category=dict(sorted(cnt.items())), unmatched_header=sorted(k for k, (c, _t) in cat.items() if c == "UNMATCHED") + [k for k, _w in unmatched_header],
                class_params=len(class_side), class_params_matched=len(class_side) - len(missing), class_unmatched=missing, shape_mismatch=mism,
                vision_class_params=len(class_shapes) - len(class_side), ple_tables={p: dict(shards=len(t["shards"]), rows=sum(t["rows"]), k=t["k"]) for p, t in tables.items()})


def _shape(info, name):
    tag, shape = info[name][0], info[name][1]
    return tag, tuple(shape)


# the row of this family (safetensors side). There is no name table: the class lookup is foreign_vlm_classes.FAMILIES["qwen4_exp"], gated by the same unproven switch; this record is what
# tools/day0_pack reads to name what is not proven yet.
ROW = dict(model_type="qwen4_exp", architectures=("Qwen4ExpForConditionalGeneration",), gate="qwen4exp", proven=False,
           unproven=("no Metal identity run of the qwen4_exp class on the FP4-Local pack (mapping proven on headers only)",
                     "routed experts: gate / up (k=2560) and down (k=640) both fail the stored nvfp4 gather rule k % 1024 == 0 (foreign_linear.g2a_ok); the route they take instead is unmeasured",
                     "NVFP4 n-gram table read by row id (KYOJIN_MAPPED_ROW_NVFP4): CPU-exact against the full dequant, host decode speed and the global-scale direction unmeasured on the real file",
                     "serve routing: no modeldesc row, so serve.backends.load_backend does not call foreign_linear.load_foreign for this model_type"))

#!/usr/bin/env python3
"""Neutral export of the architecture import kit for a loader that is not MLX (CUDA / HIP engine, anything that reads JSON): the tables generated from llama.cpp
(foreign_arch_gen), the vetted rows, the config rows and the value transforms as named operations, in ONE json file. No Python objects: every value is a string,
number, bool, null, list or object. Also holds the numpy reference interpreter of the transform rows (tests/test_arch_kit_export.py).

  python tools/archkit_export.py OUT.json            (no mlx needed; numpy only for the reference interpreter)

Schema "kyojin-archkit/1":
  schema, llama_cpp_commit, generated_by,
  expression_language   : the grammar of every `expr` string below (same as foreign_cfgrows)
  operations            : the named operations of the transform rows (same text as foreign_xform_rows)
  architectures{arch}   :
    gguf_tensors[]      : {key, gguf (name template, {bid} = block index), status (required|optional|conditional|skip|listed), hf [Hugging Face name templates the converter maps onto it]}
    converter[]         : {cls, hf_arch [HF architectures], flags [value-changing converter transforms: permute plus1 split ...]}
    metadata[]          : {key (GGUF key), source [kind, ...] (hparam [names...] | block_count | rope [name] | const [value] | expr [python text of the converter]), conditional, writer}
    loader{}            : {tensors{KEY: status}, hparams{KEY: status}} as the C++ loader requires them
    vetted{}            : hand-vetted rows, only where one exists: flags, xform (set name), keys (file keys the model class cannot use), names {KEY: hf name}, exact {gguf template: hf name}, cfg {field: expr}
    config_rows{}       : only for architectures whose config is not placed by key name: {model_type, architectures, trunk expr, let[[name, expr]], guards[[refuse_when expr, message]], fields[[field, expr]], extends}
    transform_set       : name of the entry of `transforms` that undoes the converter's value changes of this architecture, or null
  transforms{set}       : {extends, let[[name, expr]], rules[{match regex, ops[{op, when, ...}]}]}; `aliases` maps a vetted xform name to its set
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import foreign_arch_gen as G, foreign_archkit as K, foreign_cfgrows as CR, foreign_xform_rows as XR  # noqa: E402

SCHEMA = "kyojin-archkit/1"


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if callable(x):
        raise TypeError(f"a callable reached the export: {x!r}")
    return x


def export() -> dict:
    archs = {}
    for arch in sorted(G.ARCH_TENSORS):
        st = K.tensor_status(arch)
        v = {k: x for k, x in K.VETTED.get(arch, {}).items()}
        d = dict(
            gguf_tensors=[dict(key=k, gguf=G.GGUF_NAME.get(k), status=st[k], hf=list(K.candidates(arch, k))) for k in G.ARCH_TENSORS[arch]],
            converter=[dict(cls=c["cls"], hf_arch=list(c["hf_arch"]), flags=list(c["flags"])) for c in G.CONVERTER.get(arch, ())],
            metadata=[dict(key=r["key"], source=list(r["src"]), conditional=bool(r["cond"]), writer=r["method"]) for r in K.metadata_rows(arch)],
            loader=dict(tensors=dict(G.LOADER.get(arch, {}).get("tensors", {})), hparams=dict(G.LOADER.get(arch, {}).get("hparams", {}))),
            vetted=v or None,
            config_rows=CR_row(arch) if CR.has_rows(arch) else None,
            transform_set=XR.ALIASES.get(v.get("xform", ""), v.get("xform")) or None)
        archs[arch] = d
    transforms = {n: dict(extends=s.get("extends"), let=[list(x) for x in s.get("let", ())], rules=s["rules"]) for n, s in XR.SETS.items()}
    return _plain(dict(schema=SCHEMA, llama_cpp_commit=G.LLAMA_CPP_COMMIT, generated_by="tools/archkit_export.py", expression_language=CR.__doc__.split("Expressions are", 1)[1].split("`extends`")[0].strip(),
                       operations=XR.__doc__.split("Operations,", 1)[1].strip(), block_count_keys=list(G.BLOCK_COUNT_KEYS), architectures=archs,
                       transforms=transforms, aliases=XR.ALIASES))


def CR_row(arch):
    r = dict(CR.ROWS[arch])
    return {**r, "let": [list(x) for x in r.get("let", ())], "guards": [list(x) for x in r.get("guards", ())], "fields": [list(x) for x in r.get("fields", ())]}


# ------------------------------------------------------------------------------------------------------------------------------ numpy reference interpreter
def _index(spec: dict, env, rows: int):
    k, ev = spec["kind"], (lambda e: CR.ev(e, env))
    if k in ("v_grouped", "v_tiled"):
        nk, nv, hd = int(ev(spec["nk"])), int(ev(spec["nv"])), int(ev(spec["hd"]))
        tiled = np.arange(nv * hd).reshape(nk, nv // nk, hd).transpose(1, 0, 2).reshape(-1)
        if k == "v_tiled":
            return tiled
        skip = int(ev(spec.get("skip", "0")))
        return np.concatenate([np.arange(skip), skip + np.argsort(tiled, kind="stable")])
    if k == "rope_unpermute":
        h = int(ev(spec["heads"]))
        p = np.arange(rows).reshape(h, 2, rows // h // 2).transpose(0, 2, 1).reshape(-1)
        return np.argsort(p, kind="stable")
    raise ValueError(f"unknown index kind {k}")


_POST = {"minus1": lambda w: w.astype(np.float32) - 1.0,
         "a_log": lambda w: np.log(-w.astype(np.float32)) if np.all(w < 0) else (_ for _ in ()).throw(ValueError("ssm_a holds non-negative values")),
         "conv_unsqueeze": lambda w: w.reshape(w.shape[0], 1, w.shape[1]).astype(np.float16),
         "conv_unsqueeze_f32": lambda w: w.reshape(w.shape[0], 1, w.shape[1]),
         "row": lambda w: w.reshape(1, -1) if w.ndim == 1 else w,
         "flat": lambda w: w.reshape(-1),
         "a_log_flat": lambda w: (np.log(-w.astype(np.float32)) if np.all(w < 0) else (_ for _ in ()).throw(ValueError("ssm_a holds non-negative values"))).reshape(-1)}


def apply(set_name: str, arch: str, name: str, kv: dict, w: np.ndarray) -> np.ndarray:
    """The transform rows of `set_name` applied to the decoded tensor `name` of a file of `arch` (kv = its metadata): the reference of any other loader."""
    lets, rules = XR.rules(set_name)
    env = CR._Env(arch, kv, {})
    for n, e in lets:
        try:
            env.names[n] = CR.ev(e, env)
        except (TypeError, KeyError, ValueError, GF_ERR):
            pass
    for r in rules:
        if not re.fullmatch(r["match"], name):
            continue
        for op in r["ops"]:
            if op.get("when") and not CR.ev(op["when"], env):
                continue
            if op["op"] == "cols_inverse_gather":
                w = w[:, np.argsort(_index(op["index"], env, w.shape[0]), kind="stable")]
            elif op["op"] == "gather_rows":
                w = w[_index(op["index"], env, w.shape[0])]
            elif op["op"] == "post":
                w = _POST[op["fn"]](w)
            else:
                raise ValueError(op)
        return w
    return w


from kyojin_mlx.foreign_gguf import GgufError as GF_ERR  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "archkit_tables.json"
    d = export()
    with open(out, "w") as f:
        json.dump(d, f, indent=1, sort_keys=False)
    print(out, os.path.getsize(out), "bytes;", len(d["architectures"]), "architectures;", sum(1 for a in d["architectures"].values() if a["vetted"]), "vetted;",
          sum(1 for a in d["architectures"].values() if a["config_rows"]), "config rows;", len(d["transforms"]), "transform sets")

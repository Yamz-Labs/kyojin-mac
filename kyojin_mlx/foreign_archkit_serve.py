# SPDX-License-Identifier: Apache-2.0
"""Serve path of an architecture that has no hand-written foreign_names row: the rows are built from the generated llama.cpp tables.
`register(path)` resolves the tensor names against the mlx-lm class, wraps them in a foreign_names.Arch and a config builder, and puts both in
foreign_names.ARCHS / _CONFIG; load_foreign then runs its usual steps (ForeignLinear rows for the stored K-quant / Q8_0 ... types, dense tensors for the rest)."""
from __future__ import annotations

import dataclasses
import os

from . import foreign_arch_gen as G, foreign_archkit as K, foreign_archkit_cpu as KC, foreign_cfgrows as CR, foreign_gguf as GF, foreign_names as N

_PRE = "model.layers.{bid}."


def register(path, allow_keys=(), allow_flags=()) -> str:
    """Add the GGUF's architecture to foreign_names.ARCHS from the generated tables; returns the architecture name. KitError when the kit cannot map it."""
    g = GF.GgufFile(GF.find_gguf(path))
    try:
        arch = g.kv["general.architecture"]
        if arch in N.ARCHS:
            return arch
        if not K.have(arch):
            raise K.KitError(f"GGUF architecture {arch!r} is not in the generated tables")
        from . import foreign_skeleton as SK
        try:
            if SK.enabled() and SK.has_rows(arch):                      # opt-in (KYOJIN_SKELETON, default off): the skeleton even when an mlx-lm class exists (checks)
                raise K.KitError("KYOJIN_SKELETON")
            if CR.has_rows(arch):                                       # a config row (foreign_cfgrows: gemma4 behind KYOJIN_GEMMA4_ROWS) replaces kit_config, as in load_cpu
                from mlx_lm.utils import _get_classes
                cfg = CR.build(arch, g)
                from . import foreign_vlm_classes as VC
                Model, Args = VC.classes_for_config(cfg)
                mt = cfg["model_type"]
            else:
                cfg = None
                Model, Args, mt = KC._classes(arch)
        except K.KitError:                                              # no model class: the skeleton only behind KYOJIN_SKELETON; otherwise the KitError as before
            if not (SK.enabled() and SK.has_rows(arch)):
                raise
            g.close()
            return SK.register(path)
        fields = {f.name for f in dataclasses.fields(Args)}
        if cfg is None:
            cfg, _ = K.kit_config(arch, g.kv, fields, vocab=g.tensors["token_embd.weight"].ne[1], tied="output.weight" not in g.tensors,
                                  allow_keys=allow_keys, allow_flags=allow_flags, tensors=tuple(g.tensors))
            cfg["model_type"] = mt
            nt = int(g.kv[f"{arch}.block_count"]) - int(g.kv.get(f"{arch}.nextn_predict_layers", 0)) if g.kv.get(f"{arch}.nextn_predict_layers") else None
        else:
            nt = CR.n_trunk(arch, g.kv)
            nt = nt if nt != int(g.kv[f"{arch}.block_count"]) else None
        from mlx.utils import tree_flatten
        names = [k for k, _ in tree_flatten(Model(Args.from_dict(cfg)).parameters())]
        if names and all(n.startswith("language_model.") for n in names if not n.startswith("vision_tower.")):      # VL wrapper classes: the text trunk sits under it
            names = [n[len("language_model."):] for n in names if n.startswith("language_model.")]
        nm = K.resolve(arch, names, n_trunk=nt)
    finally:
        g.close()
    bad = [k for k, v in nm.blk.items() if not v.startswith(_PRE)]
    if bad:
        raise K.KitError(f"{arch}: block tensors {bad} are not under model.layers: needs a hand-written row")
    top = {k + sfx: v + sfx for k, v in nm.top.items() for sfx in (".weight", ".bias")}
    hf_arch = tuple(h for c in G.CONVERTER.get(arch, ()) for h in c["hf_arch"])[:1]
    # the loader groups routed experts under `<container>.experts.<proj>` (foreign_linear._read_gguf) whatever the class calls its stacked module (`switch_mlp`); a router is never a quantised linear
    blk = {k: v[len(_PRE):].replace(".switch_mlp.", ".experts.") for k, v in nm.blk.items()}
    routers = tuple(v + ".weight" for k, v in blk.items() if k.startswith("ffn_gate_inp"))
    ckey = "kit:" + arch
    N._CONFIG[ckey] = lambda gf, a, _cfg=cfg: {k: v for k, v in _cfg.items() if k not in ("model_type", "architectures")}
    exact = {k[len("blk.{bid}."):]: v[len(_PRE):] for k, v in nm.exact.items() if v.startswith(_PRE)}      # tensors whose name has no .weight / .bias form (gemma4 layer_output_scale, router scales)
    N.ARCHS[arch] = N.Arch(mt, hf_arch, blk=blk, exact=exact, top=top, dense=routers, skip_nextn=nt is not None, xform=K.vetted(arch, "xform", ""), config=ckey)
    return arch

# SPDX-License-Identifier: Apache-2.0
"""Correctness-level loader of a GGUF whose architecture has no hand-written table: generated llama.cpp tables (foreign_archkit) + the stock mlx-lm class.
Every tensor is decoded to a dense array (foreign_gguf.dequant) and loaded into the class: no stored-format kernels, no speed. It is the day-0 tier:
the logits of a new architecture can be compared with llama.cpp's before any engine work."""
from __future__ import annotations

import dataclasses
import importlib
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from . import foreign_archkit as K, foreign_cfgrows as CR, foreign_gguf as GF, foreign_xform as XF


def N_TRUNK(kv, arch):
    from . import foreign_names as N
    return N.n_trunk_of(kv, arch)


def _classes(arch: str):
    """(Model class, ModelArgs class, model_type): the first converter HF architecture whose snake-case name is an mlx-lm model."""
    from . import foreign_vlm_classes as VC
    try:
        Model, Args, mt, _ = VC.resolve(K.model_type_candidates(arch), families=False)          # the one lookup (serve loader and day0_inspect use it too): mlx-lm first, mlx-vlm only with KYOJIN_KIT_VLM_CLASSES
    except LookupError:
        raise K.KitError(f"{arch}: no mlx-lm model for {K.model_type_candidates(arch)}") from None
    return Model, Args, mt


def _has_class(arch: str) -> bool:
    try:
        _classes(arch)
        return True
    except K.KitError:
        return False


def load_cpu(path, dtype=mx.float32, allow_keys=(), allow_flags=(), verbose=False, hand_config=False, skeleton=False):
    """-> (model, report). Raises KitError rather than guess. Config: kit_config from the generated metadata rows; the config rows of foreign_cfgrows for the architectures that have them;
    hand_config=True forces the hand builders of foreign_names (the reference)."""
    g = GF.GgufFile(GF.find_gguf(path))
    try:
        arch = g.kv["general.architecture"]
        if not K.have(arch):
            raise K.KitError(f"GGUF architecture {arch!r} is not in the generated tables")
        from . import foreign_skeleton as SK
        if skeleton or (SK.enabled() and SK.has_rows(arch)):       # the config-driven skeleton (rows only): behind KYOJIN_SKELETON (flags.py arch_skeleton, default off); forced even where a model class exists
            return SK.load_cpu(path, dtype)
        rows = not hand_config and CR.has_rows(arch)       # the 4 archs whose config is not derivable by key name alone carry declarative config rows (foreign_cfgrows)
        if hand_config or rows:   # config from the hand-written builder (foreign_names, the reference) or from the rows; names / tensors / transforms from the generated tables
            from mlx_lm.utils import _get_classes
            from . import foreign_names as N
            cfg = N.gguf_config(g) if hand_config else CR.build(arch, g)
            from . import foreign_vlm_classes as VC
            Model, Args = VC.classes_for_config(cfg)
            mt, rep = cfg["model_type"], dict(used=[], unmapped=[])
        else:
            Model, Args, mt = _classes(arch)
        fields = {f.name for f in dataclasses.fields(Args)}
        vocab = g.tensors["token_embd.weight"].ne[1]
        if not (hand_config or rows):
            cfg, rep = K.kit_config(arch, g.kv, fields, vocab=vocab, tied="output.weight" not in g.tensors, allow_keys=allow_keys, allow_flags=allow_flags, tensors=tuple(g.tensors))
            cfg["model_type"] = mt
        miss = [] if (hand_config or rows) else [f.name for f in dataclasses.fields(Args) if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in cfg]
        if miss:
            raise K.KitError(f"{arch}: the {mt} args class needs {miss} and the GGUF metadata does not give them")
        model = Model(Args.from_dict(cfg))
        from . import foreign_vlm_classes as VC
        if VC.enabled():
            VC.drop_vision(model, g.tensors)
        names = [k for k, _ in tree_flatten(model.parameters())]
        pre = "language_model." if names and all(n.startswith("language_model.") for n in names if not n.startswith("vision_tower.")) else ""     # VL wrapper classes: the text trunk sits under it
        names = [n[len(pre):] for n in names if n.startswith(pre)]
        nm = K.resolve(arch, names, n_trunk=N_TRUNK(g.kv, arch) if hand_config else (CR.n_trunk(arch, g.kv) if rows else None))
        w, skipped = {}, []
        for name, ti in g.tensors.items():
            if name == "rope_freqs.weight":
                if name in K.vetted(arch, "consumed"):          # read by the config row (ROTATED), which refuses anything but a partial-rotation factor list
                    skipped.append(name)
                    continue
                raise K.KitError("rope_freqs.weight: the converter baked rope scaling into the file; not mapped")
            hf = nm.hf(name)
            if hf is None:
                skipped.append(name)
                continue
            a = g.tensor_f32(name).astype(np.float32)
            xf = XF.plan(name, K.vetted(arch, 'xform', ''), g.kv, ti.ne[-1] if len(ti.ne) == 2 else None)
            if xf is not None:
                if xf.cols is not None:                                   # the stored op reads x_hf[..., cols]: W_hf[:, cols] = W_stored
                    a = a[:, np.argsort(xf.cols, kind="stable")]
                a = a[xf.rows] if xf.rows is not None else a
                a = XF.POST[xf.post](a).astype(np.float32) if xf.post else a
            w[hf if hasattr(model, "sanitize") else pre + hf] = mx.array(a).astype(dtype)       # a sanitize() that knows the wrapper adds the prefix itself (qwen3_5)
            if verbose:
                print(f"{name:34s} {ti.tname:7s} -> {hf}")
        pre_sanitize = dict(w)
        if hasattr(model, "sanitize"):
            w = model.sanitize(w)
        model.load_weights(list(w.items()), strict=True)       # a parameter without a tensor, or a tensor without a parameter, raises
        mx.eval(model.parameters())
        return model, dict(weights=pre_sanitize, arch=arch, model_type=mt, config=cfg, names=nm, skipped=skipped, kit=rep, n_tensors=len(w))
    finally:
        g.close()

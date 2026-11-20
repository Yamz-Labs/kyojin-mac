# SPDX-License-Identifier: Apache-2.0
"""Text model classes that live in mlx-vlm only (cohere2_moe, flint ...), for the arch kit.

The kit's class lookup (foreign_archkit_cpu._classes and the config-row path) takes mlx-lm classes. With KYOJIN_KIT_VLM_CLASSES on, an architecture for which mlx-lm has NO
class (every candidate fails) falls back to the mlx-vlm module of the same model_type. mlx-lm is always asked first, so no architecture that resolves today changes class
(tests/test_arch_kit_vlm_classes.py proves it on the census). A vlm wrapper class is built from the flat text config; its vision tower, if it has one, is replaced by an
empty module when the file carries no vision tensors, so strict weight loading sees only the language model.
"""
from __future__ import annotations

import importlib
import os

ENV = "KYOJIN_KIT_VLM_CLASSES"


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "")


def module_names() -> dict:
    import pkgutil
    import mlx_vlm.models as pkg
    return {m.name.replace("_", ""): m.name for m in pkgutil.iter_modules(pkg.__path__)}


def find(candidates) -> tuple | None:
    """(Model class, config class, model_type) of the first candidate that is an mlx-vlm model module, or None."""
    if not enabled():
        return None
    try:
        mods = module_names()
    except ImportError:
        return None
    for mt in candidates:
        mod = mods.get(mt.replace("_", ""))
        if mod is None:
            continue
        try:
            m = importlib.import_module(f"mlx_vlm.models.{mod}")
        except (ImportError, SystemExit):
            continue
        try:
            importlib.import_module(f"mlx_vlm.models.{mod}.language")        # a text trunk module: encoders and embedding classes (bert, clef, modernbert) have none
        except ImportError:
            continue
        cfg = getattr(m, "ModelConfig", None) or getattr(m, "ModelArgs", None)
        if cfg is None or not hasattr(m, "Model"):
            continue
        return m.Model, cfg, mod
    return None


# model_type -> (mlx-vlm package, unproven-row gate or None) of a family whose model code only mlx-vlm has. glm5_next is always on (a served model); the others only while
# their rows are (foreign_unproven switch). This is the one table: foreign_linear.VLM_FAMILIES is this dict.
FAMILIES = {"glm5_next": ("mlx_vlm.models.glm5_next", None), "qwen4_exp": ("mlx_vlm.models.qwen4_exp", "qwen4exp")}


def family(model_type):
    """The mlx-vlm package for `model_type` when it is a FAMILIES entry whose gate is on, else None."""
    pkg, gate = FAMILIES.get(model_type, (None, None))
    if pkg is None or gate is None:
        return pkg
    from . import foreign_unproven as UP
    return pkg if UP.on(gate) else None


def resolve(candidates, config: dict | None = None, families: bool = True) -> tuple:
    """THE class lookup, used by the serve loader (config given), the arch kit and day0_inspect (candidates only).
    -> (Model, config class, name, source) with source "family" | "mlx-lm" | "mlx-vlm". Order: (1) a FAMILIES entry (families=True; the kit's census passes False so its
    verdicts for the hand-row families do not move); (2) mlx-lm (mlx_lm.utils._get_classes on the config when given, else the module of each candidate); (3) with KYOJIN_KIT_VLM_CLASSES
    on, the mlx-vlm text class of the same model_type. Raises ValueError (config given) or LookupError (candidates only) when nothing resolves."""
    candidates = [c for c in candidates if c]
    if families:
        for mt in candidates:
            pkg = family(mt)
            if pkg is not None:
                m = importlib.import_module(pkg)
                return m.Model, m.ModelConfig, mt, "family"
    err = None
    if config is not None:
        from mlx_lm.utils import _get_classes
        try:
            a, b = _get_classes(config=dict(config))
            return a, b, config.get("model_type", ""), "mlx-lm"
        except (ValueError, ImportError, SystemExit) as e:
            err = e
    else:
        import pkgutil
        import mlx_lm.models as pkg
        mods = {m.name.replace("_", ""): m.name for m in pkgutil.iter_modules(pkg.__path__)}
        for mt in candidates:
            name = mods.get(mt.replace("_", ""))
            if name is None:
                continue
            try:
                m = importlib.import_module(f"mlx_lm.models.{name}")
            except (ImportError, SystemExit):                          # an mlx-lm module that needs a missing package (olmo) exits instead of raising
                continue
            return m.Model, m.ModelArgs, name, "mlx-lm"
    r = find(candidates)
    if r is not None:
        return r[0], r[1], r[2], "mlx-vlm"
    if err is not None:
        raise err
    raise LookupError(f"no model class for {candidates}")


def classes_for_config(cfg: dict) -> tuple:
    """(Model, config class) for a config dict (row path): mlx-lm first, mlx-vlm only when mlx-lm has none and the switch is on. No family step (as before)."""
    r = resolve([cfg.get("model_type", "")], config=cfg, families=False)
    return r[0], r[1]


def drop_vision(model, tensor_names) -> bool:
    """Replace the vision tower / projector of a vlm wrapper by empty modules when the file has no vision tensors. True if something was replaced."""
    import mlx.nn as nn
    if any(n.startswith(("v.", "mm.", "visual.")) for n in tensor_names):
        return False
    done = False
    for attr in ("vision_tower", "vision_model", "multi_modal_projector", "embed_vision"):
        if isinstance(getattr(model, attr, None), nn.Module):
            setattr(model, attr, nn.Module())
            done = True
    return done

# SPDX-License-Identifier: Apache-2.0
"""One switch for every arch-kit row that has no Metal identity proof yet.

A row (a config row of foreign_cfgrows.ROWS, a VETTED line of foreign_archkit, a model_type alias, the rows of a rows-only module) that was added without a Metal proof carries
`proven=False` in its table (or sits in UNPROVEN_ALIASES / UNPROVEN_ARCHS below). Such a row does not exist while KYOJIN_KIT_UNPROVEN_ROWS is off (default): every load path and
tools/day0_inspect.py behave as without it. KYOJIN_GEMMA4_ROWS stays as an alias of the same switch for gemma4 alone. Declared in flags.py (kit_unproven_rows).
"""
from __future__ import annotations

import os

ENV = "KYOJIN_KIT_UNPROVEN_ROWS"
LEGACY = {"gemma4": "KYOJIN_GEMMA4_ROWS"}                     # per-architecture alias kept from the first version of the gate
UNPROVEN_ALIASES = {"minimax-m3": ("minimax_m3_vl",), "nemotron_h_moe": ("nemotron_h",)}      # model_type aliases of foreign_archkit.model_type_candidates
UNPROVEN_ARCHS = ("qwen4exp", "hy_v4_flash")       # glm5-next left with its Metal layer proof (tools/glm_gguf_layer_proof.py, layers 3 + 4); architectures whose rows live in a rows-only module that registers them only while the switch is on


def _set(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "")


def on(arch: str | None = None) -> bool:
    return _set(ENV) or (arch is not None and arch in LEGACY and _set(LEGACY[arch]))


def has_unproven(arch: str | None) -> bool:
    """True when some row of this architecture is marked unproven (whatever the switch says)."""
    if not arch:
        return False
    from . import foreign_archkit as K, foreign_cfgrows as CR
    return (arch in UNPROVEN_ALIASES or arch in UNPROVEN_ARCHS or CR.ROWS.get(arch, {}).get("proven") is False or K.VETTED.get(arch, {}).get("proven") is False)


def gated_off(arch: str) -> bool:
    """The config row / vetted line of `arch` is unproven and the switch is off."""
    from . import foreign_archkit as K, foreign_cfgrows as CR
    unproven = CR.ROWS.get(arch, {}).get("proven") is False or K.VETTED.get(arch, {}).get("proven") is False
    return unproven and not on(arch)


def aliases(arch: str) -> tuple:
    return UNPROVEN_ALIASES.get(arch, ()) if on(arch) else ()


def hint(arch: str | None) -> str | None:
    """The one line day0_inspect adds while rows exist but are off."""
    if has_unproven(arch) and not on(arch):
        return f"unproven rows exist for this architecture: set {ENV} to try them"
    return None

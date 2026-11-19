# SPDX-License-Identifier: Apache-2.0
"""Op-named lever flags, their model-named aliases, and the per-architecture defaults (ArchSpec.defaults). Pure python (no mlx): the audit and the tests read it.

Why: the speed levers of the GLM lane were named after the model (KYOJIN_GLM_VROWS, KYOJIN_GLM_HOSTGAP, ...) and switched on by a lane file (flags.env), so a new
architecture got none of them and the engine did not know which levers a model wants. Here every lever has ONE op name (what the code does, not who uses it); the old
name stays a working alias (no rename sweep: tests, scripts and flags.env keep their spelling), and an ArchSpec carries the architecture's default values.

Mechanism (env mirror, no lever module changes): the lever modules keep reading the spelling they always read. `sync_env` copies the value of whichever spelling the
environment sets onto every other spelling of the same lever, BEFORE those modules are imported (kyojin_mlx/__init__ -> flags.apply_defaults, and load_backend for the
architecture defaults). Conflict rule: the op name wins over its aliases (logged). `apply_arch_defaults` sets a lever of ArchSpec.defaults only when no spelling of it
is set by the user (a start value that flags.apply_defaults filled in is not "set by the user")."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    env: str                      # the op name (canonical)
    aliases: tuple = ()           # model-named spellings that keep working
    what: str = ""                # one line: the op
    module: str = ""              # kyojin_mlx module that reads the variable once at import (stale check), "" = read per call / by a serve object


OPS: dict[str, Op] = {o.env: o for o in (
    # verify-rows GEMV: a dense one-byte projection of a speculative verify block (B * L <= MAX rows) is ONE launch that reads the weights once
    Op("KYOJIN_VERIFY_ROWS_GEMV", ("KYOJIN_GLM_VROWS",), "verify-rows dense GEMV, one launch, weights read once", "glm_dense_rows"),
    Op("KYOJIN_VERIFY_ROWS_GEMV_MAX", ("KYOJIN_GLM_VROWS_MAX",), "most rows of that launch", "glm_dense_rows"),
    # host loop: async-submit the decode step in groups of layers while the host is still building the rest
    Op("KYOJIN_HOST_EARLY_SUBMIT", ("KYOJIN_GLM_HOSTGAP",), "host loop: early submit of the decode step in groups of layers", "glm_hostgap"),
    Op("KYOJIN_HOST_EARLY_SUBMIT_FIRST", ("KYOJIN_GLM_HOSTGAP_FIRST",), "layers built before the first submit", "glm_hostgap"),
    Op("KYOJIN_HOST_EARLY_SUBMIT_EVERY", ("KYOJIN_GLM_HOSTGAP_EVERY",), "layers between two submits", "glm_hostgap"),
    Op("KYOJIN_HOST_EARLY_SUBMIT_ROWS", ("KYOJIN_GLM_HOSTGAP_ROWS",), "most rows per session of a pass that submits early", "glm_hostgap"),
    # one-byte dense GEMV (G3): fp8 / MXFP8 linears of k % 512 == 0, rows <= 4
    Op("KYOJIN_DENSE_GEMV_1B", ("KYOJIN_GLM_DENSE_GEMV",), "one-byte (fp8 / mxfp8) dense GEMV for rows <= 4", "foreign_linear"),
    # prefill: tiled_linear (256-row slices) off for batch 1; the lm_head last-row cut is KYOJIN_HEAD_LAST (already op-named, default on)
    Op("KYOJIN_PF_UNTILE", ("KYOJIN_GLM_PF_UNTILE",), "prefill: router / indexer projections as one GEMM instead of 256-row tiles", "glm_pf_untile"),
    # MoE expert kernel E (decode, NVFP4 routed experts, one row per slot)
    Op("KYOJIN_MOE_EXPERT_E", ("KYOJIN_GLM_MOE_E",), "decode MoE expert kernel E", ""),
    # fused sparse-attention (DSA) decode
    Op("KYOJIN_DSA_FUSE", ("KYOJIN_GLM_FUSE_DSA",), "fused DSA decode (index + attention)", "glm_fuse_dsa"),
    Op("KYOJIN_DSA_FUSE_EVAL", ("KYOJIN_GLM_FUSE_DSA_EVAL",), "fused DSA: async_eval of the attention on a verify pass", "glm_fuse_dsa"),
    Op("KYOJIN_DSA_FAST", ("KYOJIN_GLM_DSA_FAST",), "DSA decode fast path", "glm_dsa"),
    # KDA (linear attention) prefill
    Op("KYOJIN_KDA_PF", ("KYOJIN_GLM_KDA_PF",), "KDA linear-attention prefill scan", "kda_pf"),
    # grouped-expert GEMM on stored rows (NVFP4 experts, clamped SwiGLU). Written for any foreign-format MoE, OFF unless the architecture's defaults or the env turn it on.
    Op("KYOJIN_KQ_GMM_STORED", ("KYOJIN_KQ_GMM_GLM",), "grouped-expert GEMM on stored NVFP4 rows (prefill)", ""),
)}

ALIAS_OF: dict[str, str] = {a: o.env for o in OPS.values() for a in o.aliases}


def spellings(env: str) -> tuple:
    """Every spelling of the lever `env` is one of (op name first); (env,) for a variable that has no op name."""
    op = OPS.get(ALIAS_OF.get(env, env))
    return (op.env, *op.aliases) if op else (env,)


def canonical(env: str) -> str:
    return ALIAS_OF.get(env, env)


# start values that flags.apply_defaults put into os.environ: not an explicit choice of the user
_FILLED: dict[str, str] = {}


def note_filled(env: str, value: str) -> None:
    _FILLED[env] = value


def user_set(env: str, environ=None) -> bool:
    e = os.environ if environ is None else environ
    if env not in e:
        return False
    return not (e is os.environ and _FILLED.get(env) == e[env])


def sync_env(environ=None, log=None) -> dict:
    """Mirror each lever onto all its spellings. Returns {op env: value} of the levers that were set by the user. The op name wins a conflict (logged)."""
    e = os.environ if environ is None else environ
    out = {}
    for op in OPS.values():
        sp = (op.env, *op.aliases)
        got = [(n, e[n]) for n in sp if user_set(n, e)]
        if not got:
            continue
        val = e[op.env] if user_set(op.env, e) else got[0][1]
        if log and len({v for _, v in got}) > 1:
            log(f"[flags] {op.env}: spellings disagree ({', '.join(f'{n}={v}' for n, v in got)}); the op name wins" if user_set(op.env, e)
                else f"[flags] {op.env}: aliases disagree ({', '.join(f'{n}={v}' for n, v in got)}); using {got[0][0]}")
        for n in sp:
            e[n] = val
        out[op.env] = val
    return out


def resolve(environ: dict, defaults: dict | None = None) -> dict:
    """Pure: {canonical env: value} of a plain dict environment plus the architecture defaults (the user's values win). No os.environ, no side effect."""
    e = dict(environ)
    sync_env(e)
    res = {canonical(k): v for k, v in e.items() if k.startswith("KYOJIN_")}
    for k, v in (defaults or {}).items():
        if not any(n in e for n in spellings(k)):
            res[canonical(k)] = str(v)
    return res


def validate_defaults(defaults: dict) -> list:
    """Problems of an ArchSpec.defaults dict: a key must be a canonical op name or a plain KYOJIN_* variable that has no op name (never an alias)."""
    bad = []
    for k, v in defaults.items():
        if k in ALIAS_OF:
            bad.append(f"{k}: alias of {ALIAS_OF[k]}; key defaults by the op name")
        elif not k.startswith("KYOJIN_"):
            bad.append(f"{k}: not a KYOJIN_* variable")
        if not isinstance(v, str):
            bad.append(f"{k}: value {v!r} is not a string")
    return bad


def apply_arch_defaults(spec, environ=None, log=print) -> dict:
    """Set the levers of `spec.defaults` that the user did not set (any spelling), on every spelling. Logs what was defaulted and what the environment kept.
    Returns {env: value} of the defaulted levers. Call before the lever modules are imported (they read their variable once at import)."""
    defaults = getattr(spec, "defaults", None) or {}
    if not defaults:
        return {}
    e = os.environ if environ is None else environ
    if e.get("KYOJIN_ARCH_DEFAULTS", "0").strip().lower() not in ("1", "on", "true", "yes"):
        # master switch, OFF until the GPU greedy-identity proof (STATUS_E4A.md) and review: benches whose arms rely on
        # "unset = off" keep their meaning; the lane keeps its flags.env.
        return {}
    sync_env(e, log)
    applied, kept = {}, {}
    for k, v in defaults.items():
        sp = spellings(k)
        mine = [n for n in sp if user_set(n, e)]
        if mine:
            kept[k] = e[mine[0]]
            continue
        for n in sp:
            e[n] = str(v)
        applied[k] = str(v)
    if log:
        log(f"[kyojin] arch {spec.name} defaults (env unset): " + (", ".join(f"{k}={v}" for k, v in applied.items()) or "none")
            + (f"; kept from env: {', '.join(f'{k}={v}' for k, v in kept.items())}" if kept else ""))
        late = [m for k in applied if (m := OPS[k].module if k in OPS else "") and f"kyojin_mlx.{m}" in sys.modules]
        if late:
            log(f"[kyojin] WARNING arch defaults reached modules already imported ({', '.join(sorted(set(late)))}): they read their flag once at import; set the env at launch")
    return applied

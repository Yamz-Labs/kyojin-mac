# SPDX-License-Identifier: Apache-2.0
"""Serve defaults derived from the pack, so `python -m kyojin_mlx.serve --model PATH` needs no per-model flags.

Every value here is either read from the pack (config.json limits, the draft-head tensors, the chat template) or a generic
constant of the engine. Nothing keys on the model directory name. An explicit CLI flag always wins.

    ctx                 max_position_embeddings of the pack (text_config first), else 32768
    prefill chunk       8192 for a long-context pack (>= 131072 tokens), else the memory-driven auto chunk of server.resolve_chunk
    prefix block        smallest multiple of the chunk >= 512
    prefix cache MB     ctx / 64, between 1024 and 16384 (1M ctx -> 16384 MB)
    kv disk             user cache dir keyed by the pack, 100 GiB cap, block = 8192 tokens (--no-kv-disk opts out)
    mtp depth           the pack's draft block size (DS4.1: dspark_block_size 5) when draft-head tensors exist, else 0
    thinking            on when the family / chat template has a thinking switch
    reasoning effort    "medium"
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SESSIONS = 4
DEFAULT_CTX = 32768
PREFILL_CHUNK = 8192                       # the engine's hard cap (serve.backends.MAX_PREFILL_CHUNK)
LONG_CONTEXT_TOKENS = 16 * PREFILL_CHUNK   # from here a pack is served with the large chunk
PREFIX_BLOCK_MIN = 512
PREFIX_CACHE_MIN_MB = 1024
PREFIX_CACHE_MAX_MB = 16384
PREFIX_CACHE_TOKENS_PER_MB = 64            # ctx / 64 MB: 1M -> 16 GiB, 128K -> 2 GiB
DEFAULT_MTP_DEPTH = 5                      # when the config has no draft block size
KV_DISK_GIB = 100.0
KV_DISK_BLOCK = 8192
DEFAULT_REASONING_EFFORT = "medium"


@dataclass
class Pack:
    """What the resolver needs to know about a pack; read without loading any tensor or importing MLX."""
    path: str
    cfg: dict
    max_ctx: int
    family: str | None
    has_draft_head: bool
    draft_block: int | None
    thinking: bool


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def read_config(path: str) -> dict:
    with open(os.path.join(path, "config.json"), "rb") as fh:
        return json.loads(fh.read())


def has_draft_head(path: str, mtp_head: str | None = None) -> bool:
    """`mtp.*` tensors in the pack (a `mtp/` sidecar, the index weight map, or a safetensors header). Same rule as backends.has_mtp_tensors,
    kept here without importing the MLX stack so --print-config stays cheap."""
    import glob
    import struct
    root = mtp_head or path
    from ..foreign_names import gguf_has_mtp
    if gguf_has_mtp(root):                                                 # the nextn block of a GGUF
        return True
    if glob.glob(os.path.join(root, "mtp", "*.safetensors")):
        return True
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "rb") as fh:
            if any(k.startswith("mtp.") for k in json.loads(fh.read()).get("weight_map", {})):
                return True
    for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if any(k.startswith("mtp.") for k in json.loads(fh.read(n))):
                    return True
        except (OSError, ValueError, struct.error):
            continue
    return False


def _template_text(path: str, chat_template: str | None) -> str:
    if chat_template:
        try:
            return Path(chat_template).read_text()
        except OSError:
            return ""
    out = []
    jinja = Path(path) / "chat_template.jinja"
    if jinja.is_file():
        out.append(jinja.read_text())
    tc = Path(path) / "tokenizer_config.json"
    if tc.is_file():
        try:
            tpl = json.loads(tc.read_text()).get("chat_template")
        except (OSError, ValueError):
            tpl = None
        if isinstance(tpl, str):
            out.append(tpl)
        elif isinstance(tpl, list):
            out.extend(str(t.get("template", "")) for t in tpl if isinstance(t, dict))
    return "\n".join(out)


def template_has_thinking(path: str, family: str | None, chat_template: str | None = None) -> bool:
    """A thinking switch in the chat template, or a builtin renderer that carries one (the DS4.1 family)."""
    if family == "ds41":                       # kyojin_mlx.serve.ds41_chat: thinking / reasoning_effort are part of the renderer
        return True
    text = _template_text(path, chat_template).lower()
    return any(k in text for k in ("enable_thinking", "thinking_mode", "reasoning_effort", "<think"))


def family_of_pack(cfg: dict, path: str | None = None) -> str | None:
    from .chat import family_of, family_of_gguf
    tc = _text_config(cfg)
    if path is not None and cfg.get("model_type") and not modeldesc_backend(cfg.get("model_type")):
        from .. import foreign_linear as FL
        if FL.is_foreign(path) == "gguf":                      # bare GGUF of a mapped architecture without a table row: generic (or the Qwen one for qwen2)
            return family_of_gguf(cfg["model_type"])
    for model_type in (cfg.get("model_type"), tc.get("model_type")):
        if not model_type:
            continue
        try:
            return family_of(model_type)
        except ValueError:
            continue
    return None


def modeldesc_backend(model_type: str):
    from .. import modeldesc
    return modeldesc.backend_desc(model_type)


def read_pack(path: str, mtp_head: str | None = None, chat_template: str | None = None) -> Pack:
    cfg = read_config(path)
    tc = _text_config(cfg)
    ctx = tc.get("max_position_embeddings") or cfg.get("max_position_embeddings") or DEFAULT_CTX
    family = family_of_pack(cfg, str(path))
    block = tc.get("dspark_block_size") or cfg.get("dspark_block_size")
    return Pack(path=str(path), cfg=cfg, max_ctx=int(ctx), family=family,
                has_draft_head=has_draft_head(path, mtp_head), draft_block=int(block) if block else None,
                thinking=template_has_thinking(path, family, chat_template))


# ------------------------------------------------------------------------------------------------------------------ generic rules
def default_prefill_chunk(ctx: int) -> int | None:
    """Long-context packs get the engine's largest chunk; short packs keep the free-memory auto chunk."""
    return PREFILL_CHUNK if ctx >= LONG_CONTEXT_TOKENS else None


def default_prefix_block(chunk: int | None) -> int | None:
    return None if chunk is None else -(-PREFIX_BLOCK_MIN // chunk) * chunk


def default_prefix_cache_mb(ctx: int) -> int:
    return int(min(PREFIX_CACHE_MAX_MB, max(PREFIX_CACHE_MIN_MB, -(-int(ctx) // PREFIX_CACHE_TOKENS_PER_MB))))


def default_mtp_depth(pack: Pack) -> int:
    if not pack.has_draft_head:
        return 0
    return max(1, min(pack.draft_block or DEFAULT_MTP_DEPTH, 8))


def _safe(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._")
    return (out or "pack")[:48]


def pack_key(path: str) -> str:
    """Stable, human-readable key of a pack: directory name + config bytes + weight file names and sizes (no mtime, so a copy keeps its cache)."""
    h = hashlib.blake2b(digest_size=8)
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        with open(cfg, "rb") as fh:
            h.update(fh.read())
    try:
        names = sorted(os.listdir(path))
    except OSError:
        names = []
    for f in names:
        if f.endswith(".safetensors") or (f.endswith(".json") and f != "config.json"):
            try:
                st = os.stat(os.path.join(path, f))
            except OSError:
                continue
            h.update(f"{f}:{st.st_size};".encode())
    return f"{_safe(os.path.basename(os.path.normpath(path)))}-{h.hexdigest()}"


def user_cache_dir() -> str:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return str(base / "kyojin_mlx" / "kvdisk")


def default_kv_disk_dir(pack: Pack) -> str:
    return os.path.join(user_cache_dir(), pack_key(pack.path))


# ------------------------------------------------------------------------------------------------------------------ resolver
def resolve_options(pack: Pack, *, sessions: int = DEFAULT_SESSIONS, ctx: int | None = None,
                    prefill_chunk: int | None = None, prefix_block: int | None = None,
                    prefix_cache_mb: float | None = None, kv_disk: str | None = None,
                    no_kv_disk: bool = False, kv_disk_gib: float | None = None,
                    kv_disk_block: int = KV_DISK_BLOCK, mtp_depth: int | None = None,
                    thinking: bool | None = None, reasoning_effort: str | None = None) -> dict:
    """Effective serve options for a pack. Every None argument means "derive it"; explicit values win."""
    ctx = int(ctx) if ctx is not None else pack.max_ctx
    chunk = int(prefill_chunk) if prefill_chunk is not None else default_prefill_chunk(ctx)
    block = int(prefix_block) if prefix_block is not None else default_prefix_block(chunk)
    cache_mb = float(prefix_cache_mb) if prefix_cache_mb is not None else default_prefix_cache_mb(ctx)
    disk = kv_disk
    if disk is None and not no_kv_disk:
        disk = default_kv_disk_dir(pack)
    gib = float(kv_disk_gib) if kv_disk_gib is not None else (KV_DISK_GIB if disk else 0.0)
    depth = int(mtp_depth) if mtp_depth is not None else default_mtp_depth(pack)
    think = bool(thinking) if thinking is not None else pack.thinking
    effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    return {"sessions": int(sessions), "ctx": ctx, "prefill_chunk": chunk, "prefix_block": block,
            "prefix_cache_mb": cache_mb, "kv_disk": disk, "kv_disk_gib": gib,
            "kv_disk_block": int(kv_disk_block), "mtp_depth": depth, "thinking": think,
            "reasoning_effort": effort}


def _plain_args(args):
    """A positional pack path: fixed defaults, nothing derived. Benches and tools start the server this way and must not meet
    a disk KV tier or a larger context they did not ask for."""
    for key, value in (("ctx", DEFAULT_CTX), ("prefix_cache_mb", PREFIX_CACHE_MIN_MB), ("kv_disk_gib", 0.0)):
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args._resolved = True
    return args


def resolve_serve_args(args):
    """`--model PATH` (or --print-config): fill the Namespace with the derived defaults. A positional path: `_plain_args`.
    Idempotent. Keeps `_pack` for --print-config."""
    if getattr(args, "_resolved", False):
        return args
    if not (getattr(args, "model_opt", None) or getattr(args, "print_config", False)):
        return _plain_args(args)
    pack = read_pack(args.model, getattr(args, "mtp_head", None), getattr(args, "chat_template", None))
    depth = getattr(args, "mtp_depth", None)
    if depth is None and os.environ.get("KYOJIN_MTP_DEPTH") is not None:
        depth = int(os.environ["KYOJIN_MTP_DEPTH"])            # env is a convenience override; the CLI still wins
    opts = resolve_options(pack, sessions=getattr(args, "sessions", DEFAULT_SESSIONS),
                           ctx=getattr(args, "ctx", None), prefill_chunk=getattr(args, "prefill_chunk", None),
                           prefix_block=getattr(args, "prefix_block", None),
                           prefix_cache_mb=getattr(args, "prefix_cache_mb", None),
                           kv_disk=getattr(args, "kv_disk", None),
                           no_kv_disk=getattr(args, "no_kv_disk", False),
                           kv_disk_gib=getattr(args, "kv_disk_gib", None),
                           kv_disk_block=getattr(args, "kv_disk_block", KV_DISK_BLOCK), mtp_depth=depth,
                           thinking=getattr(args, "thinking", None),
                           reasoning_effort=getattr(args, "reasoning_effort", None))
    for key, value in opts.items():
        setattr(args, key, value)
    args._pack = pack
    args._resolved = True
    return args


def effective_config(args) -> dict:
    """JSON-able view of the options --print-config/autoserve_diff compare."""
    pack = getattr(args, "_pack", None)
    num = lambda v: int(v) if isinstance(v, float) and v.is_integer() else v   # noqa: E731
    return {"model": os.path.normpath(str(args.model)),
            "family": getattr(pack, "family", None),
            "sessions": int(args.sessions), "ctx": int(args.ctx),
            "prefill_chunk": args.prefill_chunk, "prefix_block": args.prefix_block,
            "prefix_cache_mb": num(args.prefix_cache_mb),
            "kv_disk": args.kv_disk, "kv_disk_gib": num(args.kv_disk_gib),
            "kv_disk_block": int(args.kv_disk_block), "mtp_depth": int(args.mtp_depth),
            "thinking": bool(args.thinking), "reasoning_effort": args.reasoning_effort,
            "kv_mode": getattr(args, "kv_mode", None), "no_ced": bool(getattr(args, "no_ced", False))}

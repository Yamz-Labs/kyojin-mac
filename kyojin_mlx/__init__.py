"""EXL3 mul1 decode kernels for Apple Silicon (MLX / Metal) plus a numpy reference."""

import os as _os
from . import flags as _flags
_flags.apply_defaults()                                             # lever start values from the flag table (one source of truth)
if _os.environ.get("KYOJIN_REPLAY", "0") not in ("0", ""):          # the Metal hooks of replay.py must exist before MLX builds the pipelines a replayed region uses
    try:
        from . import replay as _replay
        _replay.install()
    except Exception as _e:                                          # no clang / unknown Metal classes: regions stay eager (replay.region checks again)
        import sys as _sys
        print(f"kyojin_mlx: replay hooks not installed ({_e})", file=_sys.stderr)


def _ds41_argv() -> bool:
    """True when a command-line argument is a DS4.1 pack directory (config.json model_type deepseek_v41): the Engram gate needs the one-buffer MLX settings
    before mlx is imported, and `python -m kyojin_mlx.serve.server PACK` imports this package first."""
    import json as _json, sys as _sys
    for a in _sys.argv[1:]:
        try:
            if _os.path.isdir(a):
                with open(_os.path.join(a, "config.json")) as f:
                    from . import modeldesc as _md          # mlx-free
                    row = _md.by_model_type(_json.load(f).get("model_type", ""))
                    if row is not None and row.one_buffer_gate:
                        return True
        except Exception:
            pass
    return False


import sys as _sys2


def gate_wanted(mode: str) -> bool:
    """The Engram gate is opt-in (KYOJIN_ENGRAM_GATE=1). It needs one command buffer per step, and with the weights wired (serve.backends.wire_memory)
    MLX's default buffers decode faster without it: raw DS4.1 prose/code 40.0/70.9/45.0/47.0 tok/s vs 32.0/62.5/38.0/41.7 gated (Ultra, 29/10).
    "auto" (the default) and "0" are off."""
    return mode == "1"


def apply_ds41_buffer_env(environ=None) -> bool:
    """One command buffer per decode step on a DS4.1 pack when the Engram gate is asked for (KYOJIN_ENGRAM_GATE=1): the gated pipeline needs it.
    Otherwise MLX keeps its default buffer size. The old one-buffer default hid a residency fault: with no MLX wired limit every commit re-made the
    170 GiB set resident (~15 ms) and ungated decode fell to 0.6-2.5 tok/s; wire_memory fixes the cause. Must run before mlx is imported."""
    env = _os.environ if environ is None else environ
    if "mlx.core" in _sys2.modules or not _ds41_argv() or not gate_wanted(env.get("KYOJIN_ENGRAM_GATE", "auto")):
        return False
    env.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000")        # one command buffer per decode step
    env.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
    return True


apply_ds41_buffer_env()


SMALL_RAM_BYTES = 24 * 2 ** 30                                      # below this the machine keeps MLX's own command-buffer sizes (see apply_decode_buffer_env)


def physical_ram() -> int:
    try:
        return _os.sysconf("SC_PHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 1 << 62                                               # unknown: treat as large (the old behaviour)


def apply_lean_malloc_env(environ=None, argv=None, execv=None) -> bool:
    """macOS malloc keeps the large blocks the loader freed (token tables, host packs) as dirty pages of the process: 0.9 GB of footprint on the 16 GB Air
    after the load, 2 GB at its peak. MallocSpaceEfficient=1 returns them at once; libmalloc reads it at process start, so a serve process re-executes itself
    once with it set (same argv: the lane's command line is unchanged). It costs decode speed (Ultra MoE N4 266 -> 249 tok/s, N1 equal, 3 GB saved), so it
    applies only under SMALL_RAM_BYTES, where the gigabytes decide the context that fits. KYOJIN_LEAN_MALLOC=0 off, =2 on any RAM. Returns True when set."""
    env = _os.environ if environ is None else environ
    argv = list(getattr(_sys2, "orig_argv", [])) if argv is None else list(argv)
    mode = env.get("KYOJIN_LEAN_MALLOC", "1")
    if _sys2.platform != "darwin" or mode in ("0", "") or "MallocSpaceEfficient" in env:
        return False
    if mode != "2" and physical_ram() >= SMALL_RAM_BYTES:
        return False
    if not (any(a.startswith("kyojin_mlx.serve") for a in argv[:4]) or any(a.endswith("kyojin_mlx/serve/server.py") for a in argv[:4])):
        return False                                                 # only a serve process (tests and tools import this package too)
    env["MallocSpaceEfficient"] = "1"
    if execv is not None:
        execv(_sys2.executable, [_sys2.executable] + argv[1:])
    return True


def apply_decode_buffer_env(environ=None) -> bool:
    """Larger MLX command buffers for decode: MLX closes a buffer every 50 ops / ~50 MB of weights, which on a 3B-active MoE (2 GB read per token) is ~70
    commits per token and the GPU idles at each one. 200 ops / 400 MB: ~4 commits, N1 decode 60 -> 110 tok/s on Qwen3.6-35B-A3B Q4_K_XL. A DS4.1 pack keeps
    MLX's own sizes (its gate wants one buffer, the ungated lane was tuned on the defaults). Values already in the environment win. Before mlx is imported."""
    env = _os.environ if environ is None else environ
    mode = env.get("KYOJIN_DECODE_BUFFERS", "1")
    if "mlx.core" in _sys2.modules or mode in ("0", "") or _ds41_argv():
        return False
    if mode != "2" and physical_ram() < SMALL_RAM_BYTES:         # a 400 MB buffer keeps every prefill temporary of its ops alive until it completes: 16 GB Air, warm-up out of memory at 12 GB
        return False
    env.setdefault("MLX_MAX_OPS_PER_BUFFER", env.get("KYOJIN_BUFFER_OPS", "200"))
    env.setdefault("MLX_MAX_MB_PER_BUFFER", env.get("KYOJIN_BUFFER_MB", "400"))
    return True


apply_decode_buffer_env()
if "mlx.core" not in _sys2.modules:
    apply_lean_malloc_env(execv=_os.execv)

from . import applegpu as _applegpu
_applegpu.apply_residency_refresh_env()                           # KYOJIN_RESIDENCY_REFRESH_MS -> MLX env; before mlx is imported

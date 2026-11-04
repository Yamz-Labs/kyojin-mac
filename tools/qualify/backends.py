"""Backend profiles: where the python, the engine, the locks and the memory counters live. One entry point per backend, same stages.
This branch (feat/qualify-mlx) carries the `mlx` profile (Apple unified memory; the Air 16 GB for dry runs, the M5 Ultra 256 GB on day 1);
the cuda / rocm profiles live in the CUDA/ROCm qualify tree (same stage names, same qualify.json schema)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

HOME = Path.home()
MAIN = HOME / "kyojin-mlx"                       # main checkout: the venv, the GPU lock, the guard log, the packs live here


def _venv_python(repo: str | None) -> str:
    for v in (os.environ.get("KYOJIN_VENV"), str(MAIN / ".venv")):
        if v and (Path(v) / "bin/python").exists():
            return str(Path(v) / "bin/python")
    return "python3"


def mlx_env(repo: str) -> dict:
    r = Path(repo)
    vlm = os.environ.get("KYOJIN_VLM_PKG") or str(r / "vlm_pkg")
    if not (Path(vlm) / "mlx_vlm").is_dir():
        vlm = str(MAIN / "vlm_pkg")
    return {"PYTHONPATH": ":".join([str(r), vlm, str(r / "tools")]), "PYTHONUNBUFFERED": "1", "KYOJIN_NAX_LOG": "0"}


PROFILES = {
    "mlx": dict(
        python=_venv_python(None), engine=str(MAIN), locks=[os.path.expanduser(os.environ.get("KYOJIN_GPU_LOCK", str(MAIN / ".gpu.lock")))],
        env={}, guard_log=str(MAIN / "macguard/guard.log"), mem_cap_gib=13.0,        # owner/guard rule: one job < 13 GiB footprint on the Air
        gpu_used_mib="mx.get_active_memory", host_peak_note="Apple unified memory",
    ),
}


def profile(name: str, repo: str | None = None) -> dict:
    """repo = the engine checkout (PYTHONPATH + cwd of every server and tool)."""
    if name not in PROFILES:
        raise SystemExit(f"unknown backend {name!r}: {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["env"] = {**p["env"], **mlx_env(repo or p["engine"])}
    return p


def _sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def total_mem_gib() -> float:
    try:
        return int(_sysctl("hw.memsize")) / 2**30
    except ValueError:
        return 0.0


def mem_available_gib() -> float:
    """Free unified memory as `memory_pressure` reports it (free % of the physical memory; includes reclaimable pages)."""
    try:
        t = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"free percentage:\s*(\d+)%", t)
        return total_mem_gib() * int(m.group(1)) / 100 if m else 0.0
    except Exception:
        return 0.0


def gpu_free_mib(backend: str) -> float | None:
    """Memory a model may use on this machine in MiB: Metal's recommended working set, (None when it cannot be
    read). The max-ctx stage caps it with the job's footprint cap and subtracts the weights."""
    try:
        import mlx.core as mx
        ws = mx.device_info()["max_recommended_working_set_size"] / 2**20
    except Exception:
        ws = total_mem_gib() * 0.75 * 1024 or None
    if ws is None:
        return None
    return ws


def footprint_gib(pid: int) -> float | None:
    """Physical footprint of a process in GiB as `top` shows it (NOT ps RSS: MLX buffers are wired / compressed)."""
    try:
        t = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem"], capture_output=True, text=True, timeout=30).stdout
        for line in t.splitlines():
            f = line.split()
            if len(f) == 2 and f[0] == str(pid):
                m = re.match(r"([\d.]+)([KMGT])", f[1])
                return float(m.group(1)) * {"K": 2**-20, "M": 2**-10, "G": 1, "T": 2**10}[m.group(2)] if m else None
    except Exception:
        pass
    return None


def machine_info() -> dict:
    """What the numbers were measured on (the report carries it): chip, memory, power state, NAX."""
    d = dict(chip=_sysctl("machdep.cpu.brand_string"), mem_gib=round(total_mem_gib(), 1))
    try:
        t = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        d["power"] = "AC" if "AC Power" in t else "battery"
        m = re.search(r"(\d+)%", t)
        d["battery_pct"] = int(m.group(1)) if m else None
        lp = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"lowpowermode\s+(\d)", lp)
        d["low_power_mode"] = bool(int(m.group(1))) if m else None
    except Exception:
        pass
    try:
        from kyojin_mlx.metal_nax import nax_status        # feat/naxcap probe (macair42); absent on older branches
        d["nax"] = nax_status()
    except Exception:
        d["nax"] = None
    return d

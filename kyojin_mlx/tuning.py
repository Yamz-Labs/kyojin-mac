# SPDX-License-Identifier: Apache-2.0
"""Per-(device, shape) kernel choices of the dense EXL3 linears, stored next to the pack: <pack>/tuning/<device>.json.

`qualify --stage autotune` (tools/qualify/autotune_mlx.py) times the candidates of every linear shape of a pack on the device and writes the table; `load_exl3`
calls `install(pack)` and every `Exl3Linear` asks `lookup(in, out, K)` for its entry. No file (or a file of another device / kernel version, or
KYOJIN_TUNING=0) = no entry = today's constants, bit-identical.

Entry of shape "<in>x<out>x<K>" (every key optional, a missing key keeps the constant):
  tgk       k-splits of the decode GEMV (default kernels.pick_tgk): partial sums are added in k order, so another tgk rounds at 1e-7 relative
  fused_max rows up to which the fused trellis-decode + NAX matmul runs (default exl3_linear.DENSE_FUSED_MAX); above: decode + MLX matmul
  nax_cfg   {"<rows>": [bmu, nmb, bns, bks]}: NAX tile of the fused matmul for that row count (default metal_nax.pick_cfg); a row count takes the entry of the
            smallest tuned row count >= rows (the largest when above all)
  mma_max   last row count served by the MMA GEMV (rows above, up to 16, run the fused NAX matmul; default `chip()["mma_max"]`; <= GEMV_MAX_ROWS = no MMA)
  nax_ks    {"<rows>": KS}: k-split of the fused NAX matmul (KS threadgroup rows along k, fp32 partials summed by the finalize: more threadgroups for the
            few-row verify band; default exl3_linear.NAX_KS, same row-count lookup as nax_cfg; 1 = no split)

Entry "serve" (not a shape): depth-policy priors of the server in solo decode steps, read with `serve(name, default)` when no --row-cost / --draft-cost flag
is given: row_cost (one more verify row), draft_cost (one draft call); tools/ds41_bench.py --tuning measures and writes them (set_entry).
Entry "prefill" (not a shape): thresholds of the fused prefill seams (mlp_pf, layer_pf), read with `pf(name, default)`; an env var (KYOJIN_MLP_PF, KYOJIN_MLP_PF_MIN_ROWS, KYOJIN_LAYER_PF,
KYOJIN_LAYER_PF_MIN_ROWS, KYOJIN_LAYER_PF_MAX_ROWS, KYOJIN_LAYER_PF_SPLIT_MAX_ROWS) or a module attribute set by a tool wins over the table:
  mlp_pf (1/0), mlp_min_rows      fused MLP middle (gate + up finalize + swiglu + down rotation in one launch) on/off and the smallest chunk that takes it (default 1, 64)
  layer_pf (1/0), layer_min_rows  fused layer seams on/off and the smallest chunk (default 1, 64)
  layer_max_rows                  largest chunk with the one-kernel seam 1 + seam 2 (default 1024: above it the stock seams win on the Air, see layer_pf.py)
  layer_split_max_rows            chunks in (layer_max_rows, this] run seam 1 as two launches (fin+add+norm, then both rotations): bit-identical, default 0 = none

Chip defaults (`chip()`): the thresholds the entries above override per shape, keyed by the device name and core count in chip_defaults.json (a measured chip) and otherwise scaled
from the Air's measurements by the GPU core count (KYOJIN_GPU_CORES overrides the count): `nax_ks_tgs` threadgroup count from which the NAX verify matmul is not
k-split (Air 128 = 12.8 per core), `nax_ks_few` below which it splits 4 ways (32), `nax_ks_weights` smallest linear (weights) that takes a split (16M: below it the
extra launch costs more than it saves; it grows with the memory bandwidth, i.e. the core count), `mma_max` rows served by the MMA GEMV (8: one fragment tile), `dense_fused_max` (only in a measured chip entry; absent = exl3_linear.DENSE_FUSED_MAX) rows up to which a dense prefill linear runs the fused NAX matmul (Apple M5 Air: 512; above it decode + MLX matmul wins, +4-10 % whole model at 768-1024 rows on 0.8B, 4B, 9B).
`sdpa_tpc` / `sdpa_min_n` / `sdpa_min_n1` are the split-K decode attention's (sdpa_dec.py): threadgroups per core, {head dim: smallest key count} for verify rows (or one int), and
{head dim: smallest key count} for one row.
`step_regions_wide` (0/1, default 0; KYOJIN_STEP_REGIONS_WIDE wins): the compiled decode-step regions (step_regions.py) also on the served solo step and on window /
TurboQuant caches (Air M5 4B 8K / 32K: served solo plain step 23.35 / 29.64 -> 22.56 / 28.84 ms fp16, 22.94 / 25.52 -> 22.25 / 24.88 TQ6; KL vs off 0.5-1.3e-4 on 4B / 9B,
below the fp16-vs-TQ6 floor 1.7-1.9e-4, top-1 100 %, tools/sr107.py SR107_CTX=2048; a solo session then no longer matches the same session in a 2-user batch).
An entry measured on the device by `qualify --stage autotune` always wins over these."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

VERSION = 1
_FILES = ("metal_src.py", "metal_nax.py", "kernels.py", "exl3_linear.py", "tuning.py", "chip_defaults.json")      # a change of these invalidates a stored table
TABLE: dict = {}                 # "<in>x<out>x<K>" -> entry
INFO: dict = {"pack": None, "file": None, "status": "none"}        # what install() found: none | loaded | stale | off


def device_id() -> str:
    import mlx.core as mx
    d = mx.device_info()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{d.get('device_name', 'gpu')}-{d.get('architecture', '')}".strip("-"))


AIR_CORES = 10            # the GPU the constants were measured on (Apple M5, 10 cores)
_chip: dict | None = None


def gpu_cores() -> int:
    """GPU core count: KYOJIN_GPU_CORES, else ioreg's gpu-core-count, else the Air's."""
    v = os.environ.get("KYOJIN_GPU_CORES")
    if v:
        return max(1, int(v))
    try:
        import subprocess
        out = subprocess.run(["ioreg", "-rc", "AGXAccelerator"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return AIR_CORES


FILL = os.environ.get("KYOJIN_FILL", "1") != "0"          # device-aware k-split of the decode GEMV (kernels.pick_tgk): 0 = the 512-threadgroup / 8-tile constants measured on the 10-core Air
FILL_PER_CORE = int(os.environ.get("KYOJIN_FILL_PER_CORE", "8"))       # threadgroups per GPU core the k-split fills to
FILL_MIN_TILES = os.environ.get("KYOJIN_FILL_MIN_TILES")               # None = derived from the core count (see fill_min_tiles)


def fill_tgs(off: int = 512) -> int:
    """Threadgroups a 1-row decode GEMV k-splits to: FILL_PER_CORE per GPU core, never below the 512 the constants were measured with on the 10-core
    Air (`off` when the fill is disabled). The GEMV's bandwidth follows the threadgroup count on a wide GPU: measured there (64 cores, 4-bit,
    dependent chain) 32 threadgroups 126 GB/s, 128 345, 512 897 of a ~1000 GB/s roof."""
    if not FILL:
        return off
    return max(off, FILL_PER_CORE * gpu_cores())


def fill_min_tiles(off: int = 8) -> int:
    """16-element k tiles a warp keeps when the k-split fills: 8 on the Air's 10 cores, and it falls with the core count because the same k depth has to
    feed 5-8x more threadgroups (2 from 64 cores up; never 1, a warp with one tile cannot hide its first load)."""
    if not FILL:
        return off
    if FILL_MIN_TILES is not None:
        return max(1, int(FILL_MIN_TILES))
    c = gpu_cores()
    return off if c <= AIR_CORES else max(2, round(off * AIR_CORES / c))


def chip() -> dict:
    """Chip-keyed defaults of the verify-band thresholds (see the module doc): chip_defaults.json[device name] over the Air's values scaled by core count."""
    global _chip
    if _chip is None:
        import mlx.core as mx
        c = gpu_cores()
        r = c / AIR_CORES
        d = dict(mma_max=8, nax_ks_tgs=round(128 * r), nax_ks_few=round(32 * r), nax_ks_weights=int(16_000_000 * r), cores=c)
        f = Path(__file__).parent / "chip_defaults.json"
        if f.exists():
            ent = json.loads(f.read_text()).get(str(mx.device_info().get("device_name", "")), {})
            if ent.get("cores", c) == c:                                    # a measured entry holds for its own core count only
                d.update(ent)
        _chip = d
    return _chip


def kernel_version() -> str:
    h = hashlib.sha1(str(VERSION).encode())
    for f in _FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def key(kin: int, nout: int, K: int) -> str:
    return f"{kin}x{nout}x{K}"


def path_for(pack, dev: str | None = None) -> Path:
    return Path(pack) / "tuning" / f"{dev or device_id()}.json"


def install(pack) -> str:
    """Load <pack>/tuning/<device>.json into TABLE (replacing the previous table). Returns the status: loaded | none | stale | off."""
    TABLE.clear()
    INFO.update(pack=str(pack), file=None, status="none")
    if os.environ.get("KYOJIN_TUNING", "1") == "0":
        INFO["status"] = "off"
        return "off"
    f = path_for(pack)
    if not f.exists():
        return "none"
    d = json.loads(f.read_text())
    INFO["file"] = str(f)
    if d.get("version") != VERSION or d.get("kernel_version") != kernel_version():
        INFO["status"] = "stale"
        return "stale"
    TABLE.update(d["shapes"])
    INFO["status"] = "loaded"
    return "loaded"


def lookup(kin: int, nout: int, K: int) -> dict:
    return TABLE.get(key(kin, nout, K), {})


def pf(name: str, default):
    """Setting `name` of the table entry "prefill" (see the module doc), else `default`."""
    return TABLE.get("prefill", {}).get(name, default)


def serve(name: str, default):
    """Setting `name` of the table entry "serve" (server depth-policy priors), else `default`."""
    return TABLE.get("serve", {}).get(name, default)


def set_entry(pack, name: str, entry: dict) -> Path:
    """Merge `entry` into the non-shape entry `name` of <pack>/tuning/<device>.json, keeping every other entry (a new or stale file starts empty,
    except the MTP round cost table "spec" (write_spec), which does not depend on the kernel version)."""
    f = path_for(pack)
    d = json.loads(f.read_text()) if f.exists() else {}
    if d.get("version") != VERSION or d.get("kernel_version") != kernel_version():
        d = {"spec": d["spec"]} if d.get("spec") else {}
    shapes = d.pop("shapes", {})
    shapes[name] = {**shapes.get(name, {}), **entry}
    meta = {k: v for k, v in d.items() if k not in ("version", "kernel_version", "device")}
    return write(pack, shapes, meta)


def cfg_for(entry: dict, rows: int):
    """Tuned NAX tile for `rows` (tuple) or None."""
    t = entry.get("nax_cfg")
    if not t:
        return None
    ks = sorted(int(k) for k in t)
    k = next((k for k in ks if k >= rows), ks[-1])
    return tuple(t[str(k)])


def ks_for(entry: dict, rows: int):
    """Tuned k-split of the fused NAX matmul for `rows` (int) or None."""
    t = entry.get("nax_ks")
    if not t:
        return None
    ks = sorted(int(k) for k in t)
    return int(t[str(next((k for k in ks if k >= rows), ks[-1]))])


def write(pack, shapes: dict, meta: dict) -> Path:
    f = path_for(pack)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(version=VERSION, kernel_version=kernel_version(), device=device_id(), shapes=shapes, **meta), indent=1, sort_keys=True))
    tmp.replace(f)
    return f


def write_spec(pack, mode: str, ctx: int, entry: dict) -> Path:
    """Store the per-round cost table of the served MTP path (tools/cost103.py) under "spec" -> mode (fp16 | tq) -> ctx of <pack>/tuning/<device>.json,
    keeping the rest of the file (a new file gets an empty shape table). KYOJIN_SPEC_COSTS=FILE writes there instead."""
    env = os.environ.get("KYOJIN_SPEC_COSTS")
    f = Path(env) if env and env not in ("0", "1") else path_for(pack)
    d = json.loads(f.read_text()) if f.exists() else dict(version=VERSION, kernel_version=kernel_version(), device=device_id(), shapes={})
    d.setdefault("spec", {}).setdefault(mode, {})[str(int(ctx))] = entry
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    tmp.replace(f)
    return f


def spec_costs(pack, mode: str = "fp16") -> dict | None:
    """The per-round cost table of the served MTP path on this device: {ctx: [ms of a round at depth 0, 1, ...]} for KV `mode`, read from
    <pack>/tuning/<device>.json "spec" (or the file in KYOJIN_SPEC_COSTS); the fp16 table stands in for a mode without one. None = no table (the
    depth policy keeps its constant costs). The table only steers the depth: the greedy tokens do not depend on it, so a stale kernel version
    does not void it. Opt-in (Air M5 A/B: ahead at 4K, mixed per text at 32K): KYOJIN_SPEC_COSTS=1 (the pack table) or =FILE; unset, 0 or
    KYOJIN_TUNING=0: None."""
    env = os.environ.get("KYOJIN_SPEC_COSTS", "0")
    if env == "0" or os.environ.get("KYOJIN_TUNING", "1") == "0":
        return None
    f = path_for(pack) if env == "1" else Path(env)
    if not f.exists():
        return None
    sp = json.loads(f.read_text()).get("spec") or {}
    t = sp.get(mode) or sp.get("fp16")
    if not t:
        return None
    out = {}
    for c, e in t.items():
        if e.get("round"):
            r = [float(x) for x in e["round"]]
            out[int(c)] = [max(r[:i + 1]) for i in range(len(r))]       # a deeper round never costs less (the medians carry thermal noise)
    return out or None

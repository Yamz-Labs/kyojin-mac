"""The planner: pure functions, no GPU. build_plan(meta, backend, args, state) -> ordered Stage list.

Stages are resumable: <out>/<stage>.json with {"status": "done", "key": fingerprint, "wall_s": ..., "data": {...}}.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# stage -> (job, needs, gpu). Same table as the CUDA/ROCm version. Estimates (seconds) are per scale: `air` = measured on the 16 GB Air with the 0.8B pack
# (macair45 dry run, see REPORT.md), `ultra` = estimate for the DS4.1 pack (model loads of minutes dominate: 1 per max-ctx probe + 1 for serve).
STAGE_TABLE = {
    "fetch":   dict(job="cpu",    needs=[],                         gpu=False),
    "meta":    dict(job="cpu",    needs=["fetch"],                  gpu=False),
    "autotune": dict(job="autotune", needs=["meta"],               gpu=True),
    "parity":  dict(job="parity", needs=["meta"],                   gpu=True),
    "kv":      dict(job="kv",     needs=["meta"],                   gpu=True),
    "maxctx":  dict(job="maxctx", needs=["meta", "kv"],             gpu=True),
    "serve":   dict(job="serve",  needs=["meta", "kv", "maxctx"],   gpu=True),
    "smoke":   dict(job="serve",  needs=["serve"],                  gpu=True),
    "report":  dict(job="cpu",    needs=["meta"],                   gpu=False),
}
EST = {
    "air":   dict(fetch=2, meta=2, autotune=150, parity=90, kv=120, maxctx=240, serve=400, smoke=200, report=1),
    "ultra": dict(fetch=120, meta=5, autotune=600, parity=1800, kv=1800, maxctx=1800, serve=1500, smoke=900, report=2),
}
ORDER = list(STAGE_TABLE)
# generic = same code for every arch; per-arch = needs a family adapter (pack.FAMILIES row, refs/<adapter>.py, KV mode). Shown in the report.
KIND = dict(fetch="generic", meta="generic (FAMILIES row per arch)", autotune="generic (dense EXL3 shapes read from the pack header)", parity="per-arch (refs/<family>.py)", kv="per-arch (kvwin for the hybrid; native caches skip)",
            maxctx="generic (KV formula per arch, else measured)", serve="generic (decode_curve tool for ds41)", smoke="generic", report="generic")


@dataclass
class Stage:
    name: str
    job: str
    needs: list
    est_s: int
    gpu: bool
    status: str = "todo"        # todo | done | skip (resumed) | blocked
    note: str = ""
    cmd: list = field(default_factory=list)


RELEVANT = {"fetch": ("ref",), "meta": (), "autotune": ("autotune_rounds", "kernel_version"), "parity": ("ref", "doc_tokens"), "kv": ("kv_kl_max", "kv_bits", "kv_n", "kv_eval"), "maxctx": ("sessions", "ctx_cap", "mem_cap_gib"),
            "serve": ("sessions", "ctx_cap", "scale"), "smoke": ("sessions", "scale"), "report": ()}


def fingerprint(stage: str, meta: dict, args: dict) -> str:
    """Input key of a stage: pack index sha, backend + the arguments that change THIS stage's result. A done-file with another key is stale."""
    rel = {k: args.get(k) for k in ("backend",) + RELEVANT[stage]}
    return hashlib.sha1(json.dumps([stage, meta.get("index_sha"), rel], sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_state(out: Path) -> dict:
    st = {}
    for s in ORDER:
        f = out / f"{s}.json"
        if f.exists():
            try:
                st[s] = json.loads(f.read_text())
            except ValueError:
                pass
    return st


def save_stage(out: Path, name: str, key: str, status: str, wall_s: float, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(dict(status=status, key=key, wall_s=round(wall_s, 1), data=data, at=time.strftime("%F %T")), indent=1, default=str))


def build_plan(meta: dict, args: dict, state: dict, only: list | None = None, redo: list | None = None, force_deps: bool = False) -> list[Stage]:
    """Plan = all stages in order, marked done (valid done-file, same key), todo, or blocked (a need is neither done nor selected)."""
    redo = set(redo or [])
    plan: list[Stage] = []
    for name in ORDER:
        t = STAGE_TABLE[name]
        st = Stage(name, t["job"], list(t["needs"]), EST[args.get("scale", "ultra")][name], t["gpu"])
        cap = args.get("mem_cap_gib") or 13.0
        if name == "parity" and not args.get("ref"):
            st.status, st.note = "skip", "no --ref BF16 dir: parity unmeasured"
        elif name == "parity" and not meta.get("ref_adapter"):
            st.status, st.note = "skip", f"no reference adapter for family {meta.get('family')} (per-arch: DS4.1 is checked against the box reference, DAY1_RUNBOOK 4): parity unmeasured"
        elif name == "parity" and meta.get("ref_bytes", 0) > (cap - 2.5) * 2**30:
            st.status, st.note = "skip", f"BF16 reference {meta['ref_bytes']/2**30:.1f} GiB does not fit under the {cap:.0f} GiB footprint cap (needs the streamed adapter): parity unmeasured"
        elif name == "kv" and meta.get("kv_mode") != "kvwin":
            st.status, st.note = "skip", f"{meta.get('family')} has a native cache (no KV bits to choose)"
        d = state.get(name)
        if st.status == "todo" and d and d.get("status") == "done" and d.get("key") == fingerprint(name, meta, args) and name not in redo:
            st.status, st.note = "done", f"resumed ({d.get('wall_s')} s)"
        if name == "meta" and meta.get("problems"):
            st.note = "; ".join(meta["problems"])
        plan.append(st)
    if only:
        sel = set(only)
        by = {s.name: s for s in plan}
        for st in plan:
            if st.name not in sel and st.status == "todo":
                st.status, st.note = "unselected", "not selected"
        for st in plan:
            if st.name not in sel or st.status != "todo":
                continue
            for n in st.needs:
                nd = by[n]
                if nd.status == "unselected":
                    if force_deps:
                        nd.status, nd.note = "todo", "dependency forced"
                    else:
                        st.status, st.note = "blocked", f"needs {n} (not done; add it to --stage or use --force-deps)"
                        break
    return plan


def estimate(plan: list[Stage]) -> dict:
    tot = sum(s.est_s for s in plan if s.status == "todo")
    gpu = sum(s.est_s for s in plan if s.status == "todo" and s.gpu)
    return dict(total_s=tot, gpu_s=gpu)


def render_plan(plan: list[Stage], meta: dict, serve_cmd: list, args: dict, env: dict | None = None) -> str:
    L = [f"pack {meta['name']}: {meta['arch']} ({meta['family']}), {meta['layers']} layers ({meta['kv_layers']} with KV), {meta['kv_heads']} KV heads x {meta['head_dim']}, "
         f"{meta['quant']} {meta['bpw']} bpw, weights {meta['weights_bytes']/2**30:.1f} GiB, mtp layers {meta['mtp_layers']}, vision {meta['vision']}, KV mode {meta['kv_mode']}"]
    L.append(f"backend {args['backend']} (scale {args.get('scale')}), out {args['out']}")
    for p in meta["problems"]:
        L.append(f"  PROBLEM: {p}")
    L.append("stage    job     gpu  status   est     note")
    for s in plan:
        L.append(f"{s.name:<8} {s.job:<7} {'yes' if s.gpu else 'no ':<4} {s.status:<8} {s.est_s//60:>3} min  {s.note}")
    e = estimate(plan)
    L.append(f"remaining estimate: {e['total_s']//60} min total, {e['gpu_s']//60} min GPU ({args.get('scale')} constants, not a promise)")
    L.append("serve command (ctx and KV are replaced by the maxctx and kv stages): " + " ".join([f"{k}={v}" for k, v in (env or {}).items()] + serve_cmd))
    return "\n".join(L)

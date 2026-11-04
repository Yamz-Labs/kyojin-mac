#!/usr/bin/env python3
"""qualify --backend mlx: one command. A converted Kyojin-MLX pack in, parity/KL + KV choice + max ctx + served matrix + greedy sha + real-use smoke out.
Same stage names, resumability and qualify.json schema as the cuda / rocm version (CUDA/ROCm tools/qualify). See DESIGN.md. Examples:
  tools/qualify.sh --pack ~/kyojin-mlx/models/q35-2b-exl3-2.10 --dry-run                       (Air: --scale air is the default under 64 GB)
  tools/qualify.sh --pack ~/ds41/model                                                        (Ultra, DAY1_RUNBOOK 0b: the whole thing)
  tools/qualify.sh --pack P --ref BF16_DIR --stage fetch,meta,parity
Resumable: a stage whose <out>/<stage>.json is done with the same input key is skipped (--redo STAGE to rerun). Every GPU job holds the Mac GPU lock."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends  # noqa: E402
import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import report  # noqa: E402
import stages as st  # noqa: E402


# stage sizes per scale. `cfg` = serve / smoke sizes (tokens); the rest are defaults of the arguments above.
SCALES = {
    "air": dict(sessions="1,3", doc_tokens=512, kv_n=2048, kv_eval=128, ctx_cap=16384, mem_cap_gib=13.0,
                cfg=dict(prefill_tokens=4096, agg_tokens=1024, realuse=dict(prefix=2048, task=300, decode=400, sub=48, turns=2))),
    "ultra": dict(sessions="1,4", doc_tokens=2048, kv_n=32768, kv_eval=512, ctx_cap=262144, mem_cap_gib=200.0,
                  cfg=dict(prefill_tokens=8192, agg_tokens=4096, realuse=dict(prefix=16384, task=2000, decode=800, sub=200, turns=3))),
}


def kernel_version(repo) -> str:
    """Stamp of the kernel sources a tuning table is valid for (kyojin_mlx/tuning.py, loaded by path: no mlx import here)."""
    import importlib.util
    f = Path(repo) / "kyojin_mlx" / "tuning.py"
    spec = importlib.util.spec_from_file_location("_tuning_stamp", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kernel_version()


def parse(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mlx", choices=sorted(backends.PROFILES))
    ap.add_argument("--scale", choices=sorted(planmod.EST), help="stage sizes: air = scaled down for the 16 GB Air, ultra = the full sizes (default: by installed memory, < 64 GB = air)")
    ap.add_argument("--pack", required=True, help="converted pack directory")
    ap.add_argument("--ref", help="BF16 reference model directory (for parity + KV KL; kept until the gate passes)")
    ap.add_argument("--out", help="state/output dir (default: <repo>/qualify_out/<pack>/<backend>)")
    ap.add_argument("--stage", help="comma list of stages to run (default: all). Stages: " + ",".join(planmod.ORDER))
    ap.add_argument("--redo", help="comma list of stages to clear and rerun")
    ap.add_argument("--force-deps", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, derived serve command and estimates; touch no GPU")
    ap.add_argument("--sessions", help="session counts for the max-ctx search; the largest is the served config (N=1..max matrix). Default 1,4 (air: 1,3)")
    ap.add_argument("--kv-bits", default="8,6,4", help="symmetric KV bits measured by the kv stage (8 is the baseline)")
    ap.add_argument("--doc-tokens", type=int, help="tokens per parity doc (air 512, ultra 2048)")
    ap.add_argument("--kv-n", type=int, help="context of the KV stage's long doc (air 2048, ultra 32768)")
    ap.add_argument("--kv-eval", type=int, help="scored positions at the end of the long doc (air 128, ultra 512)")
    ap.add_argument("--mem-cap-gib", type=float, help="footprint cap of one job (default 13 GiB = the macguard rule on the Air; set it for the Ultra)")
    ap.add_argument("--autotune-rounds", type=int, default=5, help="interleaved timing rounds per candidate of the autotune stage")
    ap.add_argument("--kv-kl-max", type=float, default=0.02)
    ap.add_argument("--ctx-cap", type=int, help="cap of the max-ctx search (default: the model's max_position_embeddings)")
    ap.add_argument("--disk-floor-gb", type=float, default=40)
    ap.add_argument("--hf", help="HF repo to download into --pack first (license/size/disk-floor checked)")
    ap.add_argument("--conv-log", help="converter report (per-layer proxy error); default <pack>/conv_report.json")
    ap.add_argument("--port", type=int, default=8590)
    ap.add_argument("--repo", default=str(HERE.parents[1]), help="engine checkout (kyojin_mlx + tools): PYTHONPATH and cwd of the servers (default: this checkout)")
    ap.add_argument("--gates", type=json.loads, default=None, help='JSON override of verdict gates, e.g. \'{"kl_mean": 0.1}\'')
    a = vars(ap.parse_args(argv))
    a["kernel_version"] = kernel_version(a["repo"])
    a["scale"] = a["scale"] or ("air" if backends.total_mem_gib() < 64 else "ultra")
    for k, v in SCALES[a["scale"]].items():
        if k != "cfg" and a.get(k) is None:
            a[k] = v
    a["scale_cfg"] = SCALES[a["scale"]]["cfg"]
    a["gates"] = {"ctx_min": min(32768, a["ctx_cap"]), **(a["gates"] or {})}      # air: the max-ctx gate follows the scaled-down cap
    a["sessions"] = sorted({int(x) for x in a["sessions"].split(",")})
    a["kv_bits"] = [int(x) for x in a["kv_bits"].split(",")]
    a["pack"] = str(Path(a["pack"]).expanduser())
    if a["ref"]:
        a["ref"] = str(Path(a["ref"]).expanduser())
    a["out"] = str(Path(a["out"]).expanduser() if a["out"] else Path(a["repo"]) / "qualify_out" / Path(a["pack"]).name / a["backend"])
    return a


def logger(out: Path):
    def log(msg):
        line = f"{time.strftime('%F %T')} qualify {msg}"
        print(line, flush=True)
        with open(out / "qualify.log", "a") as f:
            f.write(line + "\n")
    return log


def main(argv=None) -> int:
    a = parse(argv)
    out = Path(a["out"]); out.mkdir(parents=True, exist_ok=True)
    meta = packmod.read_pack(a["pack"])
    if a["ref"]:
        meta["ref_bytes"] = packmod.weights_dir_bytes(Path(a["ref"]))
    state = planmod.load_state(out)
    only = a["stage"].split(",") if a["stage"] else None
    for s in (only or []) + (a["redo"].split(",") if a["redo"] else []):
        if s not in planmod.ORDER:
            sys.exit(f"unknown stage {s!r}")
    plan = planmod.build_plan(meta, a, state, only=only, redo=a["redo"].split(",") if a["redo"] else None, force_deps=a["force_deps"])
    prof = backends.profile(a["backend"], a["repo"])
    cmd = [prof["python"], *packmod.serve_flags(meta, port=a["port"], ctx=0, sessions=max(a["sessions"]), k_bits=6, v_bits=6, chunk=st.chunk_for(meta))]
    if a["dry_run"]:
        print(planmod.render_plan(plan, meta, cmd, a, packmod.serve_env(meta, *((16, 16) if meta["kv_mode"] == "native" else (6, 6)))))
        return 0
    log = logger(out)
    c = st.Ctx(meta, a, out, Path(a["repo"]), prof, log)
    c.data = {n: s["data"] for n, s in state.items() if s.get("status") == "done"}
    for s in plan:
        if s.status == "blocked":
            sys.exit(f"stage {s.name} blocked: {s.note}")
    todo = [s for s in plan if s.status == "todo"]
    results = {n: s for n, s in state.items()}
    done_jobs = set()
    t_all = time.time()
    for s in todo:
        if s.name in ("serve", "smoke"):
            if "serve" in done_jobs:
                continue
            done_jobs.add("serve")
            names = [x.name for x in todo if x.name in ("serve", "smoke")]
            runner = lambda: _split_serve(c, "serve" in names, "smoke" in names)
        else:
            runner = {"fetch": lambda: st.stage_fetch(c), "meta": lambda: st.stage_meta(c), "autotune": lambda: st.job_autotune(c), "parity": lambda: st.job_parity(c),
                      "kv": lambda: st.job_kv(c), "maxctx": lambda: st.job_maxctx(c), "report": None}[s.name]
        if s.name == "report":
            continue
        t0 = time.time(); log(f"stage {s.name} start")
        try:
            data = runner()
        except Exception as e:
            log(f"stage {s.name} FAILED after {time.time()-t0:.0f} s: {e}\n{traceback.format_exc()[-800:]}")
            planmod.save_stage(out, s.name, planmod.fingerprint(s.name, meta, a), "failed", time.time() - t0, dict(error=str(e)))
            return 1
        wall = time.time() - t0
        for name, d in (data.items() if s.name in ("serve", "smoke") else [(s.name, data)]):
            planmod.save_stage(out, name, planmod.fingerprint(name, meta, a), "done", wall, d)
            c.data[name] = d
            results[name] = dict(status="done", wall_s=round(wall, 1), data=d)
        log(f"stage {s.name} done in {wall:.0f} s")
    if True:     # the report is cheap: always rebuilt from the stage files
        results = planmod.load_state(out)
        j = report.assemble(meta, a, results, Path(a["repo"]))
        report.write(out, j)
        planmod.save_stage(out, "report", planmod.fingerprint("report", meta, a), "done", 0, dict(verdict=j["verdict"]))
        log(f"report written ({out}/REPORT.md); verdict pass={j['verdict']['pass']} reasons={j['verdict']['reasons']} unmeasured={j['verdict']['unmeasured']}; total {time.time()-t_all:.0f} s")
    return 0


def _split_serve(c, do_serve, do_smoke):
    r = st.job_serve(c, do_serve, do_smoke)
    out = {}
    if do_serve:
        out["serve"] = {**r["serve"], "cmd": r["cmd"], "env": r["env"]}
    if do_smoke:
        out["smoke"] = {**(r["smoke"] or {}), "cmd": r["cmd"], "env": r["env"]}
    return out


if __name__ == "__main__":
    sys.exit(main())

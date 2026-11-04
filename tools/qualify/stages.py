"""Stage executors (MLX). Each takes the Ctx and returns the stage's data dict (saved by the driver into <out>/<stage>.json).
Same stage names and data shapes as the CUDA/ROCm version; GPU stages run inside job functions that hold the Mac GPU lock (one heavy job at a time).
Generic = the code here; per-arch = the pack.FAMILIES row + refs/<adapter>.py + the KV mode (see DESIGN.md)."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import backends
import load
import pack as packmod
import rules
import search
import smoke
from locks import gpu_locks, guard_tail
from server import guard_kill_for, oom_in, running_server


@dataclass
class Ctx:
    meta: dict
    args: dict
    out: Path
    repo: Path
    prof: dict
    log: callable = print
    data: dict = field(default_factory=dict)       # stage name -> data (done in this run or resumed)

    def env(self, extra: dict | None = None) -> dict:
        return {**os.environ, **self.prof["env"], **(extra or {})}

    def py(self, script: str, *a, env_extra: dict | None = None, timeout: int = 7200) -> str:
        r = subprocess.run([self.prof["python"], "-u", str(Path(__file__).parent / script), *map(str, a)], cwd=self.repo, env=self.env(env_extra),
                           capture_output=True, text=True, timeout=timeout)
        (self.out / f"{Path(script).stem}.last.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{script} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        return r.stdout

    def need_ram(self, need_gib: float | None = None):
        a = backends.mem_available_gib()
        need = need_gib if need_gib is not None else self.meta["weights_bytes"] / 2**30 * 1.2 + 2.0
        if a < need:
            raise RuntimeError(f"free memory {a:.1f} GiB < {need:.1f} GiB needed: not starting a GPU job (guard: {guard_tail(self.prof['guard_log'], 1)})")

    def locks(self):
        return gpu_locks(self.prof["locks"], self.log)

    @property
    def cap(self) -> float:
        return float(self.args.get("mem_cap_gib") or self.prof["mem_cap_gib"])


# ---------------------------------------------------------------- cpu stages
def stage_fetch(c: Ctx) -> dict:
    a = c.args
    d = dict(pack=c.meta["path"], shard_problems=packmod.verify_shards(c.meta["path"]), license=c.meta["license"],
             disk_free_gb=round(packmod.disk_free_gb(c.meta["path"]), 1), disk_floor_gb=a["disk_floor_gb"])
    if a.get("hf"):
        d["hf"] = fetch_hf(a["hf"], c.meta["path"], a["disk_floor_gb"])
    if a.get("ref"):
        d["ref_problems"] = packmod.verify_shards(a["ref"])
    d["ok"] = not d["shard_problems"] and not d.get("ref_problems") and d["disk_free_gb"] >= a["disk_floor_gb"]
    if not d["ok"]:
        raise RuntimeError("fetch/verify failed: " + json.dumps({k: d[k] for k in ("shard_problems", "ref_problems", "disk_free_gb") if k in d}))
    return d


def fetch_hf(repo: str, dest: str, floor_gb: float) -> dict:
    """Download under the disk floor (license + size + floor checked first)."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings) / 1e9
    lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None)
    free = packmod.disk_free_gb(dest)
    if free - size < floor_gb:
        raise RuntimeError(f"{repo}: {size:.1f} GB would leave {free-size:.1f} GB free < floor {floor_gb} GB")
    snapshot_download(repo, local_dir=dest)
    return dict(repo=repo, size_gb=round(size, 1), license=lic)


def chunk_for(meta: dict) -> int:
    """Prefill chunk the server is run with (DAY1_RUNBOOK step 10: 2048 for DS4.1 so CED applies; 512 for the small hybrid packs)."""
    from kyojin_mlx import modeldesc
    row = modeldesc.by_architecture(meta["arch"])
    return row.chunk if row is not None else 512


def serve_cmd_for(c: Ctx, ctx: int, sessions: int) -> tuple[list, dict]:
    k, v = chosen_kv(c)
    argv = packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=sessions, k_bits=k, v_bits=v, chunk=chunk_for(c.meta))
    return argv, packmod.serve_env(c.meta, k, v)


def stage_meta(c: Ctx) -> dict:
    m = dict(c.meta)
    argv, env = serve_cmd_for(c, 0, 0)
    m["serve_cmd_template"] = [c.prof["python"], *argv]
    m["draft_policy"] = packmod.draft_policy(m)
    m["machine"] = backends.machine_info()
    if m["problems"]:
        raise RuntimeError("pack problems: " + "; ".join(m["problems"]))
    return m


# ---------------------------------------------------------------- autotune
def job_autotune(c: Ctx) -> dict:
    """Time the kernel candidates of every dense EXL3 linear shape of the pack on THIS device, write <pack>/tuning/<device>.json (kyojin_mlx/tuning.py); the
    serve, the evals and every later stage load it with the pack (no file / other device / other kernel version = the constants). Shapes come from the pack
    header and the weights are random, so the pack is never loaded (footprint < 1 GiB whatever the model)."""
    c.need_ram(2.0)
    raw = c.out / "autotune_raw.json"
    with c.locks():
        c.py("autotune_mlx.py", "tune", c.meta["path"], raw, "--rounds", c.args.get("autotune_rounds", 5))
    r = json.loads(raw.read_text())
    sh = r["shapes"]
    tuned = {k: v["entry"] for k, v in sh.items() if v["changes"]}
    gains = []
    for k, v in sh.items():
        d = v["decode"]
        gains.append(dict(shape=k, count=v["count"], changes=v["changes"], decode_ms_default=d["ms"].get(str(d["default"])), decode_ms_tuned=d["ms"].get(str(d["pick"])), tgk=[d["default"], d["pick"]],
                          prefill={R: [p["ms"].get(p["default"]), p["ms"].get(p["pick"]), p["default"], p["pick"]] for R, p in v["prefill"].items() if p["pick"] != p["default"]}))
    return dict(file=r["file"], device=r["device"], kernel_version=r["kernel_version"], shapes=len(sh), tuned_shapes=len(tuned), wall_s=r["wall_s"], peak_gib=r["peak_gib"],
                rounds=r["rounds"], min_gain=r["min_gain"], entries=tuned, gains=gains)


# ---------------------------------------------------------------- parity / kv
def conv_summary(pack: Path, conv_log: str | None) -> dict:
    """Per-layer quantization error of the conversion: the MLX converter writes conv_report.json ([{layer, name, K, proxy, g}]) next to the pack
    (the CUDA/ROCm EXL3 log carries SQNR dB instead; same slot in the report: `sqnr`)."""
    f = Path(conv_log).expanduser() if conv_log else pack / "conv_report.json"
    if not f.exists():
        return dict(measured=False, note="no conversion report (--conv-log or <pack>/conv_report.json)")
    rows = json.loads(f.read_text())
    per_layer: dict = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r["proxy"])
    lm = {k: statistics.mean(v) for k, v in per_layer.items()}
    worst = max(lm, key=lm.get)
    return dict(measured=True, kind="proxy error (lower is better)", layers=len(lm), mean=round(statistics.mean(lm.values()), 5), max=round(lm[worst], 5), worst_layer=worst,
                bits_hist={str(k): sum(1 for r in rows if r["K"] == k) for k in sorted({r["K"] for r in rows})})


def aggregate_parity(res: dict) -> dict:
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    tot = sum(v["tokens"] for v in res.values())
    w = lambda k: sum(v[k] * v["tokens"] for v in res.values()) / tot    # noqa: E731
    return dict(kl_mean=w("kl"), top1_mean=w("top1"), dnll_mean=w("dnll"), by_domain=res)


def job_parity(c: Ctx) -> dict:
    c.need_ram(max(c.meta["weights_bytes"], c.meta.get("ref_bytes", 0)) / 2**30 * 1.2 + 2.0)
    d = c.out / "docs_short.json"
    if not d.exists():
        c.py("docs.py", "short", c.meta["path"], d, c.args["doc_tokens"])
    with c.locks():
        refdir = c.out / "ref_short"
        t0 = time.time()
        if not (refdir / "ref_wiki.npz").exists():
            c.py(f"refs/{c.meta['ref_adapter']}.py", c.args["ref"], d, refdir)
        c.log(f"parity: BF16 reference computed in {time.time()-t0:.0f} s")
        c.py("mlx_eval.py", "parity", c.meta["path"], d, refdir, c.out / "parity_raw.json")
    raw = json.loads((c.out / "parity_raw.json").read_text())
    res = aggregate_parity(raw)
    res["peak_gib"] = raw.get("_peak_gib")
    res["sqnr"] = conv_summary(Path(c.meta["path"]), c.args.get("conv_log"))
    return res


def job_kv(c: Ctx) -> dict:
    c.need_ram()
    d = c.out / "docs_long.json"
    if not d.exists():
        c.py("docs.py", "long", c.meta["path"], d, c.args["kv_n"])
    bits = c.args["kv_bits"]
    with c.locks():
        c.py("mlx_eval.py", "kv", c.meta["path"], d, c.out / "kv_raw.json", c.args["kv_eval"], *bits)
    raw = json.loads((c.out / "kv_raw.json").read_text())
    meta = raw.pop("_meta", {})
    kl8 = {int(b): v["kl_vs_k8"] for b, v in raw.items() if "kl_vs_k8" in v}
    return dict(table=raw, chosen=list(rules.choose_kv(kl8, c.args["kv_kl_max"])), kl_max=c.args["kv_kl_max"], n=meta.get("N"), n_eval=meta.get("n_eval"),
                peak_gib=meta.get("peak_gib"), rule="smallest symmetric bits >= Q6 with KL vs k8v8 <= kl_max, else k8v8 (KL vs the fp16 cache of the same pack)")


def chosen_kv(c: Ctx) -> tuple:
    kv = c.data.get("kv")
    if kv:
        return tuple(kv["chosen"])
    return (16, 16) if c.meta["kv_mode"] == "native" else (6, 6)     # native cache: no choice; no measurement: the owner's Q6 minimum


# ---------------------------------------------------------------- max ctx
def probe_factory(c: Ctx, sessions: int):
    cl = load.Client(c.args["port"])
    state = {"cal": False}
    cap = c.cap

    def probe(ctx: int):
        log = c.out / f"srv_maxctx_s{sessions}_c{ctx}.log"
        argv, env = serve_cmd_for(c, ctx, sessions)
        with running_server(c.prof, c.repo, argv, c.args["port"], log, env_extra=env, logf=c.log) as h:
            if not h.ok:
                g = guard_kill_for(c.prof["guard_log"], h.proc.pid)
                tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                return False, ("killed by macguard: " + g if g else "no start: " + tail[0][:160])
            if not state["cal"]:
                cl.calibrate(); state["cal"] = True
            for slot in (0, 1):                                  # 2 distinct near-full prompts back to back
                r = cl.near_full(ctx, slot)
                if r["err"] or oom_in(log):
                    return False, f"near-full prompt {slot} failed: {r['err']}"
            if sessions > 1:                                     # `sessions` concurrent shares, twice (drill 0: the 2nd round OOMed)
                for rnd in range(2):
                    per = int(ctx * 0.97 / sessions) - 64
                    rr = cl.concurrent(sessions, per, 32)
                    if rr["errors"] or oom_in(log):
                        return False, f"concurrent round {rnd} failed: {rr['errors'][:1]}"
            if h.over_cap(cap):
                return False, f"footprint {h.peak_gib:.1f} GiB > cap {cap:.1f} GiB"
            probe.peak[ctx] = round(h.peak_gib, 2)
            return (not oom_in(log)), f"ok, footprint {h.peak_gib:.2f} GiB"
    probe.peak = {}
    return probe


def job_maxctx(c: Ctx) -> dict:
    c.need_ram()
    out = {}
    k, v = chosen_kv(c)
    with c.locks():
        free = backends.gpu_free_mib(c.args["backend"])
        free = min(free, c.cap * 1024) if free else None
        cap_ctx = c.args.get("ctx_cap") or c.meta["max_pos"]
        for s in c.args["sessions"]:
            bpt = packmod.kv_bytes_per_token(c.meta, k, v)
            if bpt:
                free_b = (free * 2**20 if free else c.cap * 2**30) - c.meta["weights_bytes"]
                hi = search.upper_bound(free_b, bpt, 1.5 * 2**30, cap_ctx)
                c.log(f"maxctx sessions={s}: KV {bpt/1024:.1f} KB/token, search up to {hi}")
            else:                                                # native cache: no formula, the footprint decides
                hi = search.grid_floor(cap_ctx)
                c.log(f"maxctx sessions={s}: native cache (no KV formula), search up to {hi}")
            lo = min(hi, 8192)
            probe = probe_factory(c, s)
            r = search.bisect_max(lo, hi, probe)
            out[str(s)] = r["best"]
            out[f"tried_s{s}"] = r["tried"]
            out[f"footprint_gib_s{s}"] = probe.peak.get(r["best"])
    return out


# ---------------------------------------------------------------- serve + smoke (one server load)
def job_serve(c: Ctx, do_serve: bool, do_smoke: bool) -> dict:
    c.need_ram()
    mc = c.data.get("maxctx") or {}
    sess = max(c.args["sessions"])
    ctx = mc.get(str(sess)) or max((v for k, v in mc.items() if k.isdigit() and v), default=None)
    if not ctx:
        ctx = c.args.get("ctx_cap") or 16384
        c.log(f"serve: no max-ctx result, using {ctx}")
    sess_eff = sess if mc.get(str(sess)) else 1
    sc = c.args["scale_cfg"]
    out = dict(serve={}, smoke=None)
    with c.locks():
        log = c.out / "srv_serve.log"
        argv, env = serve_cmd_for(c, ctx, sess_eff)
        out["cmd"] = [c.prof["python"], *argv]
        out["env"] = env
        with running_server(c.prof, c.repo, argv, c.args["port"], log, env_extra=env, logf=c.log) as h:
            if not h.ok:
                g = guard_kill_for(c.prof["guard_log"], h.proc.pid)
                raise RuntimeError("serve: server did not become ready; see " + str(log) + (f" ({g})" if g else ""))
            cl = load.Client(c.args["port"], "qual")
            cl.calibrate()
            s = {}
            if do_serve:
                s["conformance"] = load.conformance(c.meta["path"])
                s["decode_short"] = [cl.stream(cl.prompt(300, i), 128) for i in range(3)]
                pf = min(sc["prefill_tokens"], ctx // 2)
                s["prefill_tokens"] = pf
                s["prefill_8k"] = [cl.stream(cl.prompt(pf, i + 3), 1) for i in range(3)]
                curve, t = [], 2048
                while t <= int(ctx * 0.9):
                    r = cl.stream(cl.prompt(t, 7), 64)
                    curve.append(dict(prompt=r["ptok"], ttft=r["ttft"], prefill_tps=round(r["ptok"] / r["ttft"], 1) if r["ptok"] and r["ttft"] else None, dec_tps=r["dec_tps"], err=r["err"]))
                    t *= 4
                s["curve"] = curve
                s["agg"] = [cl.concurrent(n, min(sc["agg_tokens"], ctx // max(4, sess_eff) - 256), 96) for n in range(1, sess_eff + 1)]
                a, b = cl.sha_set(), cl.sha_set()
                s["sha"], s["sha_self_equal"] = a, [x["sha"] for x in a] == [x["sha"] for x in b]
                s["oom"] = oom_in(log)
                for r in s["decode_short"] + s["prefill_8k"]:
                    for k in ("t0", "first", "end"):
                        r.pop(k, None)
                s["decode_tps_median"] = statistics.median(r["dec_tps"] for r in s["decode_short"] if r["dec_tps"])
                s["prefill_8k_tps"] = statistics.median(r["ptok"] / r["ttft"] for r in s["prefill_8k"] if r["ptok"] and r["ttft"])
                s["footprint_gib"] = round(h.peak_gib, 2)
                out["serve"] = s
            if do_smoke:
                subs = min(sess_eff - 1, 4)
                if subs >= 1:
                    sm = smoke.run_realuse(c.prof["python"], c.repo, {**c.prof["env"], **env}, c.args["port"], subs, sc["realuse"], c.meta["vocab"], c.out / "realuse_b.json")
                else:
                    r = cl.stream(cl.prompt(1000, 9), 64)
                    sm = dict(kind="realuse_b", total=1, passed=int(not r["err"] and bool(r["ntok"])), checks=[dict(name="completed", ok=not r["err"])], note="1 session: no sub-agents")
                out["smoke"] = sm
            h.peak_gib = max(h.peak_gib, backends.footprint_gib(h.proc.pid) or 0)
            if out["serve"]:
                out["serve"]["footprint_gib"] = round(h.peak_gib, 2)
    return out

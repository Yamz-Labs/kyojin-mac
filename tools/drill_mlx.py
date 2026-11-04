#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fire drill on the Mac in ONE command: HF fetch -> tokenizer-generic calibration -> EXL3 conversion on Metal with a bpw LADDER -> qualify --backend mlx (incl. autotune).

usage: tools/drill_mlx.py --hf Qwen/Qwen3.5-0.8B --work DIR [--k-attn 5 --k-mlp 4 --head-bits 6] [--cal CAL.safetensors] [--no-ladder]
Steps are resumable (done marker per step in WORK/drill.json); wall time per step is written there, lock waits apart from the work time. Heavy jobs hold ~/kyojin-mlx/.gpu.lock.

Default recipe for a small dense model: attention (full + GatedDeltaNet projections) K5, MLP K4, head K6 = ~4.1 bpw on the blocks (RECIPE_v1: 5-bit attention, 6-bit head;
~4 bpw dense; NOT 2). The ladder: convert, measure parity (qualify's gate: mean KL <= 0.15, top-1 >= 0.75 per domain); on failure raise +1 bit on the cells (block x
{attention, MLP}) with the highest measured local error per parameter (conv_sens.json: block output error with only that cell quantised, FP input) until +STEP_FRAC bpw
is added, re-convert ONLY the changed blocks (fp-prefix cache: unchanged blocks and the head are loaded), repeat until PASS, the size cap (--max-bpw) or a plateau
(KL gain < --plateau between two rungs). Each rung is logged (KL / top-1 / size / bits)."""
from __future__ import annotations

import argparse, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = os.path.expanduser("~/kyojin-mlx/.gpu.lock")
OPEN_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "llama3", "llama3.1", "llama3.2", "gemma", "openrail", "cc-by-4.0"}
HEAD_PROXY_MAX = 0.002          # Hessian-weighted relative error of the head above which the head gets +1 bit (K6 on Qwen3.5-0.8B: 1e-4)
WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model", "tokenizer*"]


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def fetch(repo: str, dest: Path, floor_gb: float, layers: int = 0) -> dict:
    """License, size and disk floor checked here, before a byte is downloaded. layers > 0: only the first N decoder blocks + embedding + head (hf_slice.py, HTTP range
    requests per tensor): for a checkpoint whose BF16 weights do not fit the machine (the slice is the model of the drill, its BF16 is the parity reference)."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, files_metadata=True)
    lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None)
    if lic not in OPEN_LICENSES:
        raise SystemExit(f"{repo}: license {lic!r} not in the open list {sorted(OPEN_LICENSES)}")
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith((".safetensors", ".json", ".txt", ".jinja", ".model"))) / 1e9
    free = free_gb(dest)
    if layers:
        sys.path.insert(0, str(REPO / "tools"))                                # python -I: the script directory is not on sys.path
        from hf_slice import slice_repo
        r = slice_repo(repo, dest, layers, info.sha)
        return dict(repo=repo, license=lic, size_gb=round(r["bytes"] / 1e9, 2), full_size_gb=round(size, 2), layers=layers, disk_free_before_gb=round(free, 1), revision=info.sha)
    if free - size < floor_gb:
        raise SystemExit(f"{repo}: {size:.1f} GB would leave {free - size:.1f} GB free < floor {floor_gb} GB")
    snapshot_download(repo, local_dir=str(dest), allow_patterns=WEIGHT_PATTERNS)
    return dict(repo=repo, license=lic, size_gb=round(size, 2), disk_free_before_gb=round(free, 1), revision=info.sha)


def mem_ok(min_free_pct: int = 40, wait_s: int = 600) -> int:
    t0 = time.time()
    while True:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        pct = int(out.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))
        if pct >= min_free_pct or time.time() - t0 > wait_s:
            return pct
        time.sleep(15)


def cells_of(r: dict) -> tuple:
    """Cells of one block: attention, MLP (dense MLP, or the shared expert of a MoE block) and, for a MoE block, its routed experts."""
    return ("attn", "mlp", "exp") if "params_exp" in r else ("attn", "mlp")


def cell_params(r: dict) -> int:
    return sum(r["params_" + g] for g in cells_of(r))


def next_cells(sens: list, k_max: int, step_frac: float, max_bpw: float):
    """Bit allocation for the next rung. sens = conv_sens.json rows {layer, err_attn, err_mlp, params_attn, params_mlp, K_attn, K_mlp}. Cells (block x {attn, mlp}) are ranked by
    local error per parameter (= error removed per added bit, the error falling ~4x per bit): attention cells (few parameters, large error per parameter) come first, then the
    MLP cells of the worst blocks. +1 bit goes to the top cells until step_frac of the block parameters are covered (= +step_frac bpw) or the max_bpw cap would be crossed.
    Returns (strategy {"L3.attn": K, ...} for ALL cells, raised [(layer, group, K_new)]) or (None, []) when nothing can be raised."""
    total = sum(cell_params(r) for r in sens)
    bits = sum(r["K_" + g] * r["params_" + g] for r in sens for g in cells_of(r))
    cells = [(r["err_" + g] / r["params_" + g], r["layer"], g, r["K_" + g], r["params_" + g]) for r in sens for g in cells_of(r) if r["K_" + g] < k_max]
    cells.sort(key=lambda c: -c[0])
    strat = {f"L{r['layer']}.{g}": r["K_" + g] for r in sens for g in cells_of(r)}
    added, raised = 0, []
    for score, li, g, k, p in cells:
        if added >= step_frac * total:
            break
        if (bits + p) / total > max_bpw:
            continue
        strat[f"L{li}.{g}"] = k + 1; bits += p; added += p; raised.append((li, g, k + 1))
    return (strat, raised) if raised else (None, [])


def body_bpw(sens: list) -> float:
    return sum(r["K_" + g] * r["params_" + g] for r in sens for g in cells_of(r)) / sum(cell_params(r) for r in sens)


def parity_gate(par: dict, kl_max: float = 0.15, top1_min: float = 0.75) -> list:
    """The parity part of qualify/rules.py verdict (same thresholds and reason texts); [] = pass."""
    reasons = [f"mean KL {par['kl_mean']:.3f} > {kl_max}"] if par["kl_mean"] > kl_max else []
    return reasons + [f"domain {d} top-1 {v['top1']:.3f} < {top1_min}" for d, v in par["by_domain"].items() if v["top1"] < top1_min]


def run(cmd: list, log: Path, lock: bool) -> int:
    if lock:  # the marker is written once the lock is held, so the queue wait can be told apart from the job time
        cmd = ["lockf", "-k", LOCK, "sh", "-c", 'date +%s > "$0"; exec "$@"', str(log) + ".locked", *cmd]
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO}/tools", PYTHONUNBUFFERED="1")
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cal", default=None, help="token-id calibration (safetensors input_ids); default: the raw text set given by --text, tokenized by the model's own tokenizer")
    ap.add_argument("--text", default=None, help="raw calibration text (jsonl, see tools/calib_text.py build); required unless --cal is given")
    ap.add_argument("--cal-rows", type=int, default=128); ap.add_argument("--cal-cols", type=int, default=1024)
    ap.add_argument("--k-attn", type=int, default=5); ap.add_argument("--k-mlp", type=int, default=4); ap.add_argument("--head-bits", type=int, default=6)
    ap.add_argument("--k-expert", type=int, default=3, help="MoE: routed experts (the shared expert uses --k-mlp); the ladder raises them as the third cell of a block")
    ap.add_argument("--slice", type=int, default=0, help="checkpoint too big for the machine: convert and qualify the first N decoder blocks only (hf_slice.py)")
    ap.add_argument("--no-ladder", action="store_true", help="one conversion at the given bits, no retry")
    ap.add_argument("--k-max", type=int, default=6, help="ladder: highest K per cell")
    ap.add_argument("--max-bpw", type=float, default=5.0, help="ladder size cap: mean bits per weight of the blocks")
    ap.add_argument("--step-frac", type=float, default=0.25, help="ladder: +1 bit on this fraction of the block parameters per rung (= +bpw per rung)")
    ap.add_argument("--max-rungs", type=int, default=6)
    ap.add_argument("--plateau", type=float, default=0.05, help="ladder: stop when a rung improves the mean KL by less than this fraction")
    ap.add_argument("--kl-max", type=float, default=0.15); ap.add_argument("--top1-min", type=float, default=0.75)
    ap.add_argument("--disk-floor-gb", type=float, default=40.0)
    ap.add_argument("--qualify-args", default="", help="extra args for qualify.sh")
    a = ap.parse_args()
    if not a.cal and not a.text:
        ap.error("give --text CALIB.jsonl (raw calibration text, not shipped) or --cal CAL.safetensors")
    work = Path(a.work).expanduser(); work.mkdir(parents=True, exist_ok=True)
    state_f = work / "drill.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else dict(model=a.hf, steps={})
    src, out = work / "src", work / "pack"
    py = sys.executable

    def step(name: str, fn):
        if st["steps"].get(name, {}).get("ok"):
            print(f"[{name}] done, skipped"); return
        t0 = time.time(); print(f"[{name}] start", flush=True)
        r = fn(); r = dict(r or {}); r["wall_s"] = round(time.time() - t0, 1); r["ok"] = r.get("ok", True)
        st["steps"][name] = r; state_f.write_text(json.dumps(st, indent=1))
        print(f"[{name}] {'ok' if r['ok'] else 'FAILED'} {r['wall_s']} s", flush=True)
        if not r["ok"]:
            sys.exit(1)

    step("fetch", lambda: fetch(a.hf, src, a.disk_floor_gb, a.slice))

    cal = Path(a.cal) if a.cal else work / "cal.safetensors"

    def calib():
        if a.cal:
            return dict(file=a.cal, generic=False)
        rc = run([py, str(REPO / "tools/calib_text.py"), "tokens", str(src), "--text", str(a.text), "--rows", str(a.cal_rows), "--cols", str(a.cal_cols), "--out", str(cal)], work / "calib.log", lock=False)
        return dict(ok=rc == 0 and cal.exists(), rc=rc, generic=True, rows=a.cal_rows, cols=a.cal_cols, info=(work / "calib.log").read_text().strip().splitlines()[-1][:300])
    step("calib", calib)

    def ladder():
        lf = work / "ladder.json"
        rungs = json.loads(lf.read_text()) if lf.exists() else []
        strategy, head_bits, stop = {}, a.head_bits, None
        (work / "rungs").mkdir(exist_ok=True)
        for r in range(a.max_rungs + 1):
            rdir = work / "rungs" / f"r{r}"; rdir.mkdir(exist_ok=True)
            if r < len(rungs):
                rec = rungs[r]
            else:
                pct = mem_ok()
                (rdir / "strategy.json").write_text(json.dumps(strategy))
                marker = rdir / "job.out.locked"; marker.unlink(missing_ok=True)
                cmd = [py, str(REPO / "tools/drill_rung.py"), "--src", str(src), "--pack", str(rdir / "pack"), "--cal", str(cal), "--cache", str(work / "cache"),
                       "--strategy", str(rdir / "strategy.json"), "--k-attn", str(a.k_attn), "--k-mlp", str(a.k_mlp), "--k-expert", str(a.k_expert), "--head-bits", str(head_bits),
                       "--docs", str(work / "docs_short.json"), "--ref", str(work / "ref_short"), "--result", str(rdir / "rung.json")]
                t0 = time.time()
                rc = run(cmd, rdir / "job.out", lock=True)
                if rc or not (rdir / "rung.json").exists():
                    return dict(ok=False, rc=rc, rungs=rungs, why=f"rung {r} failed, see {rdir}/job.out")
                job = json.loads((rdir / "rung.json").read_text())
                held = int(marker.read_text())
                sens = json.loads((rdir / "pack/conv_sens.json").read_text())
                rep = json.loads((rdir / "pack/conv_report.json").read_text())
                head = next(x for x in rep if x["layer"] == "head")
                par = job["parity"]
                rec = dict(rung=r, body_bpw=round(body_bpw(sens), 3), head_bits=head_bits, head_proxy=head["proxy"], pack_mb=round(os.path.getsize(rdir / "pack/model.safetensors") / 1e6, 1),
                           kl_mean=round(par["kl_mean"], 4), top1_mean=round(par["top1_mean"], 4), by_domain={d: dict(kl=round(v["kl"], 3), top1=round(v["top1"], 3)) for d, v in par["by_domain"].items()},
                           reasons=parity_gate(par, a.kl_max, a.top1_min), lock_wait_s=max(0.0, round(held - t0, 1)), work_s=round(time.time() - held, 1),
                           convert_s=job["convert_s"], ref_s=job.get("ref_s"), parity_s=job["parity_s"], conv_stages=job["conv_stages"], free_pct_before=pct,
                           cells={g: {str(k): sum(1 for x in sens if x.get("K_" + g) == k) for k in sorted({x["K_" + g] for x in sens if "K_" + g in x})} for g in cells_of(sens[0])})
                rungs.append(rec); lf.write_text(json.dumps(rungs, indent=1))
            print(f"[ladder] rung {r}: {rec['body_bpw']} bpw (head K{rec['head_bits']}) {rec['pack_mb']} MB  KL {rec['kl_mean']}  top-1 {rec['top1_mean']}  "
                  f"convert {rec['convert_s']} s (lock wait {rec['lock_wait_s']} s)  {'PASS' if not rec['reasons'] else '; '.join(rec['reasons'])}", flush=True)
            if not rec["reasons"]:
                stop = "pass"; break
            if a.no_ladder:
                stop = "no-ladder"; break
            if r >= 1 and rungs[r - 1]["kl_mean"] - rec["kl_mean"] < a.plateau * rungs[r - 1]["kl_mean"]:
                stop = f"plateau: KL {rungs[r - 1]['kl_mean']} -> {rec['kl_mean']} (< {a.plateau:.0%})"; break
            sens = json.loads((rdir / "pack/conv_sens.json").read_text())
            nxt, raised = next_cells(sens, a.k_max, a.step_frac, a.max_bpw)
            if rec["head_proxy"] > HEAD_PROXY_MAX and head_bits < 8:               # the head tier comes first, but 6 bits leave a proxy ~1e-4: never fires on the drill model
                head_bits += 1
            if nxt is None and head_bits == rec["head_bits"]:
                stop = f"size cap: no cell can be raised (max {a.max_bpw} bpw / K{a.k_max})"; break
            strategy = nxt or strategy
            print(f"[ladder] rung {r + 1}: +1 bit on {len(raised)} cells: " + " ".join(f"{li}{g[0]}->{k}" for li, g, k in raised[:40]), flush=True)
        else:
            stop = f"max rungs ({a.max_rungs})"
        best = min(rungs, key=lambda x: (bool(x["reasons"]), x["kl_mean"])) if stop != "pass" else rungs[-1]
        if out.is_symlink() or out.exists():
            out.unlink() if out.is_symlink() else shutil.rmtree(out)
        out.symlink_to(work / "rungs" / f"r{best['rung']}" / "pack")
        waits = sum(x["lock_wait_s"] for x in rungs)
        return dict(ok=True, stop=stop, passed=stop == "pass", final_rung=best["rung"], body_bpw=best["body_bpw"], rungs=len(rungs), lock_wait_s=round(waits, 1),
                    work_s=round(sum(x["work_s"] for x in rungs), 1), recipe=f"K{a.k_attn}/K{a.k_mlp}/experts K{a.k_expert}/head{a.head_bits} then ladder")
    step("ladder", ladder)

    def qualify():
        mem_ok()
        # qualify.sh takes the GPU lock per job itself
        rc = run(["bash", str(REPO / "tools/qualify.sh"), "--pack", str(out), "--ref", str(src), "--out", str(work / "qualify"),
                  *a.qualify_args.split()], work / "qualify.log", lock=False)
        q = work / "qualify/qualify.json"
        return dict(ok=rc == 0 and q.exists(), rc=rc)
    step("qualify", qualify)
    print(json.dumps({k: v["wall_s"] for k, v in st["steps"].items()}))


if __name__ == "__main__":
    main()

"""REPORT.md + qualify.json from the stage data. The lanes read qualify.json: serve.cmd / serve.kv / serve.ctx need no hand flags."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import rules
from plan import KIND

SCHEMA = 1


def assemble(meta: dict, args: dict, stages: dict, repo: Path) -> dict:
    """stages: name -> {status, wall_s, data}. Builds the qualify.json document."""
    d = {n: s.get("data") for n, s in stages.items() if s.get("status") == "done"}
    ctxs = {k: v for k, v in (d.get("maxctx") or {}).items() if k.isdigit()}
    sv = d.get("serve") or {}
    sm = d.get("smoke")
    q = dict(autotune=d.get("autotune"), parity=d.get("parity"), kv=None, maxctx=ctxs or None, serve=sv or None, smoke=sm)
    kvd = d.get("kv")
    if not kvd and meta.get("kv_mode") == "native":      # no KV bits to choose: not "unmeasured"
        kvd = dict(chosen=[16, 16], native=True, rule="native cache of the family (sliding ring + compressed pools / MLA): no quantized-KV choice", table={}, kl_max=None)
    q["kv"] = kvd
    kv = (kvd or {}).get("chosen") or [6, 6]
    try:
        git = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None
    return dict(
        schema=SCHEMA, at=time.strftime("%F %T"), git=git, backend=args["backend"],
        pack={k: meta[k] for k in ("name", "arch", "family", "layers", "kv_layers", "kv_heads", "head_dim", "quant", "bpw", "head_bits", "index_sha", "license", "mtp_layers", "vision")},
        machine=(d.get("meta") or {}).get("machine") or meta.get("machine"), scale=args.get("scale"),
        stages={n: dict(status=s.get("status"), wall_s=s.get("wall_s")) for n, s in stages.items()},
        serve=dict(cmd=(d.get("serve") or {}).get("cmd") or (d.get("smoke") or {}).get("cmd"), env=(d.get("serve") or {}).get("env") or (d.get("smoke") or {}).get("env"),
                   kv=kv, ctx=ctxs, draft_policy=meta.get("draft_policy"), template=meta["template"]),
        autotune=d.get("autotune"),
        quality=dict(parity=d.get("parity"), kv=kvd),
        speed={k: sv.get(k) for k in ("decode_tps_median", "prefill_8k_tps", "prefill_tokens", "curve", "agg", "footprint_gib")} if sv else None,
        sha=sv.get("sha"), smoke=sm,
        verdict={**rules.verdict(q, args.get("gates")), "pass": rules.verdict(q, args.get("gates"))["pass_"]},
    )


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(j: dict) -> str:
    p, v = j["pack"], j["verdict"]
    m = j.get("machine") or {}
    L = [f"# qualify {p['name']} on {j['backend']} — {'PASS' if v['pass'] else 'NOT PASSED'}", "",
         f"{p['arch']} ({p['family']}), {p['layers']} layers ({p['kv_layers']} with KV), {p['quant']} {p['bpw']} bpw, git {j['git']}, {j['at']}",
         f"machine: {m.get('chip')}, {m.get('mem_gib')} GiB, power {m.get('power')} (battery {m.get('battery_pct')} %, low power mode {m.get('low_power_mode')}), scale {j.get('scale')}", ""]
    if v["reasons"]:
        L += ["Failed gates: " + "; ".join(v["reasons"])]
    if v["unmeasured"]:
        L += ["Unmeasured (not counted as pass): " + ", ".join(v["unmeasured"])]
    L += ["", "## Stages", "| stage | status | wall s | kind |", "|---|---|---|---|"] + [f"| {n} | {s['status']} | {fmt(s['wall_s'], 0)} | {KIND.get(n, '')} |" for n, s in j["stages"].items()]
    at = j.get("autotune")
    if at:
        L += ["", f"## Autotune ({at['device']}, {at['tuned_shapes']}/{at['shapes']} shapes differ from the constants, {at['wall_s']:.0f} s, peak {at['peak_gib']} GiB)", f"table: {at['file']} (kernel version {at['kernel_version']}); "
              f"a choice replaces the constant only when its min over {at['rounds']} interleaved rounds is >= {at['min_gain']:.0%} faster", "",
              "| shape (in x out x K) | x | decode ms (default -> tuned) | k-splits | prefill rows: ms default -> tuned | differs from the constants |", "|---|---|---|---|---|---|"]
        L += [f"| {g['shape']} | {g['count']} | {fmt(g['decode_ms_default'], 4)} -> {fmt(g['decode_ms_tuned'], 4)} | {g['tgk'][0]} -> {g['tgk'][1]} | "
              + ("; ".join(f"{R}: {v[0]} -> {v[1]} ({v[2]} -> {v[3]})" for R, v in g["prefill"].items()) or "-") + f" | {', '.join(g['changes']) or 'nothing'} |" for g in at["gains"]]
    par = j["quality"]["parity"]
    if par:
        L += ["", "## Parity vs streamed BF16", f"mean KL {par['kl_mean']:.4f}, top-1 {par['top1_mean']:.4f}, dNLL {par['dnll_mean']:+.4f}; SQNR: {par['sqnr']}", "",
              "| domain | KL | top-1 | dNLL |", "|---|---|---|---|"] + [f"| {d} | {x['kl']:.4f} | {x['top1']:.4f} | {x['dnll']:+.4f} |" for d, x in par["by_domain"].items()]
    kv = j["quality"]["kv"]
    if kv:
        L += ["", f"## KV (chosen k{kv['chosen'][0]}v{kv['chosen'][1]}; {kv['rule']}, kl_max {kv['kl_max']})", "| bits | KL vs fp16 cache | top-1 vs fp16 cache | KL vs k8v8 |", "|---|---|---|---|"]
        L += [f"| k{b}v{b} | {x['kl_ref']:.4f} | {x['top1_ref']:.4f} | {fmt(x.get('kl_vs_k8'), 4)} |" for b, x in kv["table"].items()]
    if j["serve"]["ctx"]:
        L += ["", "## Max context (k%dv%d, no OOM on 2 distinct near-full prompts)" % tuple(j["serve"]["kv"]), "| sessions | max -c |", "|---|---|"] + [f"| {s} | {c} |" for s, c in j["serve"]["ctx"].items()]
    sp = j["speed"]
    if sp:
        L += ["", f"## Speed: decode {fmt(sp['decode_tps_median'],1)} tok/s, prefill 8K {fmt(sp['prefill_8k_tps'],0)} tok/s", "", "| prompt tok | ttft s | prefill tok/s | decode tok/s |", "|---|---|---|---|"]
        L += [f"| {r['prompt']} | {r['ttft']} | {fmt(r['prefill_tps'],0)} | {fmt(r['dec_tps'],1)} |" for r in sp["curve"]]
        L += ["", "| N | agg decode tok/s | per user | ttft s |", "|---|---|---|---|"] + [f"| {a['n']} | {fmt(a['agg_dec_tps'],1)} | {a['dec_tps']} | {a['ttft']} |" for a in sp["agg"]]
    if j["sha"]:
        L += ["", "Greedy sha: " + ", ".join(x["sha"] for x in j["sha"])]
    if j["smoke"]:
        L += ["", f"## Real-use smoke ({j['smoke']['kind']}): {j['smoke']['passed']}/{j['smoke']['total']} checks passed"]
        L += [f"- {x['name']}: {'ok' if x['ok'] else 'FAIL'}" + (f" ({x['value']:.3f})" if isinstance(x.get('value'), float) else "") for x in j["smoke"].get("checks", [])]
    env = " ".join(f"{k}={v}" for k, v in (j["serve"].get("env") or {}).items())
    L += ["", "## Serve command (no hand flags)", "`" + (env + " " if env else "") + " ".join(j["serve"]["cmd"] or []) + "`"]
    return "\n".join(L) + "\n"


def write(out: Path, j: dict) -> None:
    (out / "qualify.json").write_text(json.dumps(j, indent=1, default=str))
    (out / "REPORT.md").write_text(render_md(j))

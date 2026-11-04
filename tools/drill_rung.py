#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One rung of the drill's bpw ladder, run INSIDE the GPU lock as ONE job (a lock queue is minutes to an hour; convert + parity must not queue twice):
convert (conv_mlx.py --cache: only the blocks whose bits changed are quantised) -> eval docs (the pack's own tokenizer) -> BF16 reference (once) -> parity.
usage: drill_rung.py --src SRC --pack OUT --cal CAL --cache DIR --strategy S.json --k-attn K --k-mlp K --head-bits K --docs DOCS.json --ref REFDIR --result R.json [--doc-tokens 512]
Writes R.json: seconds per phase, the conv stage times, parity {kl_mean, top1_mean, dnll_mean, by_domain}."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sh(cmd: list, log: Path) -> float:
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO}/tools:{REPO}/tools/qualify", PYTHONUNBUFFERED="1")
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise SystemExit(f"{cmd[1]} failed rc={rc} (see {log})")
    return round(time.time() - t0, 1)


def aggregate(res: dict) -> dict:
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    tot = sum(v["tokens"] for v in res.values())
    w = lambda k: sum(v[k] * v["tokens"] for v in res.values()) / tot    # noqa: E731  same weighting as qualify stages.aggregate_parity
    return dict(kl_mean=w("kl"), top1_mean=w("top1"), dnll_mean=w("dnll"), by_domain=res)


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("src", "pack", "cal", "cache", "strategy", "docs", "ref", "result"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k-attn", type=int, required=True); ap.add_argument("--k-mlp", type=int, required=True); ap.add_argument("--k-expert", type=int, default=3); ap.add_argument("--head-bits", type=int, required=True)
    ap.add_argument("--doc-tokens", type=int, default=512)
    a = ap.parse_args()
    py, pack, res = sys.executable, Path(a.pack), {}
    log = Path(a.result).with_suffix(".log")
    res["convert_s"] = sh([py, REPO / "tools/conv_mlx.py", a.src, pack, "--cal", a.cal, "--cache", a.cache, "--strategy", a.strategy, "--k-attn", a.k_attn,
                          "--k-mlp", a.k_mlp, "--k-expert", a.k_expert, "--head-bits", a.head_bits], log)
    docs, ref = Path(a.docs), Path(a.ref)
    if not docs.exists():
        res["docs_s"] = sh([py, REPO / "tools/qualify/docs.py", "short", pack, docs, a.doc_tokens], log)
    if not (ref / "ref_wiki.npz").exists():
        res["ref_s"] = sh([py, REPO / "tools/qualify/refs/qwen35.py", a.src, docs, ref], log)
    raw = Path(a.result).with_suffix(".parity.json")
    res["parity_s"] = sh([py, REPO / "tools/qualify/mlx_eval.py", "parity", pack, docs, ref, raw], log)
    res["parity"] = aggregate(json.loads(raw.read_text()))
    last = [l for l in log.read_text().splitlines() if "STAGES" in l][-1:]
    res["conv_stages"] = last[0].split("STAGES ", 1)[1] if last else None
    Path(a.result).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

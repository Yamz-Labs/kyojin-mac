"""Autotune A/B driver: alternates the arms cold / tuned (one process each, same lock hold), `rounds` times, and prints min / p25 per metric plus the ratio tuned/cold.
  ab_tuning_drive.py PACK OUT.json [--rounds 3] [--reps 5] [--prefill 128,512,2048] [--decode 128]    run it under lockf -k <gpu lock>"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--prefill", default="128,512,2048"); ap.add_argument("--decode", type=int, default=128)
a = ap.parse_args()
here = Path(__file__).parent
runs = {"cold": [], "tuned": []}
for r in range(a.rounds):
    for arm in (("cold", "tuned") if r % 2 == 0 else ("tuned", "cold")):
        p = subprocess.run([sys.executable, "-u", str(here / "ab_tuning.py"), arm, a.pack, "--reps", str(a.reps), "--prefill", a.prefill, "--decode", str(a.decode)],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            sys.exit(f"arm {arm} failed: {(p.stderr or p.stdout)[-500:]}")
        runs[arm].append(json.loads(line[7:]))
        print(f"round {r} {arm}: " + " ".join(f"{k}={v['median']:.1f}" for k, v in runs[arm][-1].items() if isinstance(v, dict) and "median" in v), flush=True)
metrics = [k for k in runs["cold"][0] if isinstance(runs["cold"][0][k], dict) and "median" in runs["cold"][0][k]]
summ = {}
for m in metrics:
    row = {}
    for arm in runs:
        row[arm] = dict(min=min(x[m]["min"] for x in runs[arm]), best_p25=max(x[m]["p25"] for x in runs[arm]), best_median=max(x[m]["median"] for x in runs[arm]),
                        max=max(x[m]["max"] for x in runs[arm]))
    row["ratio_best_median"] = row["tuned"]["best_median"] / row["cold"]["best_median"]
    row["ratio_best_p25"] = row["tuned"]["best_p25"] / row["cold"]["best_p25"]
    row["ratio_max"] = row["tuned"]["max"] / row["cold"]["max"]
    summ[m] = row
same_tokens = all(x["tokens"] == runs["cold"][0]["tokens"] for arm in runs for x in runs[arm])
out = dict(summary=summ, tokens_identical=same_tokens, logit_sum={arm: [x["logit_sum"] for x in runs[arm]] for arm in runs}, tuning=runs["tuned"][0]["tuning"],
           peak_gib=max(x["peak_gib"] for arm in runs for x in runs[arm]), runs=runs)
Path(a.out).write_text(json.dumps(out, indent=1))
for m, r in summ.items():
    print(f"{m:>14}: cold best-median {r['cold']['best_median']:.1f} -> tuned {r['tuned']['best_median']:.1f}  x{r['ratio_best_median']:.3f} (p25 x{r['ratio_best_p25']:.3f}, max x{r['ratio_max']:.3f})")
print(f"greedy tokens identical: {same_tokens}; logit sums cold {out['logit_sum']['cold']} tuned {out['logit_sum']['tuned']}; tuning {out['tuning']['status']}; peak {out['peak_gib']} GiB")

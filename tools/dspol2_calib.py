"""Calibration error of the dspol_v2 policy from a round_log trace (scratch/roundlog.jsonl).

usage: python -I tools/dspol2_calib.py TRACE.jsonl [--ph PROMPT_TAG] [--mode 1|2] [--refit]
(--mode keeps the records the dspol_v2 mode wrote: the trace of an interleaved run holds both arms)

Prints, per draft position and per predicted-p bin, the mean p_k the policy used against the realised accept rate over the rounds that
reached the position, the mean E[T] it expected against the tokens realised, and the plain / verify round counts with their tokens per ms.
--refit prints the logistic prior (a, b) fitted on the trace's own confidences (KYOJIN_DSPOL_V2_PRIOR="a,b" feeds it back)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import dspol2 as D      # noqa: E402


def main(argv):
    path = argv[1]
    ph = argv[argv.index("--ph") + 1] if "--ph" in argv else None
    mode = int(argv[argv.index("--mode") + 1]) if "--mode" in argv else None
    recs = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and '"k"' in line:
            r = json.loads(line)
            if (ph is None or r.get("ph") == ph) and (mode is None or r.get("v2") == mode):
                recs.append(r)
    rep = D.calibration_report(recs)
    print(f"rounds with pk: {rep['et']['rounds']}   E[T] policy {rep['et']['pred']:.3f}  realised {rep['et']['real']:.3f}   ECE {rep['ece']:.3f}")
    for i, e in rep["pos"].items():
        print(f"  position {i}: n={e['n']:5d}  pred {e['pred']:.3f}  real {e['real']:.3f}  err {e['pred'] - e['real']:+.3f}")
    for e in rep["bins"]:
        print(f"  bin pred~{e['pred']:.2f}: n={e['n']:5d}  real {e['real']:.3f}  err {e['pred'] - e['real']:+.3f}")
    for kind, name in (("r", "verify"), ("p", "plain")):
        rr = [r for r in recs if r.get("k") == kind]
        tok, ms = sum(r.get("tok") or 0 for r in rr), sum(r.get("ms") or 0.0 for r in rr)
        if rr and ms:
            print(f"  {name}: {len(rr)} rounds, {tok / len(rr):.2f} tok/round, {ms / len(rr):.1f} ms/round, {1000 * tok / ms:.1f} tok/s")
    rows = {}
    for r in recs:
        if r.get("k") == "r" and "R" in r:
            rows[r["R"]] = rows.get(r["R"], 0) + 1
    if rows:
        print("  chosen R (verify rounds):", dict(sorted(rows.items())))
    if "--refit" in argv:
        a, b = D.fit_prior(D.samples_from_trace(recs))
        print(f"  refit prior: KYOJIN_DSPOL_V2_PRIOR={a:.3f},{b:.3f}   (default {D.PRIOR_A},{D.PRIOR_B})")


if __name__ == "__main__":
    main(sys.argv)

"""Engine A/B on the fake backends of tests/test_serve_spec.py (CPU, mlx only, no model): the same scenarios under any tree, every lever at its default.

Run it from two checkouts (perf/integ and the merged tree) and diff the JSON: `python tools/cmerge_engine_ab.py > out.json`. It prints, per scenario, the tokens and finish
reason of every request and the engine counters, minus the always-present zero counters the merged levers add (those are listed under "new_keys", not compared).
One request per run (sessions=1), so no thread timing enters; only integer counters are compared (seconds are floats)."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
for k in list(os.environ):
    if k.startswith("KYOJIN_"):
        del os.environ[k]
os.environ["KYOJIN_ROUND_LOG"] = "0"

from test_serve_spec import FakeSpec, prompts, run  # noqa: E402

NEW = {"mix_slice_rounds", "t_pf_inline_decode", "share_hits", "share_tokens", "share_seeds", "share_takers", "share_refused_budget", "kv_drops", "rebuilds", "kv_parks",
       "kv_refused", "kv_over_budget", "tok_gap_max_s", "tok_gap_pf_max_s", "kv_digest_check", "kv_digest_mismatch"}


def clean(st: dict) -> dict:
    return {k: v for k, v in sorted(st.items()) if isinstance(v, int) and not isinstance(v, bool) and k not in NEW}


def main() -> int:
    out, new_seen = {}, set()
    reqs = prompts(5)
    for name, kw in (("plain_prefix", dict(prefix_bytes=1 << 20)), ("plain_noprefix", dict(prefix_bytes=0)), ("spec_d3", dict(mtp_depth=3, prefix_bytes=1 << 20)),
                     ("spec_d5_wrong", dict(mtp_depth=5, prefix_bytes=1 << 20))):
        for n in sorted(reqs):
            got, st, _ = run({n: reqs[n]}, FakeSpec(err=3 if "wrong" not in name else 1) if name.startswith("spec") else None, sessions=1, **kw)
            out[f"{name}/{n}"] = {"tokens": [got[n][0], got[n][1]], "stats": clean(st)}
            new_seen |= (NEW & set(st))
    shared = [(list(range(3, 40)) + [50 + i, 60 + i, 70], dict(max_new=12)) for i in range(4)]
    for i, r in enumerate(shared):                       # the prefix cache: each run starts from a fresh engine, so this checks the snapshot path inside one request
        got, st, _ = run({"s": r}, None, sessions=1, prefix_bytes=1 << 20, chunk=4, block=8)
        out[f"shared_prefix/{i}"] = {"tokens": [got["s"][0], got["s"][1]], "stats": clean(st)}
    out["new_keys_present"] = sorted(new_seen)
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
